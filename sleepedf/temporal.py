"""Soft temporal staging objective with an exact, untruncated duration tail.

All costs use natural logarithms. Durations count 30-second epochs. The finite
duration table ends at a bin boundary, not a maximum permitted sleep-run length.
These priors and the decoder are research components, not adopted improvements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import content_id
from .protocol import development_records, load_development_truth

def fit_priors(sequences: list[np.ndarray], *, cutoff: int = 60) -> dict:
    """Estimate positive priors; invalid labels break adjacency and censor runs.

Transitions include self transitions. Duration fitting uses only runs bounded
on both sides by a different valid stage, avoiding edge/gap-censored lengths.
Dirichlet(1) smooths transition rows and duration bins. A Beta(1,29) hazard
prior gives every stage a nonzero geometric tail even without observed tails.
The caller must enforce the training-participant boundary.
"""
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("Duration bin cutoff must be positive")
    transitions = np.zeros((5, 5), np.int64)
    durations = np.zeros((5, cutoff + 1), np.int64)
    excess_sum = np.zeros(5, np.int64)
    valid_epochs = 0
    if not sequences:
        raise ValueError("Training sequences are required")
    for values in sequences:
        y = np.asarray(values)
        if y.ndim != 1 or y.dtype.kind not in "iu" or np.any((y < -1) | (y > 4)):
            raise ValueError("Training labels require fixed classes or invalid -1")
        valid_epochs += int(np.count_nonzero(y >= 0))
        pairs = (y[:-1] >= 0) & (y[1:] >= 0)
        np.add.at(transitions, (y[:-1][pairs], y[1:][pairs]), 1)
        start = 0
        while start < len(y):
            end = start + 1
            while end < len(y) and y[end] == y[start]:
                end += 1
            stage = int(y[start])
            if stage >= 0 and start > 0 and end < len(y) and y[start-1] >= 0 and y[end] >= 0:
                length = end - start
                durations[stage, min(length-1, cutoff)] += 1
                if length > cutoff:
                    excess_sum[stage] += length - cutoff
            start = end
    if not valid_epochs:
        raise ValueError("Temporal priors require observed valid training labels")
    trans = (transitions + 1) / (transitions.sum(axis=1, keepdims=True) + 5)
    mass = (durations + 1) / (durations.sum(axis=1, keepdims=True) + cutoff + 1)
    hazard = (durations[:, -1] + 1) / (excess_sum + 30)
    return {"schema_version": "1.0", "cutoff_epochs": cutoff, "epoch_seconds": 30,
            "observed_valid_training_epochs": valid_epochs,
            "transition_probability": trans.tolist(), "duration_probability": mass[:, :-1].tolist(),
            "tail_mass": mass[:, -1].tolist(), "tail_hazard": hazard.tolist(),
            "transition_counts": transitions.tolist(), "duration_counts": durations.tolist(),
            "duration_tail_excess_sum": excess_sum.tolist(),
            "censoring": "exclude runs touching recording boundaries or invalid intervals",
            "duration_target": "observed complete interior runs; conditional on observability",
            "edge_duration_assumption": "same complete-run cost at inference boundaries; not censor-aware",
            "smoothing": {"transition_dirichlet": 1, "duration_dirichlet": 1,
                          "tail_hazard_beta": [1, 29]}}


def fit_fold_priors(root: Path, fold_id: int, *, cutoff: int = 60) -> dict:
    """Read only the original development fold's training reference labels."""
    protocol, split, records = development_records(root)
    if type(fold_id) is not int or not 0 <= fold_id < 5:
        raise ValueError("Unknown development fold")
    allowed = set(split["folds"][fold_id]["train"])
    selected = [r for r in records if r["participant_id"] in allowed]
    if len(allowed) != 48 or {r["participant_id"] for r in selected} != allowed:
        raise ValueError("Training fold is incomplete")
    sequences = [load_development_truth(r, split)["reference_label"] for r in selected]
    priors = fit_priors(sequences, cutoff=cutoff)
    priors.update(protocol_hash=protocol["protocol_hash"], fold_id=fold_id,
                  train_participants=sorted(allowed), source_recordings=[r["recording_id"] for r in selected],
                  training_truth_sha256={r["recording_id"]: r["frozen_truth"] for r in selected})
    priors["prior_hash"] = content_id(priors)
    return priors


def _costs(probabilities: np.ndarray, priors: dict):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or np.any(p < 0) or
            np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Emissions require genuine normalized five-class probabilities")
    trans = np.asarray(priors["transition_probability"], dtype=np.float64)
    duration = np.asarray(priors["duration_probability"], dtype=np.float64)
    tail = np.asarray(priors["tail_mass"], dtype=np.float64)
    hazard = np.asarray(priors["tail_hazard"], dtype=np.float64)
    cutoff = priors["cutoff_epochs"]
    if (type(cutoff) is not int or cutoff < 1 or trans.shape != (5, 5) or
            duration.shape != (5, cutoff) or tail.shape != (5,) or hazard.shape != (5,) or
            any(not a.size or not np.isfinite(a).all() or np.any(a <= 0) or np.any(a >= 1)
                for a in (trans, duration, tail, hazard)) or
            not np.allclose(trans.sum(axis=1), 1, rtol=0, atol=1e-10) or
            not np.allclose(duration.sum(axis=1) + tail, 1, rtol=0, atol=1e-10)):
        raise ValueError("Temporal priors must be strictly positive normalized distributions")
    p = p / p.sum(axis=1, keepdims=True)
    emission = np.full_like(p, np.inf)
    np.log(p, out=emission, where=p > 0)
    emission[p > 0] *= -1
    return emission, -np.log(trans), -np.log(duration), -np.log(tail)-np.log(hazard), -np.log1p(-hazard)


def objective(probabilities: np.ndarray, labels: np.ndarray, priors: dict, *,
              transition_weight: float, duration_weight: float) -> float:
    """Independently score one continuous sequence for diagnostics/enumeration."""
    emission, trans, duration, tail, slope = _costs(probabilities, priors)
    _weights(transition_weight, duration_weight)
    y = np.asarray(labels)
    if y.shape != (len(emission),) or y.dtype.kind not in "iu" or np.any((y < 0) | (y > 4)):
        raise ValueError("Invalid decoded sequence")
    value = float(emission[np.arange(len(y)), y].sum())
    value += transition_weight * float(trans[y[:-1], y[1:]].sum())
    start = 0
    cutoff = duration.shape[1]
    while start < len(y):
        end = start + 1
        while end < len(y) and y[end] == y[start]:
            end += 1
        length, stage = end-start, int(y[start])
        cost = duration[stage, length-1] if length <= cutoff else tail[stage] + (length-cutoff-1)*slope[stage]
        value += duration_weight * float(cost)
        start = end
    return value


def _weights(transition_weight, duration_weight):
    if any(not np.isfinite(v) or v < 0 for v in (transition_weight, duration_weight)):
        raise ValueError("Temporal weights must be finite and nonnegative")


def _decode_block(emission, trans, duration, tail, slope, lam, mu):
    """Exact segment DP, O(T*K*(D+K)) time and O(T*K) space, K=5.

For d>D the duration cost is affine in d; a running minimum covers *all*
earlier run starts. No duration or physiological transition is forbidden.
    Within a fixed ending stage, ties prefer shorter last runs then the lower
    predecessor class. Final-stage ties prefer the lower class ID first.
"""
    n, cutoff = len(emission), duration.shape[1]
    impossible = np.vstack([np.zeros(5, dtype=np.int64), np.cumsum(~np.isfinite(emission), axis=0)])
    prefix = np.vstack([np.zeros(5), np.cumsum(np.where(np.isfinite(emission), emission, 0), axis=0)])
    best = np.full((n+1, 5), np.inf)
    boundary = np.full_like(best, np.inf)
    previous = np.full((n+1, 5), -1, np.int8)
    starts = np.zeros((n+1, 5), np.int64)
    boundary[0] = 0
    tail_best = np.full(5, np.inf)
    tail_start = np.zeros(5, np.int64)
    linear = lam*np.diag(trans) + mu*slope
    constant = -lam*np.diag(trans) + mu*(tail-(cutoff+1)*slope)
    for end in range(1, n+1):
        tail_best[~np.isfinite(emission[end-1])] = np.inf
        eligible = end-cutoff-1
        if eligible >= 0:
            candidate = boundary[eligible]-prefix[eligible]-linear*eligible
            candidate[impossible[end] != impossible[eligible]] = np.inf
            update = candidate <= tail_best  # Latest tied start gives shorter tail run.
            tail_best[update] = candidate[update]
            tail_start[update] = eligible
        lengths = np.arange(1, min(end, cutoff)+1)
        begin = end-lengths
        options = (boundary[begin]-prefix[begin]+prefix[end] +
                   lam*(lengths[:, None]-1)*np.diag(trans) + mu*duration[:, lengths-1].T)
        options[impossible[end] != impossible[begin]] = np.inf
        for stage in range(5):
            pick = int(np.argmin(options[:, stage]))
            best[end, stage] = options[pick, stage]
            starts[end, stage] = begin[pick]
            tail_option = prefix[end, stage] + linear[stage]*end + constant[stage] + tail_best[stage]
            if tail_option < best[end, stage]:
                best[end, stage] = tail_option
                starts[end, stage] = tail_start[stage]
        for stage in range(5):
            incoming = best[end] + lam*trans[:, stage]
            incoming[stage] = np.inf  # Adjacent same-stage runs are a single maximal run.
            parent = int(np.argmin(incoming))
            boundary[end, stage] = incoming[parent]
            previous[end, stage] = parent
    stage = int(np.argmin(best[n]))
    score = float(best[n, stage])
    if not np.isfinite(score):
        raise ValueError("Temporal objective has no numerically finite optimum")
    labels = np.empty(n, np.int8)
    end = n
    while end:
        begin = int(starts[end, stage])
        labels[begin:end] = stage
        stage = int(previous[begin, stage])
        end = begin
    return labels, score


def decode(probabilities: np.ndarray, priors: dict, *, transition_weight: float = 0,
           duration_weight: float = 0, block_start: np.ndarray | None = None) -> dict:
    """Decode full signal-derived blocks without any reference-label argument.

``block_start`` comes from acquisition gaps/record boundaries, never Hypnogram
validity. Each input epoch retains its position. Original emission probabilities
are not relabeled as decoder posterior probabilities.
"""
    costs = _costs(probabilities, priors)
    _weights(transition_weight, duration_weight)
    n = len(costs[0])
    boundaries = np.zeros(n, bool) if block_start is None else np.asarray(block_start)
    if boundaries.shape != (n,) or boundaries.dtype != bool:
        raise ValueError("Signal block starts must be a boolean vector on the full epoch grid")
    starts = np.unique(np.r_[0, np.flatnonzero(boundaries), n])
    labels = np.empty(n, np.int8)
    total = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        if transition_weight == 0 and duration_weight == 0:
            decoded = np.asarray(probabilities)[a:b].argmax(axis=1).astype(np.int8)
            score = float(costs[0][np.arange(a, b), decoded].sum())
        else:
            decoded, score = _decode_block(costs[0][a:b], *costs[1:], transition_weight, duration_weight)
        labels[a:b] = decoded
        total += score
    return {"hard_label": labels, "objective": total, "zero_emission_policy": "infinite_cost",
            "duration_truncated": False, "block_count": max(len(starts)-1, 0)}
