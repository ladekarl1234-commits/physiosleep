"""Numerical kernel for registered precision planning, never a readiness receipt.

Inputs here are sufficient statistics. The future controller must independently
bind them to actual development predictions and verify every prerequisite.
"""
from __future__ import annotations

from fractions import Fraction
from math import lcm

import numpy as np

from .evaluation import macro_f1_fraction


def validate_matrices(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices)
    if values.ndim != 4 or values.shape[2:] != (5, 5) or min(values.shape[:2]) < 1:
        raise ValueError("Expected systems x participants x 5 x 5 confusion counts")
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("Confusion counts must be nonnegative integers")
    # Covers row/column totals and a full participant bootstrap with repetition.
    bound = np.iinfo(np.int64).max // (10 * values.shape[1])
    if np.any(values > bound):
        raise ValueError("Confusion counts exceed safe integer accumulation")
    values = values.astype(np.int64, copy=False)
    support = values.sum(axis=-1)
    if not np.all(support == support[:1]) or np.any(support[0].sum(axis=-1) == 0):
        raise ValueError("Systems must share reference support for every participant")
    return values


def bootstrap_counts(cohorts: list[str], *, draws: int = 10000, seed: int = 2026092301) -> np.ndarray:
    """Exactly reproduce the common evaluator's paired cohort draws in order."""
    if not cohorts or any(cohort not in ("SC", "ST") for cohort in cohorts):
        raise ValueError("Unknown or empty participant cohorts")
    if type(draws) is not int or draws < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid bootstrap draw count or seed")
    strata = [[i for i, value in enumerate(cohorts) if value == cohort] for cohort in ("SC", "ST")]
    strata = [members for members in strata if members]
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros((draws, len(cohorts)), dtype=np.int64)
    for draw in range(draws):
        selected = np.concatenate([rng.choice(members, size=len(members), replace=True) for members in strata])
        counts[draw] = np.bincount(selected, minlength=len(cohorts))
    return counts


def paired_statistics(matrices: np.ndarray, counts: np.ndarray, *, candidate_count: int,
                      alpha: float = 0.05) -> dict:
    """Return all candidate/comparator paired draws and exact point differences.

Candidate systems come first; all remaining systems are distinct comparators.
Participant order must already be canonical. This function performs no selection.
"""
    matrices = validate_matrices(matrices)
    systems, participants = matrices.shape[:2]
    if type(candidate_count) is not int or not 1 <= candidate_count < systems or not 0 < alpha < 1:
        raise ValueError("Candidates, comparators and a valid alpha are required")
    counts = np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[1] != participants or len(counts) < 1 or
        counts.dtype.kind not in "iu" or np.any(counts < 0) or
        np.any(counts > participants) or np.any(counts.sum(axis=1) != participants)):
        raise ValueError("Bootstrap counts must preserve participant sample size")
    counts = counts.astype(np.int64, copy=False)
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    sufficient = np.concatenate((diagonal, denominator), axis=-1)
    pooled = (counts @ sufficient.transpose(1, 0, 2).reshape(participants, systems * 10)).reshape(-1, systems, 10)
    scores = np.divide(2 * pooled[..., :5], pooled[..., 5:],
                       out=np.zeros(pooled.shape[:-1] + (5,), dtype=np.float64),
                       where=pooled[..., 5:] != 0).mean(axis=-1)
    differences = scores[:, :candidate_count, None] - scores[:, None, candidate_count:]
    points = [macro_f1_fraction(matrix) for matrix in matrices.sum(axis=1)]
    exact = [[candidate - comparator for comparator in points[candidate_count:]]
             for candidate in points[:candidate_count]]
    tail = alpha / (2 * (systems - candidate_count))
    intervals = np.quantile(differences, [tail, 1 - tail], axis=0, method="linear")
    return {"exact_point_scores": points, "exact_differences": exact,
            "bootstrap_scores": scores, "bootstrap_differences": differences,
            "lower": intervals[0], "upper": intervals[1], "tail": tail,
            "not_a_readiness_receipt": True}


def joint_flags(statistics: dict, shifts: list[Fraction] | None = None) -> dict:
    """Shifted flags are hypothetical; support violations never get clipped."""
    differences = statistics["exact_differences"]
    count = len(differences)
    if shifts is None:
        shifts = [Fraction(0)] * count
    if len(shifts) != count or any(not isinstance(shift, Fraction) for shift in shifts):
        raise ValueError("One exact statistic shift per candidate is required")
    point = [all(delta + shift >= Fraction(1, 50) for delta in row)
             for row, shift in zip(differences, shifts)]
    intervals = np.all(statistics["lower"] + np.asarray([float(shift) for shift in shifts])[:, None] > 0, axis=1)
    violations = []
    for i, shift in enumerate(shifts):
        scores = statistics["bootstrap_scores"][:, i] + float(shift)
        point_score = statistics["exact_point_scores"][i] + shift
        violations.append({"point": not 0 <= point_score <= 1,
                           "bootstrap_draws": int(((scores < 0) | (scores > 1)).sum())})
    supported = [not row["point"] and row["bootstrap_draws"] == 0 for row in violations]
    return {"point_margin_pass": point, "positive_interval_pass": intervals.tolist(),
            "joint_pass": [p and bool(interval) and support for p, interval, support in zip(point, intervals, supported)],
            "hypothetical": any(shift != 0 for shift in shifts), "support_violations": violations}


def block_standardized(matrices: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    """Scale the sum of block means to integer counts; F1 is scale invariant."""
    matrices = validate_matrices(matrices)
    members = [index for block in blocks for index in block]
    if (not blocks or any(not block for block in blocks) or
        any(type(index) is not int for index in members) or sorted(members) != list(range(matrices.shape[1]))):
        raise ValueError("Blocks must partition the observed development participants once")
    denominator = lcm(*(len(block) for block in blocks))
    if denominator * int(matrices.max()) * matrices.shape[1] * 10 > np.iinfo(np.int64).max:
        raise ValueError("Block standardization exceeds integer bounds")
    return sum((matrices[:, block].sum(axis=1) * (denominator // len(block)) for block in blocks),
               np.zeros((matrices.shape[0], 5, 5), dtype=np.int64))


def monte_carlo_lower(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    from scipy.stats import beta
    if (type(successes) is not int or type(trials) is not int or trials < 1 or
        not 0 <= successes <= trials or not 0 < alpha < 1):
        raise ValueError("Invalid binomial Monte Carlo counts")
    return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
