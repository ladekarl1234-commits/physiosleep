"""Fixed five-class pooled staging metrics and paired participant uncertainty."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from .predictions import load_prediction, load_truth


def macro_f1_fraction(confusion: np.ndarray) -> Fraction:
    matrix = np.asarray(confusion)
    if matrix.shape != (5, 5) or matrix.dtype.kind not in "iu" or np.any(matrix < 0):
        raise ValueError("confusion must be a nonnegative integer 5x5 matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    return sum((Fraction(2 * int(matrix[i, i]), int(support[i] + predicted[i]))
                if support[i] + predicted[i] else Fraction(0) for i in range(5)), Fraction(0)) / 5


def _macro_f1_float(confusion: np.ndarray) -> float:
    matrix = np.asarray(confusion)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                     where=denominator != 0).mean(axis=-1)


def _metrics(confusion: np.ndarray) -> dict:
    cm = np.asarray(confusion, dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    f1 = [float(Fraction(2 * int(cm[i, i]), int(support[i] + predicted[i]))
                if support[i] + predicted[i] else Fraction(0)) for i in range(5)]
    precision = [int(cm[i, i]) / int(predicted[i]) if predicted[i] else 0.0 for i in range(5)]
    recall = [int(cm[i, i]) / int(support[i]) if support[i] else 0.0 for i in range(5)]
    score = macro_f1_fraction(cm)
    total = int(cm.sum())
    actual = int(np.trace(cm))
    expected_numerator = int(np.dot(support, predicted))
    kappa_denominator = total * total - expected_numerator
    kappa = float(Fraction(actual * total - expected_numerator, kappa_denominator)) if kappa_denominator else None
    return {"confusion": cm.tolist(), "support": support.tolist(),
            "predicted_support": predicted.tolist(), "per_class_precision": precision,
            "per_class_recall": recall, "undefined_class_metric_policy": "zero",
            "per_class_f1": f1, "macro_f1": float(score),
            "macro_f1_exact": f"{score.numerator}/{score.denominator}",
            "accuracy": actual / total if total else None, "kappa": kappa,
            "kappa_unavailable_reason": None if kappa_denominator else "zero_expected_disagreement",
            "evaluated_epochs": total}


def score_aligned_record(truth: dict, prediction: dict) -> tuple[np.ndarray, int]:
    """Require the entire original timeline, then score reference-valid epochs."""
    for key in ("participant_id", "recording_id"):
        if truth[key] != prediction[key]:
            raise ValueError(f"{key} differs between truth and prediction")
    for key in ("epoch_index", "onset_seconds"):
        if not np.array_equal(truth[key], prediction[key]):
            raise ValueError(f"{key} differs between truth and prediction")
    valid = truth["valid_mask"]
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (truth["reference_label"][valid], prediction["hard_label"][valid]), 1)
    return cm, int((~valid).sum())


def evaluate_saved_records(truth_paths: list[Path], prediction_paths: list[Path],
                           *, protocol_hash: str | None = None,
                           registry_hash: str | None = None,
                           model_id: str | None = None) -> dict:
    if not truth_paths or len(truth_paths) != len(prediction_paths):
        raise ValueError("truth and prediction record counts must match and be nonzero")
    truth = {}
    for path in truth_paths:
        metadata, record = load_truth(Path(path))
        rid = record["recording_id"]
        if rid in truth:
            raise ValueError("duplicate truth recording_id")
        truth[rid] = (metadata, record)
    predictions = {}
    for path in prediction_paths:
        metadata, record = load_prediction(Path(path))
        rid = record["recording_id"]
        if rid in predictions:
            raise ValueError("duplicate prediction recording_id")
        if protocol_hash is not None and metadata["protocol_hash"] != protocol_hash:
            raise ValueError("prediction protocol hash mismatch")
        if registry_hash is not None and metadata["registry_hash"] != registry_hash:
            raise ValueError("prediction registry hash mismatch")
        if model_id is not None and metadata["model_id"] != model_id:
            raise ValueError("prediction model_id mismatch")
        predictions[rid] = (metadata, record)
    if truth.keys() != predictions.keys():
        raise ValueError("recording set differs between truth and predictions")
    by_participant: dict[str, np.ndarray] = {}
    total_invalid = 0
    for rid in sorted(truth):
        _, reference = truth[rid]
        _, predicted = predictions[rid]
        cm, invalid = score_aligned_record(reference, predicted)
        pid = reference["participant_id"]
        by_participant.setdefault(pid, np.zeros((5, 5), dtype=np.int64))
        by_participant[pid] += cm
        total_invalid += invalid
    total_cm = sum(by_participant.values(), np.zeros((5, 5), dtype=np.int64))
    if any(not cm.sum() for cm in by_participant.values()):
        raise ValueError("each audit participant requires positive reference-valid support")
    result = _metrics(total_cm)
    result["invalid_epochs"] = total_invalid
    result["complete_psg_epochs"] = result["evaluated_epochs"] + total_invalid
    result["reference_valid_fraction"] = result["evaluated_epochs"] / result["complete_psg_epochs"]
    result["prediction_coverage_fraction"] = 1.0  # Missing grid entries already failed alignment.
    result["participant_count"] = len(by_participant)
    result["recording_count"] = len(truth)
    result["participant_macro_f1_mean"] = float(np.mean([
        float(macro_f1_fraction(cm)) for cm in by_participant.values()]))
    result["per_participant_confusion"] = {pid: cm.tolist() for pid, cm in sorted(by_participant.items())}
    return result


def paired_cluster_intervals(candidate: dict, comparators: dict[str, dict],
                             participant_cohort: dict[str, str], *, phase: str,
                             draws: int = 10_000, alpha: float = 0.05) -> dict:
    """Stratified paired bootstrap, drawing participants with all their nights."""
    if phase not in ("A", "B") or not comparators or type(draws) is not int or draws < 1:
        raise ValueError("phase must be A/B and draws/comparators nonempty")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    pids = sorted(candidate["per_participant_confusion"])
    if set(pids) != set(participant_cohort):
        raise ValueError("participant cohort mapping does not match candidate")
    if any(cohort not in ("SC", "ST") for cohort in participant_cohort.values()):
        raise ValueError("participant cohort must be SC or ST")
    for name, item in comparators.items():
        if set(item["per_participant_confusion"]) != set(pids):
            raise ValueError(f"comparator {name} participant set differs")
    names = sorted(comparators)
    systems = [candidate] + [comparators[name] for name in names]
    matrices = np.array([[item["per_participant_confusion"][pid] for pid in pids]
                         for item in systems], dtype=np.int64)
    strata = [[i for i, pid in enumerate(pids) if participant_cohort[pid] == cohort]
              for cohort in ("SC", "ST")]
    strata = [indices for indices in strata if indices]
    seed = 2026092301 if phase == "A" else 2026092302
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.empty((draws, len(names)), dtype=np.float64)
    for draw in range(draws):
        selected = np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                                   for indices in strata])
        pooled = matrices[:, selected].sum(axis=1)
        scores = _macro_f1_float(pooled)
        differences[draw] = scores[0] - scores[1:]
    tail = alpha / (2 * len(names))
    intervals = {}
    for i, name in enumerate(names):
        low, high = np.quantile(differences[:, i], [tail, 1 - tail], method="linear")
        intervals[name] = {"lower": float(low), "upper": float(high), "width": float(high - low)}
    return {"method": "paired_participant_cluster_stratified_SC_ST_percentile",
            "rng": "PCG64", "seed": seed, "draws": draws, "alpha": alpha,
            "multiplicity": "Bonferroni", "quantile_method": "linear",
            "participant_count": len(pids), "cohort_counts": {cohort: sum(
                participant_cohort[pid] == cohort for pid in pids) for cohort in ("SC", "ST")},
            "intervals": intervals}
