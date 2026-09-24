"""Pure joint-gate simulation component; no eligibility or readiness authority.

Callers must independently validate complete development evidence before using
real counts. Synthetic component tests do not run the registered planning study.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np

from .precision import bootstrap_counts, joint_flags, monte_carlo_lower, paired_statistics
from .precision_sampling import PlanningPopulation, location_shifts, select_matrices


@lru_cache(maxsize=1)
def _interval_counts() -> np.ndarray:
    counts = bootstrap_counts(["SC"] * 16 + ["ST"] * 4, draws=10000, seed=2026092301)
    counts.flags.writeable = False
    return counts


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def simulate_trial(matrices: np.ndarray, population: PlanningPopulation, *,
                   candidate_count: int, audit_index: int,
                   population_index: int | None = None) -> dict:
    """Apply the exact registered interval kernel to one paired pseudo-audit."""
    selected, ids = select_matrices(matrices, population, audit_index,
                                    population_index=population_index)
    statistics = paired_statistics(selected, _interval_counts(), candidate_count=candidate_count)
    observed = joint_flags(statistics)
    deltas = statistics["exact_differences"]
    shifted = {}
    for target in (Fraction(3, 100), Fraction(1, 25)):
        offsets = location_shifts(matrices, population, candidate_count=candidate_count,
                                  target=target, population_index=population_index)
        shifted[_fraction(target)] = {
            "offsets": [_fraction(offset) for offset in offsets],
            "flags": joint_flags(statistics, list(offsets)),
            "interpretation": "hypothetical_location_shift_not_observed_improvement",
        }
    return {
        "schema_version": "1.0", "artifact_type": "precision_simulation_trial_not_readiness",
        "population_index": population_index, "audit_index": audit_index,
        "participant_ids": list(ids), "candidate_count": candidate_count,
        "comparator_count": len(matrices) - candidate_count,
        "point_scores_exact": [_fraction(value) for value in statistics["exact_point_scores"]],
        "paired_differences_exact": [[_fraction(value) for value in row] for row in deltas],
        "lower": statistics["lower"].tolist(), "upper": statistics["upper"].tolist(),
        "interval_draws": 10000, "interval_seed": 2026092301,
        "interval_tail": statistics["tail"], "flags": observed,
        "bootstrap_candidate_score_bounds": np.stack([
            statistics["bootstrap_scores"][:, :candidate_count].min(axis=0),
            statistics["bootstrap_scores"][:, :candidate_count].max(axis=0)], axis=-1).tolist(),
        "individual_point_failures": [[value < Fraction(1, 50) for value in row] for row in deltas],
        "individual_interval_failures": (statistics["lower"] <= 0).tolist(),
        "hypothetical": shifted,
    }


def summarize_trials(records: list[dict], *, population_index: int | None = None) -> dict:
    """Summarize one complete registered population; never grant readiness.

The prerequisite controller and independent verifier must recompute these
records from bound matrices. A list of hand-written records is not evidence.
"""
    if population_index is not None and (type(population_index) is not int or
                                         not 0 <= population_index < 100):
        raise ValueError("Invalid population index")
    expected = 4000 if population_index is None else 256
    if (type(records) is not list or len(records) != expected or
            any(type(row) is not dict or type(row.get("audit_index")) is not int or
                row.get("audit_index") != index or
                row.get("population_index") != population_index or
                type(row.get("population_index")) is not type(population_index) or
                row.get("artifact_type") != "precision_simulation_trial_not_readiness"
                for index, row in enumerate(records))):
        raise ValueError("Require every registered audit in canonical order exactly once")
    candidates, comparators = records[0]["candidate_count"], records[0]["comparator_count"]
    if (type(candidates) is not int or candidates < 1 or
            type(comparators) is not int or comparators < 1 or
            any(row.get("candidate_count") != candidates or row.get("comparator_count") != comparators or
                row.get("interval_draws") != 10000 or row.get("interval_seed") != 2026092301 or
                row.get("interval_tail") != 0.05 / (2 * comparators) for row in records)):
        raise ValueError("Trial systems or registered interval contract differs")
    differences = np.asarray([[[float(Fraction(value)) for value in row]
                              for row in record["paired_differences_exact"]] for record in records])
    if differences.shape != (expected, candidates, comparators) or not np.isfinite(differences).all():
        raise ValueError("Trial paired differences are incomplete or nonfinite")
    for record in records:
        ids = record.get("participant_ids")
        if (type(ids) is not list or len(ids) != 20 or
                any(type(pid) is not str or len(pid) != 5 or pid[:3] not in ("SC:", "ST:") or
                    not pid[3:].isascii() or not pid[3:].isdigit() for pid in ids) or
                sorted(set(ids)) != ids or sum(pid.startswith("SC:") for pid in ids) != 16):
            raise ValueError("Trial participant composition or canonical order differs")
        scores = [Fraction(value) for value in record["point_scores_exact"]]
        exact = [[Fraction(value) for value in row] for row in record["paired_differences_exact"]]
        if (len(scores) != candidates + comparators or any(not 0 <= score <= 1 for score in scores) or
                exact != [[score - other for other in scores[candidates:]] for score in scores[:candidates]]):
            raise ValueError("Trial exact paired differences disagree with point scores")
        lower, upper = np.asarray(record["lower"]), np.asarray(record["upper"])
        if (lower.shape != (candidates, comparators) or upper.shape != lower.shape or
                lower.dtype.kind != 'f' or upper.dtype.kind != 'f' or
                not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                np.any(lower > upper) or np.any(lower < -1) or np.any(upper > 1)):
            raise ValueError("Trial interval endpoints are invalid")
        expected_point = [[value < Fraction(1, 50) for value in row] for row in exact]
        expected_interval = (lower <= 0).tolist()
        if (record["individual_point_failures"] != expected_point or
                record["individual_interval_failures"] != expected_interval or
                record["flags"]["point_margin_pass"] != [not any(row) for row in expected_point] or
                record["flags"]["positive_interval_pass"] != [not any(row) for row in expected_interval]):
            raise ValueError("Trial flags disagree with exact margins or interval endpoints")

    def booleans(name: str, shape: tuple[int, ...], *, target: str | None = None) -> np.ndarray:
        values = []
        for record in records:
            flags = record["flags"] if target is None else record["hypothetical"][target]["flags"]
            values.append(flags[name])
        array = np.asarray(values)
        if array.dtype != np.bool_ or array.shape != shape:
            raise ValueError("Trial flags must have explicit boolean shape")
        return array

    shape = (expected, candidates)
    joint = booleans("joint_pass", shape)
    point = booleans("point_margin_pass", shape)
    interval = booleans("positive_interval_pass", shape)
    if not np.array_equal(joint, point & interval):
        raise ValueError("Observed joint event differs from the actual margin-and-interval gate")
    for record in records:
        flags = record["flags"]
        if flags["hypothetical"] is not False or flags["support_violations"] != [
                {"point": False, "bootstrap_draws": 0} for _ in range(candidates)]:
            raise ValueError("Observed simulation cannot contain hypothetical support violations")
    successes = joint.sum(axis=0).tolist()
    hypothetical = {}
    smaller = [Fraction(value) for value in records[0]["hypothetical"]["3/100"]["offsets"]]
    larger = [Fraction(value) for value in records[0]["hypothetical"]["1/25"]["offsets"]]
    if (len(smaller) != candidates or len(larger) != candidates or
            any(high - low != Fraction(1, 100) for low, high in zip(smaller, larger))):
        raise ValueError("Hypothetical .03/.04 offsets must differ by exactly .01")
    for target in ("3/100", "1/25"):
        flags = booleans("joint_pass", shape, target=target)
        offsets = [Fraction(value) for value in records[0]["hypothetical"][target]["offsets"]]
        if len(offsets) != candidates or any(abs(value - Fraction(target)) > 1 for value in offsets):
            raise ValueError("Hypothetical offsets cannot arise from valid F1 centers")
        point_flags = booleans("point_margin_pass", shape, target=target)
        interval_flags = booleans("positive_interval_pass", shape, target=target)
        unsupported = np.zeros(shape, dtype=bool)
        for index, record in enumerate(records):
            hypothetical_row = record["hypothetical"][target]
            if [Fraction(value) for value in hypothetical_row["offsets"]] != offsets:
                raise ValueError("Hypothetical population-center offsets changed between trials")
            item = hypothetical_row["flags"]
            if item["hypothetical"] is not any(value != 0 for value in offsets):
                raise ValueError("Hypothetical flag differs from its offsets")
            bounds = np.asarray(record["bootstrap_candidate_score_bounds"])
            if (bounds.shape != (candidates, 2) or bounds.dtype.kind != 'f' or
                    not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]) or
                    np.any(bounds < 0) or np.any(bounds > 1)):
                raise ValueError("Bootstrap candidate score bounds are invalid")
            scores = [Fraction(value) for value in record["point_scores_exact"][:candidates]]
            exact = [[Fraction(value) for value in row] for row in record["paired_differences_exact"]]
            lower = np.asarray(record["lower"])
            expected_point = [all(value + offset >= Fraction(1, 50) for value in row)
                              for row, offset in zip(exact, offsets)]
            expected_interval = np.all(lower + np.asarray([float(value) for value in offsets])[:, None] > 0, axis=1)
            if (point_flags[index].tolist() != expected_point or
                    not np.array_equal(interval_flags[index], expected_interval)):
                raise ValueError("Hypothetical flags disagree with shifted exact margins or intervals")
            violations = item["support_violations"]
            if type(violations) is not list or len(violations) != candidates:
                raise ValueError("Hypothetical support counts are incomplete")
            for candidate, (score, offset, violation) in enumerate(zip(scores, offsets, violations)):
                point_violation = not 0 <= score + offset <= 1
                shifted_bounds = bounds[candidate] + float(offset)
                bootstrap_violation = bool(shifted_bounds[0] < 0 or shifted_bounds[1] > 1)
                if (type(violation) is not dict or set(violation) != {"point", "bootstrap_draws"} or
                        violation["point"] is not point_violation or
                        type(violation["bootstrap_draws"]) is not int or
                        not 0 <= violation["bootstrap_draws"] <= 10000 or
                        (violation["bootstrap_draws"] > 0) != bootstrap_violation):
                    raise ValueError("Hypothetical support flags disagree with exact scores or bootstrap bounds")
                unsupported[index, candidate] = point_violation or bootstrap_violation
            expected_joint = point_flags[index] & interval_flags[index] & ~unsupported[index]
            if not np.array_equal(flags[index], expected_joint):
                raise ValueError("Hypothetical joint flags disagree with shifted gate and support")
        hypothetical[target] = {
            "joint_success_fraction_all_planned_trials": flags.mean(axis=0).tolist(),
            "unsupported_trial_count": unsupported.sum(axis=0).tolist(),
            "interpretation": "unsupported_offsets_are_not_observed_model_failures",
            "readiness_authority": False,
        }
    covariance = [np.atleast_2d(np.cov(differences[:, candidate, :], rowvar=False, ddof=1)).tolist()
                  for candidate in range(candidates)]
    point_failure = np.asarray([record["individual_point_failures"] for record in records])
    interval_failure = np.asarray([record["individual_interval_failures"] for record in records])
    if (point_failure.dtype != np.bool_ or interval_failure.dtype != np.bool_ or
            point_failure.shape != (expected, candidates, comparators) or
            interval_failure.shape != point_failure.shape):
        raise ValueError("Per-comparator failures are incomplete")
    bounds = [monte_carlo_lower(int(value), expected) for value in successes]
    return {
        "schema_version": "1.0", "artifact_type": "precision_population_summary_not_readiness",
        "population_index": population_index, "trials": expected,
        "candidate_count": candidates, "comparator_count": comparators,
        "joint_success_count": successes,
        "joint_success_fraction": joint.mean(axis=0).tolist(),
        "point_success_fraction": point.mean(axis=0).tolist(),
        "interval_success_fraction": interval.mean(axis=0).tolist(),
        "individual_point_failure_fraction": point_failure.mean(axis=0).tolist(),
        "individual_interval_failure_fraction": interval_failure.mean(axis=0).tolist(),
        "one_sided_95pct_conditional_monte_carlo_lower": bounds,
        "conditional_80pct_planning_flag": [value >= 0.8 for value in bounds],
        "paired_point_difference_covariance_ddof1": covariance,
        "paired_point_difference_std_ddof1": differences.std(axis=0, ddof=1).tolist(),
        "covariance_scope": "across_pseudo_audit_point_differences_not_true_population_CI",
        "hypothetical": hypothetical,
        "readiness_authority": False,
    }
