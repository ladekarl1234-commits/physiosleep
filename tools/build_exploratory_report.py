"""Render identifier-free exploratory Audit A/B aggregates from saved results only.

No truth, predictions, participant rows, EDFs, or models are opened here.
Invoke after both completed result JSON files exist.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

MODEL_SHA = "abdc64d3cd736f0a2e6c89fb5843eee06a3af26f64f0abb80120ade794d83860"
SCOPE = "EXPLORATORY_A_AND_B_NO_GATE_PASS"
CONTROL = "Local LightGBM classifier trained on YASA EEG+EOG features, D60 seed17"
CLASSES = ("W", "N1", "N2", "N3", "REM")
RESULTS = Path("runs/exploratory-audits-v1/results")
OUTPUT = Path("reports/exploratory-audits-v1")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Exploratory JSON repeats a key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Exploratory JSON contains a non-finite constant: " + value)


def _read_json(path: Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"),
                        object_pairs_hook=_unique, parse_constant=_reject_constant)
    _require(type(result) is dict, "Exploratory JSON root must be an object")
    return result


def _json_text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        _require(_read_json(path) == value, "Existing exploratory aggregate differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(_json_text(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics(confusion: np.ndarray) -> dict:
    cm = np.asarray(confusion, dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    f1 = [float(Fraction(2 * int(cm[i, i]), int(support[i] + predicted[i]))
                if support[i] + predicted[i] else Fraction(0)) for i in range(5)]
    precision = [int(cm[i, i]) / int(predicted[i]) if predicted[i] else 0.0 for i in range(5)]
    recall = [int(cm[i, i]) / int(support[i]) if support[i] else 0.0 for i in range(5)]
    exact = sum((Fraction(2 * int(cm[i, i]), int(support[i] + predicted[i]))
                 if support[i] + predicted[i] else Fraction(0) for i in range(5)), Fraction(0)) / 5
    total = int(cm.sum())
    actual = int(np.trace(cm))
    expected_numerator = sum(int(a) * int(b) for a, b in zip(support, predicted))
    denominator = total * total - expected_numerator
    kappa = (float(Fraction(actual * total - expected_numerator, denominator))
             if denominator else None)
    return {"confusion": cm.tolist(), "support": support.tolist(),
            "predicted_support": predicted.tolist(), "per_class_precision": precision,
            "per_class_recall": recall, "per_class_f1": f1,
            "macro_f1": float(exact), "macro_f1_exact": f"{exact.numerator}/{exact.denominator}",
            "accuracy": actual / total if total else None, "kappa": kappa,
            "evaluated_epochs": total}


def _number(value, name: str, *, lower: float = -1.0, upper: float = 1.0) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
             "Exploratory aggregate has invalid " + name)
    return float(value)


def _phase(path: Path, phase: str) -> tuple[dict, dict]:
    raw = _read_json(path)
    _require(type(raw) is dict and raw.get("artifact_type") ==
             "physiosleep_exploratory_audit_result" and raw.get("schema_version") == "1.0" and
             raw.get("status") == "EXPLORATORY_EVALUATED" and raw.get("gate_status") == "NOT_RUN" and
             raw.get("phase") == phase and raw.get("scope") == SCOPE and
             raw.get("model_sha256") == MODEL_SHA and
             type(raw.get("metrics")) is dict and
             type(raw.get("prediction_sha256")) is dict and
             type(raw.get("truth_payload_sha256")) is dict and
             len(raw["prediction_sha256"]) == len(raw["truth_payload_sha256"]) == 39 and
             set(raw["prediction_sha256"]) == set(raw["truth_payload_sha256"]),
             "Exploratory result identity, phase or complete roster differs")
    metric = raw["metrics"]
    confusion = np.asarray(metric.get("confusion"))
    _require(confusion.shape == (5, 5) and confusion.dtype.kind in "iu" and
             np.all(confusion >= 0), "Exploratory pooled confusion is malformed")
    recomputed = _metrics(confusion)
    for key in ("confusion", "support", "predicted_support", "macro_f1_exact"):
        _require(metric.get(key) == recomputed[key],
                 "Exploratory pooled " + key + " differs from fixed five-class recomputation")
    for key in ("macro_f1", "accuracy", "kappa"):
        expected, observed = recomputed[key], metric.get(key)
        _require((expected is None and observed is None) or
                 (expected is not None and math.isclose(
                     _number(observed, key), expected, rel_tol=0, abs_tol=1e-12)),
                 "Exploratory pooled " + key + " differs from recomputation")
    for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
        observed = metric.get(key)
        _require(type(observed) is list and len(observed) == 5 and
                 all(math.isclose(_number(a, key, lower=0), b, rel_tol=0, abs_tol=1e-12)
                     for a, b in zip(observed, recomputed[key])),
                 "Exploratory fixed-order class metric differs")
    _require(type(metric.get("participant_count")) is int and metric["participant_count"] == 20 and
             type(metric.get("recording_count")) is int and metric["recording_count"] == 39 and
             type(metric.get("invalid_epochs")) is int and metric["invalid_epochs"] >= 0 and
             type(metric.get("complete_psg_epochs")) is int and
             metric["complete_psg_epochs"] == recomputed["evaluated_epochs"] + metric["invalid_epochs"] and
             metric.get("evaluated_epochs") == recomputed["evaluated_epochs"] and
             metric.get("prediction_coverage_fraction") == 1.0,
             "Exploratory complete-grid counts or coverage differ")
    ci = metric.get("descriptive_macro_f1_interval")
    _require(type(ci) is dict and ci.get("method") ==
             "single_model_participant_cluster_stratified_SC_ST_percentile" and
             ci.get("scope") == "exploratory_descriptive_not_paired_margin_or_gate" and
             ci.get("rng") == "PCG64" and ci.get("seed") ==
             (2026092301 if phase == "A" else 2026092302) and
             ci.get("draws") == 10000 and ci.get("confidence_level") == 0.95 and
             ci.get("quantile_method") == "linear" and ci.get("participant_count") == 20 and
             ci.get("cohort_counts") == {"SC": 16, "ST": 4},
             "Exploratory descriptive bootstrap identity differs")
    lower, upper = _number(ci.get("lower"), "CI lower", lower=0), _number(
        ci.get("upper"), "CI upper", lower=0)
    _require(lower <= upper, "Exploratory descriptive CI is reversed")
    public = {
        "phase": phase, "recording_count": 39, "participant_count": 20,
        "cohort_participants": {"SC": 16, "ST": 4},
        "class_order": list(CLASSES), "confusion": recomputed["confusion"],
        "reference_support": recomputed["support"],
        "predicted_support": recomputed["predicted_support"],
        "per_class_precision": recomputed["per_class_precision"],
        "per_class_recall": recomputed["per_class_recall"],
        "per_class_f1": recomputed["per_class_f1"],
        "macro_f1": recomputed["macro_f1"],
        "macro_f1_exact": recomputed["macro_f1_exact"],
        "accuracy": recomputed["accuracy"], "kappa": recomputed["kappa"],
        "evaluated_epochs": recomputed["evaluated_epochs"],
        "invalid_epochs": metric["invalid_epochs"],
        "complete_psg_epochs": metric["complete_psg_epochs"],
        "prediction_coverage_fraction": 1.0,
        "descriptive_macro_f1_interval": {
            "lower": lower, "upper": upper, "confidence_level": 0.95,
            "method": ci["method"], "rng": "PCG64", "seed": ci["seed"],
            "draws": 10000, "strata": ["SC", "ST"],
            "resampling_unit": "participant_all_nights",
            "coverage_scope": "individual_phase_not_simultaneous_or_paired"},
    }
    return raw, public


def aggregate(results_dir: Path) -> dict:
    paths = {phase: Path(results_dir) / (phase + ".json") for phase in ("A", "B")}
    _require(all(path.is_file() for path in paths.values()),
             "Both exploratory A/B saved results must exist before report rendering")
    loaded = {phase: _phase(paths[phase], phase) for phase in ("A", "B")}
    originals = {phase: loaded[phase][0] for phase in ("A", "B")}
    _require(len({originals[phase].get("selection_sha256") for phase in originals}) == 1 and
             len({originals[phase].get("protocol_hash") for phase in originals}) == 1 and
             all(type(originals[phase].get("selection_sha256")) is str and
                 type(originals[phase].get("protocol_hash")) is str for phase in originals),
             "Exploratory A/B results do not share one frozen selection and protocol")
    result = {"artifact_type": "physiosleep_exploratory_audits_public_aggregate",
              "schema_version": "1.0", "status": "EXPLORATORY_EVALUATED",
              "gate_status": "NOT_RUN", "scope": SCOPE,
              "control": CONTROL,
              "model_sha256": MODEL_SHA,
              "selection_sha256": originals["A"]["selection_sha256"],
              "protocol_hash": originals["A"]["protocol_hash"],
              "source_result_sha256": {phase: _sha256(paths[phase]) for phase in paths},
              "total_recordings": 78, "total_participants": 40,
              "phases": {phase: loaded[phase][1] for phase in ("A", "B")},
              "interpretation": "Exploratory holdout consumption; separate descriptive phase intervals; no paired comparator or Gate A/B pass. Fresh data are required for confirmation."}
    _json_text(result)
    return result


def _public_aggregate(path: Path) -> dict:
    """Accept only the identifier-free summary format for standalone rerenders."""
    public = _read_json(path)
    expected = {"artifact_type", "schema_version", "status", "gate_status", "scope",
                "control", "model_sha256", "selection_sha256", "protocol_hash",
                "source_result_sha256", "total_recordings", "total_participants",
                "phases", "interpretation"}
    _require(set(public) in (expected, expected | {"figure_sha256"}) and
             public["artifact_type"] == "physiosleep_exploratory_audits_public_aggregate" and
             public["schema_version"] == "1.0" and public["status"] == "EXPLORATORY_EVALUATED" and
             public["gate_status"] == "NOT_RUN" and public["scope"] == SCOPE and
             public["control"] == CONTROL and
             public["model_sha256"] == MODEL_SHA and
             public["total_recordings"] == 78 and public["total_participants"] == 40 and
             set(public["phases"]) == {"A", "B"} and
             set(public["source_result_sha256"]) == {"A", "B"} and
             all(type(value) is str and len(value) == 64 and
                 all(char in "0123456789abcdef" for char in value)
                 for value in public["source_result_sha256"].values()),
             "Public exploratory aggregate scope or source identity differs")
    phase_keys = {"phase", "recording_count", "participant_count", "cohort_participants",
                  "class_order", "confusion", "reference_support", "predicted_support",
                  "per_class_precision", "per_class_recall", "per_class_f1", "macro_f1",
                  "macro_f1_exact", "accuracy", "kappa", "evaluated_epochs",
                  "invalid_epochs", "complete_psg_epochs", "prediction_coverage_fraction",
                  "descriptive_macro_f1_interval"}
    for phase, row in public["phases"].items():
        _require(type(row) is dict and set(row) == phase_keys and row["phase"] == phase and
                 row["class_order"] == list(CLASSES) and row["recording_count"] == 39 and
                 row["participant_count"] == 20 and
                 row["cohort_participants"] == {"SC": 16, "ST": 4},
                 "Public exploratory phase includes unknown or identifying fields")
        matrix = np.asarray(row["confusion"])
        _require(matrix.shape == (5, 5) and matrix.dtype.kind in "iu" and
                 np.all(matrix >= 0), "Public exploratory confusion differs")
        computed = _metrics(matrix)
        for key, source in (("reference_support", "support"),
                            ("predicted_support", "predicted_support"),
                            ("macro_f1_exact", "macro_f1_exact")):
            _require(row[key] == computed[source], "Public exploratory metric differs: " + key)
        for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
            _require(type(row[key]) is list and len(row[key]) == 5 and
                     all(math.isclose(_number(a, key, lower=0), b, rel_tol=0, abs_tol=1e-12)
                         for a, b in zip(row[key], computed[key])),
                     "Public exploratory class metric differs")
        for key in ("macro_f1", "accuracy", "kappa"):
            expected_value = computed[key]
            _require((expected_value is None and row[key] is None) or
                     (expected_value is not None and math.isclose(
                         _number(row[key], key), expected_value, rel_tol=0, abs_tol=1e-12)),
                     "Public exploratory pooled metric differs")
        _require(row["evaluated_epochs"] == computed["evaluated_epochs"] and
                 type(row["invalid_epochs"]) is int and row["invalid_epochs"] >= 0 and
                 row["complete_psg_epochs"] == row["evaluated_epochs"] + row["invalid_epochs"] and
                 row["prediction_coverage_fraction"] == 1.0,
                 "Public exploratory original-grid count differs")
        ci = row["descriptive_macro_f1_interval"]
        _require(type(ci) is dict and set(ci) == {
                     "lower", "upper", "confidence_level", "method", "rng", "seed",
                     "draws", "strata", "resampling_unit", "coverage_scope"} and
                 ci["confidence_level"] == 0.95 and ci["rng"] == "PCG64" and
                 ci["seed"] == (2026092301 if phase == "A" else 2026092302) and
                 ci["draws"] == 10000 and ci["strata"] == ["SC", "ST"] and
                 ci["resampling_unit"] == "participant_all_nights" and
                 ci["coverage_scope"] == "individual_phase_not_simultaneous_or_paired" and
                 ci["method"] == "single_model_participant_cluster_stratified_SC_ST_percentile" and
                 0 <= _number(ci["lower"], "CI lower", lower=0) <=
                 _number(ci["upper"], "CI upper", lower=0) <= 1,
                 "Public exploratory uncertainty statement differs")
    public.pop("figure_sha256", None)
    return public


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 240,
                         "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "physiosleep-exploratory-audits-v1"})
    return plt


def _save(fig, base: Path) -> None:
    for suffix in ("png", "svg"):
        target = base.with_suffix("." + suffix)
        with tempfile.NamedTemporaryFile(dir=base.parent, suffix="." + suffix,
                                         delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, bbox_inches="tight", pad_inches=.12,
                        metadata={"Software": "PhysioSleep exploratory report"} if suffix == "png"
                                 else {"Date": None, "Creator": "PhysioSleep exploratory report"})
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def _plot_confusions(plt, phases: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, phase in zip(axes, ("A", "B")):
        row = phases[phase]
        matrix = np.asarray(row["confusion"], dtype=np.float64)
        support = np.asarray(row["reference_support"], dtype=np.float64)
        normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix),
                               where=support[:, None] > 0)
        colors = plt.get_cmap("Blues").copy()
        colors.set_bad("#e8ecf0")
        masked = np.ma.masked_where(np.broadcast_to(support[:, None] == 0, (5, 5)),
                                    normalized)
        image = ax.imshow(masked, vmin=0, vmax=1, cmap=colors, aspect="equal")
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Predicted stage")
        ax.set_ylabel("Reference stage")
        ax.set_title(f"Audit {phase} · {row['evaluated_epochs']:,} valid epochs")
        for i in range(5):
            for j in range(5):
                label = "—" if support[i] == 0 else f"{normalized[i, j]:.0%}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if normalized[i, j] > .58 else "#203040", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=.78, label="Fraction of each reference stage")
    _save(fig, output / "confusion")
    plt.close(fig)


def _plot_class_f1(plt, phases: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    positions = np.arange(5)
    for phase, offset, color in (("A", -.18, "#176b9a"), ("B", .18, "#d46a3d")):
        values = phases[phase]["per_class_f1"]
        ax.barh(positions + offset, values, height=.33, label=f"Audit {phase}", color=color)
    ax.set_yticks(positions, CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on reference-valid epochs")
    ax.set_title("Five-class F1 by exploratory audit phase")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, output / "class-f1")
    plt.close(fig)


def _plot_summary(plt, phases: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1), constrained_layout=True)
    colors = ["#176b9a", "#d46a3d"]
    names = ["Audit A", "Audit B"]
    for ax, key, label in zip(axes, ("macro_f1", "accuracy", "kappa"),
                              ("Pooled Macro-F1", "Accuracy", "Cohen's κ")):
        values = [phases[phase][key] for phase in ("A", "B")]
        for x, value in enumerate(values):
            if value is None:
                ax.text(x, .5, "unavailable", ha="center", va="center", rotation=90,
                        fontsize=8, color="#576677")
            else:
                ax.bar(x, value, color=colors[x], width=.58)
        if key == "macro_f1":
            _require(all(value is not None for value in values),
                     "Exploratory pooled Macro-F1 cannot be unavailable")
            cis = [phases[phase]["descriptive_macro_f1_interval"] for phase in ("A", "B")]
            for x, ci in enumerate(cis):
                ax.vlines(x, ci["lower"], ci["upper"], color="#172b3a", linewidth=1.4)
                ax.hlines([ci["lower"], ci["upper"]], x - .08, x + .08,
                          color="#172b3a", linewidth=1.4)
        available = [value for value in values if value is not None]
        ax.set_ylim(max(-1.0, min(available) - .1) if key == "kappa" and available and min(available) < 0
                    else 0, 1.02)
        ax.set_xticks((0, 1), names)
        ax.set_title(label)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for x, value in enumerate(values):
            if value is not None:
                ax.text(x, value + .022, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Exploratory A/B · same D60 single-model control", fontsize=13)
    fig.text(.5, -.01, "Macro-F1 bars have separate participant-cluster 95% descriptive intervals; no paired comparison.",
             ha="center", fontsize=8, color="#425466")
    _save(fig, output / "summary")
    plt.close(fig)


def _report_text(public: dict) -> str:
    phases = public["phases"]
    a, b = phases["A"], phases["B"]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"
    lines = ["# Exploratory Audit A/B — D60 YASA control", "",
             "Both reserved holdouts were consumed for an exploratory check. This is a **locally trained LightGBM classifier using YASA EEG+EOG features** (D60 seed 17), not released YASA weights or a new neural model. The separate development three-seed/transition ensemble reported pooled Macro-F1 0.7897786854; it was not evaluated here. These results cannot establish Gate A/B passage or clinical validity; fresh data are required for confirmation.",
             "", "| Measure | Audit A | Audit B |", "|---|---:|---:|",
             f"| Recordings / participants | {a['recording_count']} / {a['participant_count']} | {b['recording_count']} / {b['participant_count']} |",
             f"| Reference-valid / complete epochs | {a['evaluated_epochs']:,} / {a['complete_psg_epochs']:,} | {b['evaluated_epochs']:,} / {b['complete_psg_epochs']:,} |",
             f"| Invalid reference epochs | {a['invalid_epochs']:,} | {b['invalid_epochs']:,} |",
             f"| Pooled five-class Macro-F1 | {fmt(a['macro_f1'])} | {fmt(b['macro_f1'])} |",
             f"| Descriptive 95% interval | {fmt(a['descriptive_macro_f1_interval']['lower'])}–{fmt(a['descriptive_macro_f1_interval']['upper'])} | {fmt(b['descriptive_macro_f1_interval']['lower'])}–{fmt(b['descriptive_macro_f1_interval']['upper'])} |",
             f"| Accuracy | {fmt(a['accuracy'])} | {fmt(b['accuracy'])} |",
             f"| Cohen's κ | {fmt(a['kappa'])} | {fmt(b['kappa'])} |",
             "", "Intervals resample participants within SC and ST (16 and only 4 ST participants per phase), retaining all nights for each sampled person; 10,000 PCG64 draws use the preregistered phase seeds. They are **separate, descriptive 95% intervals** conditional on this frozen checkpoint. They omit training and model-selection variability, with no paired comparator or simultaneous-coverage claim. A and B come from the same Sleep-EDF source and cohort families, not independent external populations.",
             "", "## Class-level performance", "",
             "| Stage | A F1 | A recall | B F1 | B recall |", "|---|---:|---:|---:|---:|"]
    for i, stage in enumerate(CLASSES):
        lines.append(f"| {stage} | {fmt(a['per_class_f1'][i])} | {fmt(a['per_class_recall'][i])} | {fmt(b['per_class_f1'][i])} | {fmt(b['per_class_recall'][i])} |")
    lines += ["", "## Figures", "",
              "- [Row-normalized confusion matrices](confusion.png) ([SVG](confusion.svg)); each row is conditioned on the true stage.",
              "- [Per-class F1](class-f1.png) ([SVG](class-f1.svg)).",
              "- [Pooled summary](summary.png) ([SVG](summary.svg)); only Macro-F1 has descriptive uncertainty bars.",
              "", "## Provenance and limits", "",
              f"- Control checkpoint SHA-256: `{public['model_sha256']}`.",
              f"- Frozen selection SHA-256: `{public['selection_sha256']}`; protocol hash: `{public['protocol_hash']}`.",
              f"- Saved result SHA-256: A `{public['source_result_sha256']['A']}`; B `{public['source_result_sha256']['B']}`.",
              "- Fixed class order: W, N1, N2, N3, REM. Scores pool reference-valid original 30-second epochs; invalid epochs remain in the grid and are counted separately. Full-grid prediction coverage was required.",
              "- The public aggregate and figures contain no per-record predictions, truth arrays, participant rows or recording identifiers.",
              ""]
    return "\n".join(lines)


def build(*, results_dir: Path | None = None, aggregate_path: Path | None = None,
          output_dir: Path) -> dict:
    _require((results_dir is None) != (aggregate_path is None),
             "Choose private A/B results or a sanitized public aggregate")
    public = (aggregate(results_dir) if results_dir is not None
              else _public_aggregate(aggregate_path))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt = _style()
    _plot_confusions(plt, public["phases"], output)
    _plot_class_f1(plt, public["phases"], output)
    _plot_summary(plt, public["phases"], output)
    public["figure_sha256"] = {name: _sha256(output / name) for name in (
        "confusion.png", "confusion.svg", "class-f1.png", "class-f1.svg",
        "summary.png", "summary.svg")}
    _atomic_json(output / "aggregate.json", public)
    report = output / "report.md"
    text = _report_text(public)
    if report.exists():
        _require(report.read_text(encoding="utf-8") == text,
                 "Existing exploratory public report differs from saved aggregate")
    else:
        report.write_text(text, encoding="utf-8")
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--results-dir", type=Path,
                        help="Private A/B saved result JSON directory (both files required)")
    inputs.add_argument("--aggregate", type=Path,
                        help="Identifier-free aggregate.json for standalone public rerender")
    parser.add_argument("--output-dir", type=Path, default=root / OUTPUT)
    args = parser.parse_args()
    result = build(results_dir=args.results_dir or (None if args.aggregate else root / RESULTS),
                   aggregate_path=args.aggregate, output_dir=args.output_dir)
    print(result["status"], {phase: result["phases"][phase]["macro_f1"]
                             for phase in ("A", "B")})


if __name__ == "__main__":
    main()
