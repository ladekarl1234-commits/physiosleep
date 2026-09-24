"""Render source-traceable development figures from saved results; no EDF access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/publication-figures"
CLASS = ("W", "N1", "N2", "N3", "REM")
BLUE, GOLD, PINK, DARK, GREY = "#23658c", "#bc8429", "#a55372", "#24313a", "#70818b"
COHORT = {"SC": BLUE, "ST": GOLD}
RUNS = {
    "EOG": "provisional-eog/2174694681ab0041",
    "EEG": "yasa-eeg-s17-lr1-016e83b2f3",
    "EEG + EOG · seed 17": "yasa-eeg_eog-s17-lr1-493ef41680",
    "EEG + EOG · seed 43": "yasa-eeg_eog-s43-lr1-44993977b7",
    "EEG + EOG · seed 101": "yasa-eeg_eog-s101-lr1-d8a9381b60",
    "Two-view ensemble": "development-equal-probability-b5b1e708bd1e",
    "Three-seed ensemble": "development-equal-probability-4a6932b4182c",
    "Transition decoder": "temporal-transition-d60-153801fcc3",
}
BEST = "Transition decoder"
DIRECT = "EEG + EOG · seed 17"
SCORE_SYSTEMS = {"EOG": "eog", "EEG": "fpz_eeg", "EEG + EOG": "fpz_eeg_eog"}
ENDPOINTS = {"score": ("experimental_score", "Score · points"),
             "tst_minutes": ("tst_minutes", "TST · min"),
             "waso_minutes": ("waso_minutes", "WASO · min")}


def source(path: Path, sources: dict[str, str]) -> Path:
    path = path.resolve()
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise ValueError(f"Source outside project or missing: {path}")
    sources[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path: Path, sources: dict[str, str]) -> dict:
    return json.loads(source(path, sources).read_text(encoding="utf-8"))


def read_npz(path: Path, sources: dict[str, str]) -> dict:
    with np.load(source(path, sources), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def index_paths(paths: list[str]) -> dict[str, Path]:
    indexed = {Path(path).stem: Path(path) for path in paths}
    if len(indexed) != len(paths):
        raise ValueError("Duplicate recording paths")
    return indexed


def histogram(values, edges) -> dict:
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=np.asarray(edges, dtype=float))
    if not len(values) or not np.isfinite(values).all() or int(counts.sum()) != len(values):
        raise ValueError("Non-finite or unbinned distribution")
    return {"bin_edges": edges.tolist(), "bin_counts": counts.tolist(),
            "n": len(values), "quartiles": np.quantile(values, [.25, .5, .75]).tolist(),
            "minimum": float(values.min()), "maximum": float(values.max())}


def binned_pairs(x, y, x_edges, y_edges) -> list[dict]:
    counts, xe, ye = np.histogram2d(x, y, bins=(x_edges, y_edges))
    if int(counts.sum()) != len(x):
        raise ValueError("Pairs fell outside display bins")
    return [{"x": float((xe[i] + xe[i+1]) / 2), "y": float((ye[j] + ye[j+1]) / 2),
             "count": int(counts[i, j])}
            for i, j in zip(*np.nonzero(counts))]


def public_sources(private: dict[str, str]) -> dict[str, str]:
    groups = {
        "best_prediction_pairs_119": lambda p: p.startswith(
            f"runs/{RUNS[BEST]}/predictions/") and p.endswith((".npz", ".json")),
        "direct_prediction_pairs_119": lambda p: p.startswith(
            f"runs/{RUNS[DIRECT]}/") and "/predictions/" in p and p.endswith((".npz", ".json")),
        "reference_truth_payloads_119": lambda p: p.startswith(
            "derived/evaluator_truth/") and p.endswith(".npz"),
    }
    remaining = dict(private)
    published = {}
    for name, matches in groups.items():
        members = sorted((key, value) for key, value in private.items() if matches(key))
        expected = 119 if name == "reference_truth_payloads_119" else 238
        if len(members) != expected:
            raise ValueError(f"Expected {expected} source files for {name}")
        published[name] = hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode("utf-8")).hexdigest()
        for key, _ in members:
            del remaining[key]
    published.update(remaining)
    return dict(sorted(published.items()))


def f1(conf: np.ndarray) -> np.ndarray:
    tp = np.diag(conf).astype(float)
    den = conf.sum(axis=0) + conf.sum(axis=1)
    return np.divide(2 * tp, den, out=np.zeros(5), where=den != 0)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelcolor": DARK, "text.color": DARK,
        "xtick.color": DARK, "ytick.color": DARK, "axes.edgecolor": GREY,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "sleep-research-figures-v1",
    })


def title(fig, heading: str, subtitle: str) -> None:
    fig.suptitle(heading, x=0.055, y=0.985, ha="left", fontsize=19, weight="bold", color=DARK)
    fig.text(0.055, .946 if fig.get_figheight() > 8 else .895,
             subtitle, color=GREY, fontsize=10, ha="left")


def finish(fig, slug: str, heading: str, subtitle: str, note: str, data: dict,
           source_paths: list[str], outputs: dict) -> None:
    fig.text(0.055, 0.022, note, color=GREY, fontsize=8.8, ha="left")
    fig.savefig(OUT / f"{slug}.png", dpi=210, bbox_inches="tight", pad_inches=0.17,
                metadata={"Software": "Matplotlib"})
    fig.savefig(OUT / f"{slug}.svg", bbox_inches="tight", pad_inches=0.17,
                metadata={"Date": None})
    plt.close(fig)
    payload = {"title": heading, "subtitle": subtitle, "note": note,
               "scope": "development out-of-fold; descriptive; not confirmatory or clinical",
               "sources": sorted(set(source_paths)), "data": data}
    (OUT / f"{slug}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs[slug] = {"png": f"{slug}.png", "svg": f"{slug}.svg", "data": f"{slug}.json",
                     "source_count": len(payload["sources"])}


def load_runs(sources: dict[str, str]) -> dict[str, dict]:
    result = {name: read_json(ROOT / "runs" / run / "result.json", sources)
              for name, run in RUNS.items()}
    canonical = result[BEST]
    best_config = read_json(ROOT / "runs" / RUNS[BEST] / "config.json", sources)
    direct_config = read_json(ROOT / "runs" / RUNS[DIRECT] / "config.json", sources)
    if (best_config["split_id"] != direct_config["split_id"] or
            best_config["protocol_hash"] != canonical["protocol_hash"]):
        raise ValueError("Best and direct runs do not share the development split")
    truth = set(index_paths(canonical["truth_paths"]))
    protocol = canonical["protocol_hash"]
    for name, report in result.items():
        metrics = report["metrics"]
        if (report["confirmatory"] or report["gate_A"] != "NOT_RUN" or
                report["gate_B"] != "NOT_RUN" or report["protocol_hash"] != protocol or
                set(index_paths(report["truth_paths"])) != truth or
                set(index_paths(report["prediction_paths"])) != truth or
                metrics["evaluated_epochs"] != 274271 or metrics["recording_count"] != 119 or
                metrics["participant_count"] != 60):
            raise ValueError(f"Incomparable development result: {name}")
    expected_members = {
        "Two-view ensemble": {"yasa-eeg-s17-lr1-016e83b2f3",
                              RUNS[DIRECT]},
        "Three-seed ensemble": {RUNS[DIRECT], RUNS["EEG + EOG · seed 43"],
                                RUNS["EEG + EOG · seed 101"]},
    }
    for name, expected in expected_members.items():
        member_path = ROOT / "runs" / RUNS[name] / "members.json"
        members = read_json(member_path, sources)
        declared = members.get("members", [])
        identity = members.get("identity", {})
        run_ids = [item.get("run_id") for item in declared]
        identity_ids = [item.get("run_id") for item in identity.get("member_results", [])]
        if (len(run_ids) != len(expected) or len(set(run_ids)) != len(expected) or
                set(run_ids) != expected or set(identity_ids) != expected or
                identity.get("rule") != "unweighted_mean_float64_probabilities" or
                sources[str(member_path.relative_to(ROOT)).replace("\\", "/")] !=
                result[name].get("members_sha256")):
            raise ValueError(f"Ensemble member identity mismatch: {name}")
    return result


def staging_sources(sources: dict[str, str], include_direct: bool = False) -> list[str]:
    selected = ["best_prediction_pairs_119", "reference_truth_payloads_119"]
    selected += [f"runs/{RUNS[BEST]}/{name}" for name in ("result.json", "config.json")]
    if include_direct:
        selected += ["direct_prediction_pairs_119",
                     f"runs/{RUNS[DIRECT]}/result.json", f"runs/{RUNS[DIRECT]}/config.json"]
    if not set(selected) <= set(sources):
        raise ValueError("Missing published figure source")
    return sorted(set(selected))


def nights_and_confusion(runs: dict[str, dict], sources: dict[str, str]):
    best, direct = runs[BEST], runs[DIRECT]
    pred = index_paths(best["prediction_paths"])
    direct_pred = index_paths(direct["prediction_paths"])
    truth = index_paths(best["truth_paths"])
    conf = np.zeros((5, 5), dtype=np.int64)
    cohort_conf = {c: np.zeros((5, 5), dtype=np.int64) for c in COHORT}
    participant_conf = defaultdict(lambda: np.zeros((5, 5), dtype=np.int64))
    participant_fit_sets = {}
    nights = []
    total_complete = total_invalid = 0
    for rid in sorted(truth):
        t, p, d = (read_npz(paths[rid], sources)
                   for paths in (truth, pred, direct_pred))
        participant = str(t["participant_id"].item())
        cohort = participant.split(":")[0]
        if (cohort not in COHORT or str(t["recording_id"].item()) != rid or
                str(p["recording_id"].item()) != rid or str(d["recording_id"].item()) != rid or
                str(p["participant_id"].item()) != participant or
                str(d["participant_id"].item()) != participant):
            raise ValueError(f"Recording/participant identity mismatch: {rid}")
        for route_name, path in ((BEST, pred[rid]), (DIRECT, direct_pred[rid])):
            sidecar = read_json(path.with_suffix(".json"), sources)
            provenance = sidecar.get("provenance", {})
            fitted = provenance.get("fitted_participants")
            if (sidecar.get("class_order") != list(CLASS) or
                    provenance.get("hypnogram_required") is not False or
                    not isinstance(fitted, list) or len(fitted) != 48 or
                    len(set(fitted)) != 48 or participant in fitted):
                raise ValueError(f"Out-of-fold sidecar invalid: {route_name}/{rid}")
            key = (route_name, participant)
            fit_set = tuple(sorted(fitted))
            if key in participant_fit_sets and participant_fit_sets[key] != fit_set:
                raise ValueError(f"Participant nights have different fit ancestry: {participant}")
            participant_fit_sets[key] = fit_set
        for key in ("epoch_index", "onset_seconds"):
            if not np.array_equal(t[key], p[key]) or not np.array_equal(t[key], d[key]):
                raise ValueError(f"Prediction grid mismatch: {rid}")
        mask = t["valid_mask"]
        ref, predicted, direct_label = t["reference_label"][mask], p["hard_label"][mask], d["hard_label"][mask]
        proba = d["probabilities"]
        if (mask.dtype != bool or proba.shape != (len(mask), 5) or
                not np.isfinite(proba).all() or
                not np.allclose(proba.sum(axis=1), 1, rtol=0, atol=1e-6) or
                any(np.any((arr < 0) | (arr > 4)) for arr in (ref, predicted, direct_label))):
            raise ValueError(f"Invalid fixed-five-class grid: {rid}")
        if participant in best["metrics"]["per_participant_confusion"] and participant not in direct["metrics"]["per_participant_confusion"]:
            raise ValueError(f"Direct model participant absent: {participant}")
        local = np.zeros((5, 5), dtype=np.int64)
        np.add.at(local, (ref, predicted), 1)
        conf += local
        cohort_conf[cohort] += local
        participant_conf[participant] += local
        total_complete += len(mask)
        total_invalid += len(mask) - int(mask.sum())
        counts_ref = np.bincount(ref, minlength=5)
        counts_pred = np.bincount(predicted, minlength=5)
        nights.append({
            "recording_id": rid, "participant_id": participant, "cohort": cohort,
            "evaluated_epochs": int(mask.sum()), "invalid_epochs": int(len(mask) - mask.sum()),
            "accuracy": float(np.mean(ref == predicted)),
            "direct_accuracy": float(np.mean(ref == direct_label)),
            "direct_mean_max_probability": float(proba[mask].max(axis=1).mean()),
            "reference_fraction": (counts_ref / mask.sum()).tolist(),
            "predicted_fraction": (counts_pred / mask.sum()).tolist(),
        })
    metric = best["metrics"]
    if (len(nights) != 119 or len(participant_conf) != 60 or
            not np.array_equal(conf, metric["confusion"]) or
            int(conf.sum()) != metric["evaluated_epochs"] or
            total_invalid != metric["invalid_epochs"] or total_complete != metric["complete_psg_epochs"] or
            not np.isclose(np.trace(conf) / conf.sum(), metric["accuracy"], atol=1e-12) or
            not np.allclose(f1(conf), metric["per_class_f1"], atol=1e-12) or
            not np.isclose(f1(conf).mean(), metric["macro_f1"], atol=1e-12)):
        raise ValueError("Recomputed pooled metrics disagree with saved best result")
    for participant, local in participant_conf.items():
        if not np.array_equal(local, metric["per_participant_confusion"][participant]):
            raise ValueError(f"Participant confusion differs: {participant}")
    return nights, conf, cohort_conf, participant_conf


def score_rows(sources: dict[str, str]):
    csv_path = source(ROOT / "reports/provisional-phase2-three-sensor/recording-summaries.csv", sources)
    score = read_json(ROOT / "runs/provisional-score-study-f1c65dcf6215/result.json", sources)
    source(ROOT / "runs/provisional-score-study-f1c65dcf6215/config.json", sources)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(row["recording_id"], row["system"]): row for row in rows}
    if (len(rows) != 476 or len(indexed) != 476 or score["reference_score_available"] != 61 or
            score["clinically_validated"] or score["confirmatory"] or
            score["gate_A"] != "NOT_RUN" or score["gate_B"] != "NOT_RUN"):
        raise ValueError("SQI source cohort or status changed")
    eligible = sorted(rid for rid, system in indexed if system == "reference"
                      and indexed[rid, system]["experimental_score"])
    if len(eligible) != 61 or len({indexed[rid, "reference"]["participant_id"] for rid in eligible}) != 40:
        raise ValueError("SQI eligibility count changed")
    details, summary = {}, {}
    for label, system in SCORE_SYSTEMS.items():
        details[label] = {}
        summary[label] = {}
        for metric, (column, _) in ENDPOINTS.items():
            pairs = []
            groups = defaultdict(list)
            for rid in eligible:
                ref, row = indexed[rid, "reference"], indexed[rid, system]
                if (row["participant_id"] != ref["participant_id"] or
                        not row[column] or not ref[column]):
                    raise ValueError(f"SQI missing eligible pair: {rid}, {system}, {column}")
                a, b = float(ref[column]), float(row[column])
                if not np.isfinite([a, b]).all():
                    raise ValueError("Non-finite SQI endpoint")
                pairs.append({"recording_id": rid, "participant_id": ref["participant_id"],
                              "reference": a, "predicted": b})
                groups[ref["participant_id"]].append(b - a)
            errors = list(groups.values())
            mae = float(np.mean([np.mean(np.abs(values)) for values in errors]))
            bias = float(np.mean([np.mean(values) for values in errors]))
            saved = score["models"][system]["endpoints"][metric]
            if (saved["nights"] != 61 or saved["participants"] != 40 or
                    not np.isclose(mae, saved["mae"], atol=1e-10) or
                    not np.isclose(bias, saved["bias"], atol=1e-10)):
                raise ValueError(f"Participant-balanced SQI summary mismatch: {system}/{metric}")
            details[label][metric] = pairs
            summary[label][metric] = {"mae": mae, "bias": bias,
                "mae_ci95": saved["cluster_ci95"]["mae"],
                "nights": 61, "participants": 40}
    return details, summary


def aggregate_figures(nights, conf, cohort_conf, participant_conf, runs, details, score_summary):
    accuracy = {cohort: histogram(
        [row["accuracy"] for row in nights if row["cohort"] == cohort],
        np.linspace(.6, 1.01, 22)) for cohort in COHORT}
    confidence = {}
    for cohort in COHORT:
        rows = [row for row in nights if row["cohort"] == cohort]
        confidence[cohort] = binned_pairs(
            [row["direct_mean_max_probability"] for row in rows],
            [row["direct_accuracy"] for row in rows],
            np.linspace(.75, 1.001, 19), np.linspace(.6, 1.011, 19))
    fraction = {stage: histogram(
        [row["predicted_fraction"][i] - row["reference_fraction"][i] for row in nights],
        np.linspace(-.3, .3, 31)) for i, stage in enumerate(CLASS)}
    base = runs[DIRECT]["metrics"]["per_participant_confusion"]
    spread = {}
    for cohort in COHORT:
        people = [person for person in sorted(participant_conf)
                  if person.startswith(cohort + ":")]
        current = [float(f1(participant_conf[p]).mean()) for p in people]
        delta = [float(f1(participant_conf[p]).mean() -
                       f1(np.asarray(base[p])).mean()) for p in people]
        spread[cohort] = {"transition_macro_f1": histogram(
            current, np.linspace(.4, 1.01, 31)),
            "paired_delta": histogram(delta, np.linspace(-.1, .1, 41))}
    agreement = {}
    for metric in ENDPOINTS:
        pairs = details["EEG + EOG"][metric]
        x = np.asarray([row["reference"] for row in pairs])
        y = np.asarray([row["predicted"] for row in pairs])
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = max((hi-lo)*.06, .5)
        lo, hi = lo-pad, hi+pad
        edges = np.linspace(lo, hi, 25)
        agreement[metric] = {"bounds": [lo, hi],
                             "binned_pairs": binned_pairs(x, y, edges, edges)}
    return {
        "overview": {"accuracy_distributions": accuracy, "confidence_bins": confidence,
                     "fraction_error_distributions": fraction, "confusion": conf.tolist(),
                     "nights": len(nights), "participants": len(participant_conf),
                     "evaluated_epochs": int(conf.sum())},
        "models": [{"name": name, "run_id": RUNS[name],
                    "macro_f1": runs[name]["metrics"]["macro_f1"],
                    "accuracy": runs[name]["metrics"]["accuracy"]} for name in RUNS],
        "participant_distributions": spread,
        "cohort_confusion": {key: value.tolist() for key, value in cohort_conf.items()},
        "score_agreement": agreement, "score_summary": score_summary,
    }


def distribution_marks(ax, distribution, x, color, *, marker_scale=3):
    edges, counts = distribution["bin_edges"], distribution["bin_counts"]
    for i, count in enumerate(counts):
        if count:
            center = (edges[i] + edges[i+1]) / 2
            ax.scatter(x, center, s=14 + marker_scale*count, alpha=.55, color=color, zorder=3)
    q = distribution["quartiles"]
    ax.plot([x, x], [q[0], q[2]], color=DARK, lw=4, zorder=4)
    ax.scatter(x, q[1], color="white", edgecolor=DARK, s=49, zorder=5)


def make_overview(overview, sources, outputs):
    conf = np.asarray(overview["confusion"])
    fig = plt.figure(figsize=(16.5, 11.6))
    gs = GridSpec(2, 2, figure=fig, left=.075, right=.96, top=.89, bottom=.095,
                  hspace=.34, wspace=.30)
    title(fig, "119-night development result",
          "Transition decoder · 60 participants · five fixed classes · 274,271 valid epochs")
    ax = fig.add_subplot(gs[0, 0])
    for x, cohort in enumerate(COHORT):
        distribution = overview["accuracy_distributions"][cohort]
        ax.bxp([{"med": distribution["quartiles"][1], "q1": distribution["quartiles"][0],
                 "q3": distribution["quartiles"][2], "whislo": distribution["minimum"],
                 "whishi": distribution["maximum"], "fliers": []}],
               positions=[x], widths=.32, patch_artist=True,
               boxprops={"facecolor": COHORT[cohort], "alpha": .17, "edgecolor": COHORT[cohort]},
               medianprops={"color": DARK, "linewidth": 2},
               whiskerprops={"color": COHORT[cohort]}, capprops={"color": COHORT[cohort]})
        edges, counts = distribution["bin_edges"], distribution["bin_counts"]
        for i, count in enumerate(counts):
            if count:
                ax.scatter(x + .20, (edges[i]+edges[i+1])/2, s=13+3*count,
                           color=COHORT[cohort], alpha=.58)
        ax.text(x, .615, f"{distribution['n']} nights", ha="center", color=GREY, fontsize=9)
    ax.set_xticks([0, 1], ["SC", "ST"])
    ax.set_ylim(.6, 1.01)
    ax.set_ylabel("Accuracy on valid epochs")
    ax.set_title("A  Night accuracy by cohort", loc="left")
    ax.grid(axis="y", alpha=.16)

    ax = fig.add_subplot(gs[0, 1])
    for cohort in COHORT:
        bins = overview["confidence_bins"][cohort]
        ax.scatter([point["x"] for point in bins], [point["y"] for point in bins],
                   s=[15 + 8*point["count"] for point in bins], alpha=.68,
                   color=COHORT[cohort],
                   label=f"{cohort} · {overview['accuracy_distributions'][cohort]['n']} nights")
    ax.set_xlim(.75, 1); ax.set_ylim(.6, 1.01)
    ax.set_xlabel("Mean maximum model probability")
    ax.set_ylabel("Night accuracy on valid epochs")
    ax.set_title("B  Direct EEG + EOG model confidence", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=.16)

    ax = fig.add_subplot(gs[1, 0])
    normalized = conf / conf.sum(axis=1, keepdims=True)
    heat = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), CLASS); ax.set_yticks(range(5), CLASS)
    ax.set_xlabel("Predicted stage"); ax.set_ylabel("Reference stage")
    ax.set_title("C  Row-normalized confusion · transition decoder", loc="left")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{normalized[i,j]:.0%}", ha="center", va="center",
                    color="white" if normalized[i,j] > .55 else DARK, fontsize=9)
    fig.colorbar(heat, ax=ax, shrink=.76, pad=.02, label="Fraction of reference stage")

    ax = fig.add_subplot(gs[1, 1])
    for stage, label in enumerate(CLASS):
        distribution = overview["fraction_error_distributions"][label]
        edges, counts = distribution["bin_edges"], distribution["bin_counts"]
        for i, count in enumerate(counts):
            if count:
                ax.scatter(stage, (edges[i]+edges[i+1])/2, s=12+2.5*count,
                           alpha=.37, color=BLUE)
        q = distribution["quartiles"]
        ax.plot([stage, stage], [q[0], q[2]], color=DARK, lw=4, solid_capstyle="round")
        ax.scatter(stage, q[1], color=GOLD, s=42, edgecolor="white", lw=.6, zorder=4)
    ax.axhline(0, color=DARK, lw=1)
    ax.set_xticks(range(5), CLASS)
    ax.set_ylabel("Predicted − reference night fraction")
    ax.set_title("D  Paired stage-fraction error · 119 nights", loc="left")
    ax.grid(axis="y", alpha=.16)
    note = ("Valid mask applied once; bubbles encode anonymous night-bin counts. Panel B uses direct seed-17 "
            "probabilities and its own hard predictions; no decoder posterior or calibration claim.")
    finish(fig, "01_night_overview", "119-night development result",
           "Transition decoder · 60 participants · five fixed classes · 274,271 valid epochs",
           note, overview,
           staging_sources(sources, include_direct=True), outputs)


def make_models(models, sources, outputs):
    names = [model["name"] for model in models]
    scores = [model["macro_f1"] for model in models]
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    fig.subplots_adjust(left=.26, right=.90, top=.80, bottom=.18)
    title(fig, "Development model comparison",
          "Pooled fixed-five-class Macro-F1 · same 119 nights, 60 participants, 274,271 valid epochs")
    y = np.arange(len(names))[::-1]
    colors = [GREY, BLUE, BLUE, BLUE, BLUE, GREY, GOLD, PINK]
    for pos, name, score, color in zip(y, names, scores, colors):
        ax.scatter(score, pos, s=115 if name == BEST else 75, color=color, zorder=3)
        ax.text(score + .002, pos, f"{score:.4f}", va="center", fontsize=10, color=DARK)
    ax.set_yticks(y, names)
    ax.set_xlim(.70, .81); ax.set_ylim(-.7, len(names)-.3)
    ax.set_xlabel("Macro-F1 (point positions; focused axis)")
    ax.grid(axis="x", alpha=.16)
    note = ("Points show absolute scores on a labeled focused axis; no bar length encodes differences. "
            "All values are development out-of-fold results; architecture and compute budgets differ.")
    finish(fig, "02_model_comparison", "Development model comparison",
           "Pooled fixed-five-class Macro-F1 · same 119 nights, 60 participants, 274,271 valid epochs",
           note, {"models": models},
           [f"runs/{RUNS[n]}/result.json" for n in names] +
           [f"runs/{RUNS[n]}/members.json" for n in
            ("Two-view ensemble", "Three-seed ensemble")], outputs)


def make_class_metrics(conf, sources, outputs):
    support = conf.sum(axis=1)
    recall = np.diag(conf) / support
    scores = f1(conf)
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=.15, right=.83, top=.80, bottom=.19)
    title(fig, "Stage-wise performance",
          "Transition decoder · valid development epochs · fixed class order")
    x = np.arange(5)
    for i in x:
        ax.plot([i, i], [scores[i], recall[i]], color=GREY, alpha=.5, lw=2)
    ax.scatter(x-.055, scores, s=85, color=BLUE, label="F1", zorder=3)
    ax.scatter(x+.055, recall, s=85, color=GOLD, marker="D", label="Recall", zorder=3)
    for i, n in enumerate(support):
        ax.text(i, .045, f"n={n:,}", ha="center", color=GREY, fontsize=9)
    ax.set_xticks(x, CLASS); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Per-class score")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .77))
    ax.grid(axis="y", alpha=.16)
    finish(fig, "03_class_metrics", "Stage-wise performance",
           "Transition decoder · valid development epochs · fixed class order",
           "Support is reference-valid epochs; F1 and recall are computed from the displayed pooled confusion matrix.",
           {"class_order": CLASS, "support": support.tolist(),
            "f1": scores.tolist(), "recall": recall.tolist()},
           staging_sources(sources), outputs)


def make_participants(spread, sources, outputs):
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.6))
    fig.subplots_adjust(left=.09, right=.95, top=.80, bottom=.20, wspace=.30)
    title(fig, "Participant spread and paired change",
          "Anonymous participant bins; both models exclude the scored participant")
    for ax, field, heading in zip(
            axes, ("transition_macro_f1", "paired_delta"),
            ("A  Transition Macro-F1", "B  Transition − direct EEG + EOG")):
        for x, cohort in enumerate(COHORT):
            distribution_marks(ax, spread[cohort][field], x, COHORT[cohort])
        ax.set_xticks([0, 1], ["SC", "ST"])
        ax.set_title(heading, loc="left")
        ax.grid(axis="y", alpha=.16)
    axes[0].set_ylim(.45, .95); axes[0].set_ylabel("Participant Macro-F1")
    axes[1].axhline(0, color=DARK, lw=1)
    axes[1].set_ylabel("Paired participant Macro-F1 difference")
    finish(fig, "04_participant_spread", "Participant spread and paired change",
           "Anonymous participant bins; both models exclude the scored participant",
           "60 participants; bubbles encode counts per anonymous bin. Per-participant F1 is the five-class mean. "
           "Median is open circle; thick line spans IQR. "
           "Paired differences are descriptive development comparisons.",
           {"anonymous_distributions": spread, "comparison": [RUNS[BEST], RUNS[DIRECT]]},
           staging_sources(sources, include_direct=True), outputs)


def make_score_agreement(agreement, summary, sources, outputs):
    label = "EEG + EOG"
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7))
    fig.subplots_adjust(left=.065, right=.98, top=.77, bottom=.21, wspace=.29)
    title(fig, "Experimental score and sleep-duration agreement",
          "EEG + EOG · 61 eligible recording windows / 40 participants")
    for ax, (metric, (_, units)) in zip(axes, ENDPOINTS.items()):
        bins = agreement[metric]["binned_pairs"]
        lo, hi = agreement[metric]["bounds"]
        ax.plot([lo, hi], [lo, hi], color=DARK, lw=1, linestyle="--", label="Identity")
        ax.scatter([point["x"] for point in bins], [point["y"] for point in bins],
                   s=[15+8*point["count"] for point in bins], color=BLUE, alpha=.58)
        if metric == "tst_minutes":
            ax.axvline(420, color=GOLD, lw=1, linestyle=":", label="420-min design target")
            ax.axhline(420, color=GOLD, lw=1, linestyle=":")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Reference {units}"); ax.set_ylabel(f"Predicted {units}")
        ax.set_title(units.split(" · ")[0], loc="left")
        s = summary[label][metric]
        ax.text(.02, .98, f"Participant-balanced MAE {s['mae']:.1f}\nBias {s['bias']:+.1f}",
                transform=ax.transAxes, va="top", fontsize=9, color=DARK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .84})
    finish(fig, "05_score_agreement", "Experimental score and sleep-duration agreement",
           "EEG + EOG · 61 eligible recording windows / 40 participants",
           "Bubbles encode anonymous paired-window bin counts. Identity is descriptive, not a clinical agreement limit. "
           "420 min is the provisional score-design TST target; windows are not verified time-in-bed or whole nights.",
           {"system": label, "binned_endpoints": agreement,
            "participant_balanced_summary": summary[label], "design_tst_target_minutes": 420},
           ["reports/provisional-phase2-three-sensor/recording-summaries.csv",
            "runs/provisional-score-study-f1c65dcf6215/result.json",
            "runs/provisional-score-study-f1c65dcf6215/config.json"], outputs)


def make_score_errors(summary, sources, outputs):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8))
    fig.subplots_adjust(left=.075, right=.96, top=.78, bottom=.22, wspace=.38)
    title(fig, "Participant-balanced endpoint error",
          "61 eligible recording windows / 40 participants · equal weight per participant")
    names = list(SCORE_SYSTEMS)
    y = np.arange(3)[::-1]
    colors = [GREY, BLUE, GOLD]
    for ax, (metric, (_, units)) in zip(axes, ENDPOINTS.items()):
        for pos, name, color in zip(y, names, colors):
            s = summary[name][metric]
            low, high = s["mae_ci95"]
            ax.plot([low, high], [pos, pos], color=color, lw=2.5)
            ax.scatter(s["mae"], pos, color=color, s=64, zorder=3)
            ax.text(s["mae"], pos+.18, f"{s['mae']:.1f}", ha="center", fontsize=9)
        ax.set_yticks(y, names)
        ax.set_ylim(-.5, 2.55)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"Mean absolute error · {units.split(' · ')[-1]}")
        ax.set_title(units.split(" · ")[0], loc="left")
        ax.grid(axis="x", alpha=.16)
    finish(fig, "06_score_errors", "Participant-balanced endpoint error",
           "61 eligible recording windows / 40 participants · equal weight per participant",
           "Dots are participant-balanced MAE; lines are saved 95% participant-bootstrap intervals (2,000 draws). "
           "Score is experimental points; TST and WASO are minutes. Development only.",
           {"systems": summary, "bootstrap_draws": 2000},
           ["reports/provisional-phase2-three-sensor/recording-summaries.csv",
            "runs/provisional-score-study-f1c65dcf6215/result.json"], outputs)


def make_cohort_recall(cohort_conf, sources, outputs):
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    fig.subplots_adjust(left=.11, right=.84, top=.79, bottom=.19)
    title(fig, "Reference-stage recall by cohort",
          "Transition decoder · valid development epochs · descriptive SC/ST split")
    x = np.arange(5)
    data = {}
    for shift, cohort, marker in ((-.10, "SC", "o"), (.10, "ST", "D")):
        conf = np.asarray(cohort_conf[cohort])
        support = conf.sum(axis=1)
        recall = np.divide(np.diag(conf), support, out=np.zeros(5), where=support != 0)
        data[cohort] = {"recall": recall.tolist(), "support": support.tolist(),
                        "evaluated_epochs": int(conf.sum())}
        ax.scatter(x+shift, recall, s=85, color=COHORT[cohort], marker=marker,
                   label=f"{cohort} · {int(conf.sum()):,} epochs")
    for i in x:
        ax.plot([i-.1, i+.1], [data["SC"]["recall"][i], data["ST"]["recall"][i]],
                color=GREY, alpha=.55, lw=1.6)
    ax.set_xticks(x, CLASS); ax.set_ylim(.35, 1.01)
    ax.set_ylabel("Recall among valid reference epochs")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .7))
    ax.grid(axis="y", alpha=.16)
    finish(fig, "07_cohort_recall", "Reference-stage recall by cohort",
           "Transition decoder · valid development epochs · descriptive SC/ST split",
           "SC and ST supports differ; these are epoch-pooled cohort values, not participant-balanced estimates.",
           {"class_order": CLASS, "cohorts": data},
           staging_sources(sources), outputs)


def self_test() -> None:
    confusion = np.diag([2, 1, 0, 0, 0])
    if not np.array_equal(f1(confusion), [1, 1, 0, 0, 0]):
        raise AssertionError("Absent classes must have zero F1")
    if histogram([.2, .8], [0, .5, 1])["bin_counts"] != [1, 1]:
        raise AssertionError("Histogram count changed")
    if binned_pairs([.2, .3], [.2, .3], [0, .5, 1], [0, .5, 1])[0]["count"] != 2:
        raise AssertionError("Paired-bin count changed")
    for invalid in (lambda: histogram([float("nan")], [0, 1]),
                    lambda: binned_pairs([2], [.3], [0, 1], [0, 1])):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid aggregate was accepted")
    print("Synthetic aggregate and failure checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-summary", type=Path,
                        help="Render public aggregate summary without reading prediction or truth files")
    parser.add_argument("--self-test", action="store_true",
                        help="Run small offline aggregation and failure checks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.from_summary:
        packed = json.loads(args.from_summary.read_text(encoding="utf-8"))
        if (packed.get("schema_version") != "research-figure-summary-v1" or
                packed.get("builder_sha256") != script_sha or
                packed.get("class_order") != list(CLASS) or
                packed.get("scope") != "development_out_of_fold"):
            raise ValueError("Incompatible published aggregate figure summary")
        figures = packed["figure_data"]
        sources = packed["source_sha256"]
        summary_path = args.from_summary
    else:
        private_sources: dict[str, str] = {}
        runs = load_runs(private_sources)
        nights, conf, cohort_conf, participant_conf = nights_and_confusion(runs, private_sources)
        details, score_summary = score_rows(private_sources)
        figures = aggregate_figures(nights, conf, cohort_conf, participant_conf,
                                    runs, details, score_summary)
        sources = public_sources(private_sources)
        packed = {
            "schema_version": "research-figure-summary-v1",
            "builder_sha256": script_sha, "class_order": list(CLASS),
            "scope": "development_out_of_fold",
            "dataset": "Sleep-EDF Expanded local development cohort; no EDF content exported",
            "split_id": read_json(ROOT / "runs" / RUNS[BEST] / "config.json",
                                  private_sources)["split_id"],
            "figure_data": figures, "source_sha256": sources,
        }
        summary_path = OUT / "summary.json"
        summary_path.write_text(
            json.dumps(packed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    outputs: dict[str, dict] = {}
    overview = figures["overview"]
    conf = np.asarray(overview["confusion"])
    make_overview(overview, sources, outputs)
    make_models(figures["models"], sources, outputs)
    make_class_metrics(conf, sources, outputs)
    make_participants(figures["participant_distributions"], sources, outputs)
    make_score_agreement(figures["score_agreement"], figures["score_summary"], sources, outputs)
    make_score_errors(figures["score_summary"], sources, outputs)
    make_cohort_recall(figures["cohort_confusion"], sources, outputs)
    manifest = {
        "schema_version": "research-figures-v1", "render_command":
        r"$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; .\.venvs\research\Scripts\python.exe tools\build_research_figures.py --from-summary reports\publication-figures\summary.json",
        "script_sha256": script_sha,
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "class_order": CLASS, "development_only": True,
        "split_id": packed["split_id"],
        "gate_A": "NOT_RUN", "gate_B": "NOT_RUN",
        "pooled_metric_spotcheck": {
            "recordings": overview["nights"], "participants": overview["participants"],
            "evaluated_epochs": int(conf.sum()), "accuracy": float(np.trace(conf)/conf.sum()),
            "macro_f1": float(f1(conf).mean()),
            "confusion": conf.tolist(), "matches_saved_result": True},
        "figures": outputs, "source_sha256": dict(sorted(sources.items())),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Rendered {len(outputs)} development figures; {len(sources)} public source digests; "
          f"Macro-F1={manifest['pooled_metric_spotcheck']['macro_f1']:.9f}")


if __name__ == "__main__":
    main()
