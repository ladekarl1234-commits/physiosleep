"""Build protocol-labeled comparison figures from public aggregate inputs."""

import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "comparison-extensions"
ORDER = ["W", "N1", "N2", "N3", "REM"]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        target = OUT / f"{stem}.{suffix}"
        metadata = ({"Software": "PhysioSleep public aggregate comparison"} if suffix == "png"
                    else {"Date": None, "Creator": "PhysioSleep public aggregate comparison"})
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor(), metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def local_gap():
    evidence = read("reports/judge-guide/local-comparison.json")
    assert evidence["scope"] == "adaptive_development_only"
    assert (evidence["participants"], evidence["recordings"]) == (60, 119)
    models = {row["name"]: row["macro_f1"] for row in evidence["models"]}
    control = models["EEG + EOG · seed 17"]
    choices = [
        ("EOG only", "EOG", False),
        ("EEG only", "EEG", False),
        ("SLEEPYLAND/YASA pooled groups*", "Sleepyland/YASA pooled groups", True),
        ("Two-view ensemble", "Two-view ensemble", False),
        ("EEG + EOG · seed 101", "EEG + EOG · seed 101", False),
        ("EEG + EOG · seed 43", "EEG + EOG · seed 43", False),
        ("EEG + EOG · seed 17 (control)", "EEG + EOG · seed 17", False),
        ("Three-seed ensemble", "Three-seed ensemble", False),
        ("Three-seed + transition", "Transition decoder", False),
    ]
    labels = [row[0] for row in choices]
    gaps = np.array([100 * (models[key] - control) for _, key, _ in choices])
    colors = ["#176b9a" if value > 0 else "#bd7542" if value < 0 else "#607788" for value in gaps]
    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("white")
    bars = ax.barh(np.arange(len(labels)), gaps, color=colors, height=.67)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#1d3341", linewidth=1.2)
    ax.set_xlim(-7.5, 1.25)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e9ee")
    ax.set_xlabel("Macro-F1 difference from fixed seed-17 control (percentage points)")
    for bar, gap in zip(bars, gaps):
        ax.text(gap + (.12 if gap >= 0 else -.12), bar.get_y() + bar.get_height() / 2,
                f"{gap:+.2f}", va="center", ha="left" if gap >= 0 else "right", fontsize=9)
    ax.set_title("Local Macro-F1 differences versus the fixed control",
                 loc="left", fontsize=14, fontweight="bold", color="#17334a", pad=17)
    fig.text(.31, .06,
             "60 people / 119 recordings / five held-person folds. *Pooled route adds a Pz-Oz EEG feature group.",
             fontsize=8.5, color="#405467")
    fig.text(.31, .031,
             f"Adaptive DEV contrasts; no paired uncertainty. Control Macro-F1 {control:.6f}. Formal +2-point gate NOT_RUN.",
             fontsize=8.5, color="#405467")
    fig.subplots_adjust(left=.31, right=.94, top=.87, bottom=.18)
    save(fig, "local_ablation_gap")


def recall(confusion):
    matrix = np.asarray(confusion, dtype=float)
    assert matrix.shape == (5, 5)
    return 100 * np.diag(matrix) / matrix.sum(axis=1)


def stage_recall():
    dev = read("evidence/development-metrics.json")
    audits = read("reports/exploratory-audits-v1/aggregate.json")
    external = read("literature/stage-recall-context.json")
    assert dev["class_order"] == ORDER
    assert audits["status"] == "EXPLORATORY_EVALUATED" and audits["gate_status"] == "NOT_RUN"
    assert external["class_order"] == ORDER and external["unit"] == "recall_percent"
    assert not external["comparable_to_local"]
    local = [
        ("DEV OOF · ensemble + prior", recall(dev["confusion"]), "#176b9a"),
        ("Audit A · frozen D60 seed 17", recall(audits["phases"]["A"]["confusion"]), "#bd7542"),
        ("Audit B · frozen D60 seed 17", recall(audits["phases"]["B"]["confusion"]), "#814e7a"),
    ]
    outside = [(row["name"], np.asarray(row["values"]), color)
               for row, color in zip(external["external_series"], ["#65938f", "#a97b4b"])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    fig.patch.set_facecolor("#f7f9fb")
    x = np.arange(5)
    for ax, series, width, title in [
        (axes[0], local, .25, "A  PhysioSleep: separate local evaluation stages"),
        (axes[1], outside, .34, "B  Published models: different cohorts/protocols"),
    ]:
        ax.set_facecolor("white")
        n = len(series)
        for i, (label, values, color) in enumerate(series):
            assert len(values) == 5 and np.all((0 <= values) & (values <= 100))
            ax.bar(x + (i - (n - 1) / 2) * width, values, width=width * .91,
                   color=color, label=label)
        ax.set_xticks(x, ORDER)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e9ee")
        ax.set_title(title, fontsize=11.2, fontweight="bold", color="#17334a", pad=13)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -0.29), ncol=1, frameon=False, fontsize=8.7)
    axes[0].set_ylabel("Recall (% of reference epochs in each stage)")
    fig.suptitle("Which sleep stages are recognized?", x=.065, ha="left", fontsize=16,
                 fontweight="bold", color="#17334a")
    fig.text(.065, .075,
             "DEV: selected three-seed transition pipeline, 60 people / 119 nights. Audits: one D60 seed-17 model, 20 people / 39 nights each.",
             fontsize=8.3, color="#405467")
    fig.text(.065, .047,
             "YASA: NSRR, workbook Figure 1C transcription. SleePyCo: Sleep-EDF Figure 3. Published recall aggregation is unverified.",
             fontsize=8.3, color="#405467")
    fig.text(.065, .019, "Cross-panel gaps are not matched model effects.",
             fontsize=8.3, color="#405467")
    fig.subplots_adjust(left=.065, right=.98, top=.82, bottom=.31, wspace=.14)
    save(fig, "stage_recall_context")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "physiosleep-comparison-extensions-v1"})
    local_gap()
    stage_recall()
    paths = [
        "reports/judge-guide/local-comparison.json",
        "evidence/development-metrics.json",
        "reports/exploratory-audits-v1/aggregate.json",
        "literature/stage-recall-context.json",
        "tools/build_comparison_figures.py",
        *(f"reports/comparison-extensions/{stem}.{suffix}"
          for stem in ("local_ablation_gap", "stage_recall_context")
          for suffix in ("png", "svg")),
    ]
    manifest = {"purpose": "Public aggregate comparison-figure provenance",
                "files": [{"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
                          for path in paths]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
