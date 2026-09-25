"""Plot the three public evaluation stages from sanitized aggregate evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "evaluation-overview"


def main():
    development = json.loads((ROOT / "reports/judge-guide/local-comparison.json").read_text(encoding="utf-8"))
    audits = json.loads((ROOT / "reports/exploratory-audits-v1/aggregate.json").read_text(encoding="utf-8"))
    development_metrics = json.loads((ROOT / "evidence/development-metrics.json").read_text(encoding="utf-8"))
    models = {row["name"]: row for row in development["models"]}
    assert development["scope"] == "adaptive_development_only"
    assert audits["status"] == "EXPLORATORY_EVALUATED" and audits["gate_status"] == "NOT_RUN"
    assert (development["participants"], development["recordings"]) == (60, 119)
    assert all((audits["phases"][key]["participant_count"],
                audits["phases"][key]["recording_count"]) == (20, 39) for key in ("A", "B"))

    direct = models["EEG + EOG · seed 17"]
    best = models["Transition decoder"]
    audit_a, audit_b = (audits["phases"][key] for key in ("A", "B"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none",
                         "svg.hashsalt": "physiosleep-evaluation-overview-v1"})
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.1), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#f7f9fb")
    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e2e9ee")
        ax.set_xticks([0, .2, .4, .6, .8, 1])

    ax = axes[0]
    ax.barh([1, 0], [direct["macro_f1"], best["macro_f1"]],
            color=["#7493a7", "#176b9a"], height=.56)
    ax.set_yticks([1, 0], ["Seed-17 EEG + EOG", "Three-seed + transition"])
    ax.set_ylim(-.7, 1.75)
    ax.set_title("01  Development · 60 people / 119 recordings · five held-person folds",
                 loc="left", fontweight="bold", pad=14)
    for y, row in ((1, direct), (0, best)):
        ax.text(row["macro_f1"] + .015, y, f'{row["macro_f1"]:.4f}', va="center", fontweight="bold")
    ax.text(0, 1.46, "Out-of-fold predictions; recipe selection used these people",
            color="#526575", fontsize=9)

    ax = axes[1]
    for y, phase, row, color in ((1, "Audit 1", audit_a, "#bd7542"),
                                 (0, "Audit 2", audit_b, "#814e7a")):
        ax.barh(y, row["macro_f1"], color=color, height=.56)
        ci = row["descriptive_macro_f1_interval"]
        ax.errorbar(row["macro_f1"], y,
                    xerr=[[row["macro_f1"] - ci["lower"]], [ci["upper"] - row["macro_f1"]]],
                    fmt="none", ecolor="#182e3b", capsize=5, elinewidth=1.6)
        ax.text(.84, y, f'{row["macro_f1"]:.4f}', va="center", fontweight="bold")
    ax.set_yticks([1, 0], ["Audit 1 (A)", "Audit 2 (B)"])
    ax.set_ylim(-.7, 1.75)
    ax.set_title("02–03  Exploratory audits · 20 new people / 39 recordings each",
                 loc="left", fontweight="bold", pad=14)
    ax.text(0, 1.46, "One unchanged D60 seed-17 checkpoint; separate descriptive 95% intervals",
            color="#526575", fontsize=9)
    ax.set_xlabel("Pooled five-class Macro-F1 · reference-valid 30-second epochs")

    fig.suptitle("PhysioSleep | evidence by evaluation stage", x=.07, ha="left",
                 fontsize=16, fontweight="bold", color="#17334a")
    fig.text(.08, .015,
             "Development bars use fold-trained models; both audit bars use one D60 refit. Different people and model fits: no A-to-B improvement claim.",
             color="#405467", fontsize=8)
    fig.subplots_adjust(left=.25, right=.94, top=.84, bottom=.13, hspace=.62)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        target = OUT / f"stages.{suffix}"
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor(),
                    metadata={"Software": "PhysioSleep public aggregate overview"} if suffix == "png"
                             else {"Date": None, "Creator": "PhysioSleep public aggregate overview"})
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    confusion = development_metrics["confusion"]
    stages = development_metrics["class_order"]
    dev_f1 = []
    for i in range(5):
        reference = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        dev_f1.append(2 * confusion[i][i] / (reference + predicted)
                      if reference + predicted else 0)
    assert stages == ["W", "N1", "N2", "N3", "REM"]
    assert abs(sum(dev_f1) / 5 - best["macro_f1"]) < 1e-12
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("white")
    positions = range(5)
    series = (("Development best", dev_f1, -.24, "#176b9a"),
              ("Audit 1 · D60 control", audit_a["per_class_f1"], 0, "#bd7542"),
              ("Audit 2 · same control", audit_b["per_class_f1"], .24, "#814e7a"))
    for label, values, offset, color in series:
        ax.bar([i + offset for i in positions], values, width=.22,
               label=label, color=color)
    ax.set_xticks(list(positions), stages)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 from pooled valid-epoch confusion counts")
    ax.set_title("Five-stage recognition across the three evaluation stages",
                 loc="left", fontweight="bold", pad=14)
    ax.grid(axis="y", color="#e2e9ee")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.03))
    fig.text(.12, .015,
             "Development best is a different model pipeline. Audit 1 and 2 use one unchanged checkpoint on different people.",
             color="#405467", fontsize=8)
    fig.subplots_adjust(left=.12, right=.96, top=.81, bottom=.17)
    for suffix in ("png", "svg"):
        target = OUT / f"class-stages.{suffix}"
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor(),
                    metadata={"Software": "PhysioSleep public aggregate overview"} if suffix == "png"
                             else {"Date": None, "Creator": "PhysioSleep public aggregate overview"})
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"development_control": direct["macro_f1"],
                      "development_best": best["macro_f1"],
                      "audit_1": audit_a["macro_f1"], "audit_2": audit_b["macro_f1"],
                      "development_best_class_f1": dev_f1}, indent=2))


if __name__ == "__main__":
    main()
