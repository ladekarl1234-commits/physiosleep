"""Render the public aggregate epoch snapshot; never reads data or checkpoints."""
from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "reports/training-progress"
    meta = json.loads((base / "snapshot.json").read_text(encoding="utf-8"))
    with (base / "attnsleep-epochs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["train_loss"]) for row in rows]
    minutes = [float(row["wall_seconds"])/60 for row in rows]
    assert epochs == list(range(1, meta["completed_epochs"]+1))
    assert all(math.isfinite(x) and x >= 0 for x in loss+minutes)
    assert meta["required_epochs"] == 100 and meta["fold"] == 0 and meta["seed"] == 17
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, values, title, unit in zip(
            axes, (loss, minutes), ("Training loss", "Observed epoch runtime"),
            ("Weighted cross-entropy", "Minutes per completed epoch")):
        ax.plot(epochs, values, color="#17629a", linewidth=1.8)
        ax.scatter([epochs[-1]], [values[-1]], color="#17629a", s=26, zorder=3)
        ax.axvspan(epochs[-1]+.5, 100, color="#eeeeee", zorder=0)
        ax.text((epochs[-1]+100)/2, max(values)*.65, "Not yet observed",
                color="#666666", ha="center", fontsize=10)
        ax.set(xlim=(1, 100), ylim=(0, max(values)*1.15), xlabel="Native training epoch",
               ylabel=unit, title=title, xticks=[1, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle(f"AttnSleep development fit: {epochs[-1]}/100 epochs observed",
                 x=.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(.07, .87, f"Fold 0 of 5 · seed 17 · batch 128 · snapshot {meta['observed_at_utc'][:16]} UTC",
             color="#444444")
    fig.text(.07, .045, "Optimization diagnostics only. No validation accuracy or audit result is shown.",
             color="#444444", fontsize=10)
    fig.subplots_adjust(left=.075, right=.98, top=.75, bottom=.20, wspace=.30)
    for suffix in ("png", "svg"):
        fig.savefig(base / ("attnsleep-progress."+suffix), dpi=170, facecolor="white")
    svg = base / "attnsleep-progress.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",
                   encoding="utf-8", newline="\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
