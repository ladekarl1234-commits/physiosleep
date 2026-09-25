# Exploratory Audit A/B — D60 YASA control

Both reserved holdouts were consumed for an exploratory check. This is a **locally trained LightGBM classifier using YASA EEG+EOG features** (D60 seed 17), not released YASA weights or a new neural model. The separate development three-seed/transition ensemble reported pooled Macro-F1 0.7897786854; it was not evaluated here. These results cannot establish Gate A/B passage or clinical validity; fresh data are required for confirmation.

| Measure | Audit A | Audit B |
|---|---:|---:|
| Recordings / participants | 39 / 20 | 39 / 20 |
| Reference-valid / complete epochs | 91,915 / 92,636 | 91,466 / 93,101 |
| Invalid reference epochs | 721 | 1,635 |
| Pooled five-class Macro-F1 | 0.7590 | 0.7989 |
| Descriptive 95% interval | 0.7245–0.7873 | 0.7816–0.8156 |
| Accuracy | 0.8876 | 0.9124 |
| Cohen's κ | 0.7964 | 0.8413 |

Intervals resample participants within SC and ST (16 and only 4 ST participants per phase), retaining all nights for each sampled person; 10,000 PCG64 draws use the preregistered phase seeds. They are **separate, descriptive 95% intervals** conditional on this frozen checkpoint. They omit training and model-selection variability, with no paired comparator or simultaneous-coverage claim. A and B come from the same Sleep-EDF source and cohort families, not independent external populations.

## Class-level performance

| Stage | A F1 | A recall | B F1 | B recall |
|---|---:|---:|---:|---:|
| W | 0.9680 | 0.9623 | 0.9767 | 0.9700 |
| N1 | 0.4692 | 0.4426 | 0.4988 | 0.5824 |
| N2 | 0.8298 | 0.8302 | 0.8739 | 0.8644 |
| N3 | 0.7458 | 0.8284 | 0.7931 | 0.7398 |
| REM | 0.7823 | 0.8300 | 0.8519 | 0.8595 |

## Figures

- [Row-normalized confusion matrices](confusion.png) ([SVG](confusion.svg)); each row is conditioned on the true stage.
- [Per-class F1](class-f1.png) ([SVG](class-f1.svg)).
- [Pooled summary](summary.png) ([SVG](summary.svg)); only Macro-F1 has descriptive uncertainty bars.

## Provenance and limits

- Control checkpoint SHA-256: `abdc64d3cd736f0a2e6c89fb5843eee06a3af26f64f0abb80120ade794d83860`.
- Frozen selection SHA-256: `0eead74d2dfe8795f91126aa69f9385857011374c682925ba398e5fd7952fb0c`; protocol hash: `sha256:88cd3639d0af311faea4f645de7183d3fd51b83469c5351028218c047bdf80b6`.
- Saved result SHA-256: A `ea587149451af65e5815e70576cfbd9d86ca4a712c1d05c003df21d985209071`; B `366e3155e303207d344e8573d9d10ab4a36bb6d421ac6830d1f2467cafe92843`.
- Fixed class order: W, N1, N2, N3, REM. Scores pool reference-valid original 30-second epochs; invalid epochs remain in the grid and are counted separately. Full-grid prediction coverage was required.
- The public aggregate and figures contain no per-record predictions, truth arrays, participant rows or recording identifiers.
