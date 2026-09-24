# PhysioSleep development figures

Seven static figures were rendered from saved local **development out-of-fold** results. Gate A and Gate B remain NOT_RUN. The experimental score has no clinical validation; its recording windows are not verified time in bed or whole nights.

| Figure | Source and denominator |
|---|---|
| [01 Night overview](01_night_overview.png) | Transition-decoded predictions for 119 nights / 60 people; valid reference epochs only. Confidence panel uses the **direct seed-17 EEG+EOG model**, not decoder posterior probabilities. |
| [02 Model comparison](02_model_comparison.png) | Eight saved development runs, including distinct two-view and three-seed ensembles, on the same 119 nights and 274,271 valid epochs; pooled fixed-five-class Macro-F1. Ensemble member identities are checked against saved manifests. Compute budgets differ. |
| [03 Class metrics](03_class_metrics.png) | Transition decoder; pooled confusion on 274,271 valid epochs, class order W, N1, N2, N3, REM. |
| [04 Participant spread](04_participant_spread.png) | Transition decoder versus direct seed-17 EEG+EOG; 60 participant-level Macro-F1 values summarized into anonymous cohort bins. |
| [05 Score agreement](05_score_agreement.png) | EEG+EOG experimental score, TST and WASO; 61 eligible recording windows / 40 people. Coordinates are binned. The 420-minute TST line is a provisional score-design target. |
| [06 Score errors](06_score_errors.png) | EOG, EEG, and EEG+EOG participant-balanced MAE; saved participant-bootstrap 95% intervals, 2,000 draws, on the same 61 windows / 40 people. |
| [07 Cohort recall](07_cohort_recall.png) | Transition decoder; valid-epoch SC/ST recall with unequal cohort supports. |

Each figure has an SVG and a matching JSON with the exact aggregate values and source-group references. [summary.json](summary.json) is sufficient to regenerate every figure without prediction, truth, EDF, or model files. [manifest.json](manifest.json) includes the builder hash, fixed split identity, figure list, metric spot-check and SHA-256 source digests. Prediction payloads and sidecars are hashed as anonymous 119-pair groups; truth payloads as one anonymous 119-file group. Recording names and participant IDs are absent.

To regenerate from the public aggregate summary in the project root, using the installed research Python:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venvs\research\Scripts\python.exe tools\build_research_figures.py --from-summary reports\publication-figures\summary.json
```

To recompute the aggregate summary and verify it against the saved private development outputs, omit `--from-summary`. That source mode reads saved result, prediction, truth, and score CSV artifacts but never reads raw EDFs, fits a model, or downloads data.

Panel 01D uses all reference-valid recording epochs as its stage-fraction denominator, not sleep-only epochs.
