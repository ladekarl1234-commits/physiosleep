# Audit A and B: exploratory evaluation guide

[Measured results and class table](../reports/exploratory-audits-v1/report.md) | [Numeric aggregate](../reports/exploratory-audits-v1/aggregate.json) | [Owner decision](adr/0016-exploratory-audits.md)

## What was tested

The selected checkpoint is a LightGBM classifier trained on all 60 development participants using native YASA physiological features from Fpz-Cz EEG and horizontal EOG. It uses seed 17, learning rate 0.1 and 400 boosting rounds. Released pretrained YASA weights are not used. This single-model control differs from the three-seed ensemble with a transition prior that achieved the best development Macro-F1 of 0.789779.

The two test cohorts each contain 39 recordings from 20 previously held-out participants: 16 SC and four ST. Every participant's nights remain together. These are additional people from the same Sleep-EDF source and acquisition cohorts, not an external hospital or device-validation dataset.

## How evaluation works

1. Freeze the checkpoint, fit ancestry, implementation, protocol, owner authorization and bootstrap recipe.
2. Run signal-only inference for all 78 recordings. The prediction interface receives no Hypnogram or reference labels.
3. Save a hard label and five-class probability vector for every complete original 30-second epoch, together with a successful subprocess receipt and content hashes.
4. Require all 78 predictions and receipts before opening either phase's reference labels. Record the actual reference-access event.
5. Apply the original reference-valid mask in the evaluator. Invalid epochs remain counted at their original timeline positions; models cannot improve their score by dropping difficult epochs.
6. Recompute fixed-five-class pooled Macro-F1, accuracy, kappa and class metrics. An independent verifier checks the saved prediction/reference artifacts and reproduces the calculations.

All class metrics use W, N1, N2, N3, REM in that order. For a stage, `F1 = 2TP / (2TP + FP + FN)`; Macro-F1 is the unweighted mean of the five class F1 scores **after** pooling confusion counts across reference-valid epochs. A zero-denominator class contributes zero. The score ranges from 0 to 1 and gives each stage equal weight, revealing weak N1 recognition despite Wake-dominated accuracy. This is not an average of per-night or per-person F1 scores. The [final report](FINAL_REPORT.md#how-macro-f1-is-calculated-and-why-it-is-primary) has the complete equation and a worked Audit A example.

## Uncertainty and interpretation

Separate approximate 95% percentile intervals use 10,000 participant bootstrap draws within SC and ST, retaining all nights of each sampled person. PCG64 seeds are 2026092301 for A and 2026092302 for B. The intervals condition on the fixed checkpoint and do not include training or selection variability. Four ST participants per phase cannot support a strong ST-specific conclusion.

N1 F1 is **0.469 / 0.499**. Wake accounts for **63.4% / 63.4%** of valid epochs. High overall accuracy does not imply uniform recognition of every sleep stage.

A and B are two descriptions of the same frozen control on separate participant sets. They are not two competing models, and their difference is not a matched algorithm comparison. The workbook literature scores and development ensemble scores use different evaluation conditions and cannot supply missing same-audit baseline predictions.

## What the status means

The owner explicitly chose immediate exploratory evaluation after being informed that it would consume both reserved holdouts. A completed exploratory evaluation means that predictions were compared with actual reference labels. It does not mean the all-baseline +0.02 superiority gate passed.

The formal criteria remain unchanged, but the original A and B cohorts are retired from future confirmation. Fresh participants and a prospective confirmation protocol are needed. Rebranding the same cohorts, selecting the best model after seeing their scores, or treating B as a retry would not restore that independence.

## Independent verification and reviewer grades

[The saved-array verification receipt](../evidence/exploratory-audits-verification.json) records PASS for all 78 executions, full grids, metrics and 10,000-draw intervals. The verifier recomputed results independently rather than calling the project scoring function. The separate simulated reviews are **sleep specialist 73/100** and **data/ML 58/100** under unchanged rubrics; see [the grading rationale](JUDGING.md). These are project-evidence grades, not accuracy percentages.

## Inspect and reproduce

```bash
python tools/verify_exploratory_metrics.py
python tools/build_exploratory_report.py --aggregate reports/exploratory-audits-v1/aggregate.json --output-dir regenerated-audits
```

The first command uses the standard library to recompute point metrics from public pooled confusion counts and check figure hashes. The second requires NumPy and Matplotlib and recreates the report and figures from the sanitized aggregate. It does not rerun inference or bootstrap sampling.

Participant-linked inputs, predictions, cluster counts and private model files remain excluded from the public repository. Therefore public point-metric replay is possible, while full inference and participant-bootstrap reproduction require the protected artifacts. Source-result and checkpoint hashes identify the exact local evidence without publishing those payloads.

![A and B summary metrics with descriptive Macro-F1 intervals](../reports/exploratory-audits-v1/summary.png)

![Reference-row-normalized confusion matrices](../reports/exploratory-audits-v1/confusion.png)

![Per-class F1 for both exploratory phases](../reports/exploratory-audits-v1/class-f1.png)
