# Start here: a judge's review guide

The [final English report](FINAL_REPORT.md) contains the completed exploratory A/B results, exact Macro-F1 calculation and current evidence limits. [Dev/train → Audit 1 → Audit 2](EVALUATION_STAGES.md) names the three evaluation stages and their measured performance. The owner has ended additional baseline testing for this delivery; the historical +0.02 formal gate remains NOT_RUN.

[Documentation index](README.md) | [Repository home](../README.md)

## The project in one minute

**Problem:** make sleep-stage predictions from PSG EEG/EOG signals inspectable and reproducible, then study whether simpler measurements can support useful sleep summaries. **Current deliverable:** an offline five-stage research pipeline, measured development comparisons, reproducible evaluation contracts, a specialist review demonstration and an acquisition concept. **Intended reviewer:** a sleep specialist or research engineer; this is not a diagnostic device.

The best completed system is an integration of existing physiological features and boosted trees, three-seed probability averaging and a train-only transition prior. Its value currently lies in the traceable evaluation, preserved timing/invalid epochs, leakage controls and tested implementation. A new neural architecture, state-of-the-art performance, clinical utility or a validated sensor device is not established.

## Five-minute route

| Time | Open | Decision to make |
|---|---|---|
| 0-1 min | [Pipeline](PIPELINE.md) | Can I trace signals to stages without reference-label leakage? |
| 1-2 min | [Three-stage overview](EVALUATION_STAGES.md) and [job ledger](EXPERIMENTS.md) | Which model and people produced each score, and which jobs remain incomplete? |
| 2-3 min | [Local results](RESULTS.md) | Do Macro-F1, N1, denominators and failed score targets support the claims? |
| 3-4 min | [Local difference and published stage figures](REPOSITORY_COMPARISON.md), then [Excel rows](../literature/COMPARISON.md) | Are measured local runs separated from available code and unmatched paper scores? |
| 4-5 min | [Grades](JUDGING.md) and [claims](CLAIMS.md) | What credit is supported, what is missing and what would change the decision? |

For a deeper technical review, inspect the [17 ADRs](adr/README.md), run the [public checks](DEVELOPER_GUIDE.md), inspect the [historical fixed audit criteria](REPORT.md#historical-mandatory-benchmark-and-audit-protocol), and follow the claim links to aggregate evidence. Hardware is a separate [optional review branch](HARDWARE.md).

## Current scorecard

| Perspective | Latest | Review cutoff | Interpretation |
|---|---:|---|---|
| Sleep specialist | **75 / 100** | 25 September 2026 | Simulated review after test-linked score sensitivity and generated hardware replay; no clinical certification |
| Data / ML | **61 / 100** | 25 September 2026 | Simulated review after publisher-byte, annotation/grid and candidate-lineage checks; no official competition grade |

The [frozen criteria and original records](JUDGING.md) retain individual scores. The documentation review that produced this guide is separate and does not automatically rescore the science. Missing confirmation, external validity, adequate mandatory baselines and physical-device measurements remain substantive gaps.

## Questions to challenge us with

| Question | Answer and evidence |
|---|---|
| Have you beaten every required baseline by +0.02? | **No.** Best gain over the fixed EEG+EOG control is +0.003512. [Local results](RESULTS.md), [eleven-slot registry](BASELINES.md). |
| Is 90.7% accuracy misleading? | Wake is 63.27% of valid epochs and N1 F1 is only 0.4722. Assess fixed-five-class Macro-F1 and the confusion matrix. [Stage analysis](REPORT.md#development-experiments-and-results). |
| Did the same person appear in fit and validation? | Every person's nights stay together; five 48/12-person development folds. Recipe selection still uses development, and audits are procedural holdouts on a shared host. [Pipeline](PIPELINE.md). |
| What are the actual audit scores? | A **0.7590**, B **0.7989** Macro-F1 for the single D60 control; [full exploratory results](EXPLORATORY_AUDITS.md). Required native comparators remain incomplete, so this is not a formal gate pass. |
| What does the Excel prove? | It gives research context, not matched evidence. Each of its 56 numeric rows retains source and protocol. [Literature table](../literature/COMPARISON.md). |
| Why only 61/119 windows for scores? | The unchanged charter requires at least seven hours, complete coverage and some sleep. None has independently established whole-night boundaries. [Score definitions and failures](RESULTS.md#experimental-score). |
| Can I reproduce the results? | Public code tests, aggregate arithmetic and plots: yes. Exact private training: not from this clone alone; original data and frozen controlled artifacts are excluded. [Developer guide](DEVELOPER_GUIDE.md). |
| Does the proposed device produce these results? | No. Results come from Sleep-EDF; hardware images are concepts and engineering checks use synthetic signals. [Hardware evidence](HARDWARE.md). |

## A practical review exercise

1. Run `python tools/verify_public_metrics.py` and `python tools/verify_exploratory_metrics.py`: compare the recomputed development and Audit 1/2 Macro-F1 values with the [stage table](EVALUATION_STAGES.md).
2. Run `python -m unittest discover -s tests -v`: these are software and contract checks, not model-accuracy tests.
3. Open one literature row's source and compare its cohort/window/metric to ours before making a performance claim.
4. Read the failed score endpoints and U-Time terminal-state entry, then check whether the headline text preserves those limitations.

Judge the measured evidence and inspectability now; reserve confirmation, clinical and device credit for measurements absent from this delivery. The [fixed-rubric feasibility check](ROAD_TO_90.md) shows why the requested score above 90 cannot be earned while the stopped comparator experiments and missing confirmation evidence remain excluded.
