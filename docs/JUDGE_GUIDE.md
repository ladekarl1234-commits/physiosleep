# Start here: a judge's review guide

[Documentation index](README.md) Â· [Repository home](../README.md)

## The project in one minute

**Problem:** make sleep-stage predictions from PSG EEG/EOG signals inspectable and reproducible, then study whether simpler measurements can support useful sleep summaries. **Current deliverable:** an offline five-stage research pipeline, measured development comparisons, reproducible evaluation contracts, a specialist review demonstration and an acquisition concept. **Intended reviewer:** a sleep specialist or research engineer; this is not a diagnostic device.

The best completed system is an integration of existing physiological features and boosted trees, three-seed probability averaging and a train-only transition prior. Its value currently lies in the traceable evaluation, preserved timing/invalid epochs, leakage controls and tested implementation. A new neural architecture, state-of-the-art performance, clinical utility or a validated sensor device is not established.

## Five-minute route

| Time | Open | Decision to make |
|---|---|---|
| 0â€“1 min | [Pipeline](PIPELINE.md) | Can I trace signals to stages without reference-label leakage? |
| 1â€“2 min | [Job ledger](EXPERIMENTS.md) | Which fits completed, which tests passed, which jobs failed and which exploratory tests completed and which formal criteria remain unmet? |
| 2â€“3 min | [Local results](RESULTS.md) | Do Macro-F1, N1, denominators and failed score targets support the claims? |
| 3â€“4 min | [Excel comparison](../literature/COMPARISON.md) | Are published results attributed with their different protocols rather than treated as matched wins? |
| 4â€“5 min | [Grades](JUDGING.md) and [claims](CLAIMS.md) | What credit is supported, what is missing and what would change the decision? |

For a deeper technical review, inspect the [16 ADRs](adr/README.md), run the [public checks](DEVELOPER_GUIDE.md), inspect the [fixed audit criteria](REPORT.md#mandatory-benchmark-and-audit-protocol), and follow the claim links to aggregate evidence. Hardware is a separate [optional review branch](HARDWARE.md).

## Current scorecard

| Perspective | Initial â†’ latest | Review cutoff | Interpretation |
|---|---:|---|---|
| Sleep specialist | **69 â†’ 71 / 100** | 25 September 2026 | Simulated scientific/aggregate review; no clinical certification |
| Data / ML | **53 â†’ 59 / 100** | 25 September 2026 | Simulated scientific/reproducibility review; no official competition grade |

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

1. Run `python tools/verify_public_metrics.py`: compare the recomputed Macro-F1 and epoch total with the [result table](RESULTS.md).
2. Run `python -m unittest discover -s tests -v`: these are software and contract checks, not model-accuracy tests.
3. Open one literature row's source and compare its cohort/window/metric to ours before making a performance claim.
4. Read the failed score endpoints and U-Time terminal-state entry, then check whether the headline text preserves those limitations.

Judge the measured evidence and inspectability now; reserve confirmation, clinical and device credit for the required future measurements. See the [roadmap](ROAD_TO_90.md) for evidence needed to improve the existing scientific grades.
