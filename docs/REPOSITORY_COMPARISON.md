# How this repository relates to established sleep-staging code

[Local results](RESULTS.md) · [Mandatory comparator registry](BASELINES.md) · [Literature rows](../literature/COMPARISON.md)

This page compares **software purpose and our observed execution**, using the linked upstream repositories as primary sources. It is not a numeric leaderboard. A paper's reported score, a repository's available implementation and a matched run on our people are three different kinds of evidence.

| Upstream project | What its own repository provides | What PhysioSleep actually evaluated locally | Fair-comparison boundary |
|---|---|---|---|
| [YASA](https://github.com/raphaelvallat/yasa) | PSG analysis, physiological features and automatic staging, including a released pretrained classifier | **Locally trained** LightGBM on YASA EEG/EOG features: 0.786267 seed-17 development Macro-F1; a D60 refit scored 0.759022 / 0.798875 on Audit 1 / 2 | These numbers do **not** evaluate YASA's released pretrained classifier. The native YASA mandatory slot is incomplete. |
| [SLEEPYLAND](https://github.com/biomedical-signal-processing/sleepyland) | A framework for sleep-data analysis and evaluation of several staging models | Clean local SLEEPYLAND/YASA feature route: **0.783662** on the same 60-person development folds, with five saved-fold parity checks | It pools Fpz-Cz EEG+EOG and Pz-Oz EEG+EOG feature groups, so the channel inputs differ from our two-channel control. This is a development compatibility route, not proof that every packaged model was reproduced or beaten. |
| [U-Time / U-Sleep](https://github.com/perslev/U-Time) | Official training and evaluation software for U-Time and U-Sleep architectures | U-Sleep signal preparation and bounded numerical preflights; **no completed model fit**. The first U-Time inner fit failed at its state commit. | There is no matched local Macro-F1 to compare. Published U-Sleep results use their own datasets and protocol. |
| [AttnSleep](https://github.com/emadeldeen24/AttnSleep) | Attention-based single-channel EEG staging code, preparation and fold-training instructions | A local fold-0 training checkpoint reached **65/100 epochs**, then stopped | Training loss or a partial fold cannot be substituted for complete held-person predictions or a Macro-F1 result. |

PhysioSleep's strongest **completed development** pipeline averaged three locally trained EEG+EOG models and applied a prior fitted only on training people; it scored **0.789779** pooled Macro-F1. That is **+0.003512** over its fixed seed-17 control and **+0.006117** over the local SLEEPYLAND/YASA route on the same development people. It was **not** the model tested in Audit 1 or Audit 2. The formal eleven-slot comparison was stopped before completion, so no claim of superiority over the upstream algorithms follows.

## What to inspect in a fair future comparison

Use identical participant membership, complete 30-second grids, five-class label order, reference-valid masks and pooled-count metric arithmetic. Verify each upstream code/weight license and training lineage, channel mapping, model-fit adequacy, checkpoint selection and prediction coverage. Participant-resampled **paired** differences and the predeclared +0.02 threshold would then address the historical gate; the original A/B people are already consumed, so confirmation needs fresh people. [Protocol and registry](BASELINES.md) · [Current decision](FINAL_REPORT.md#comparison-limitations-and-decision).

The supplied workbook's [56 experimental settings](../literature/COMPARISON.md) add research context, with row-level source attribution. Its cohorts, cropping, pretraining and aggregation vary, so its numbers cannot fill missing matched local results. No performance claim on this page comes from model memory alone.
