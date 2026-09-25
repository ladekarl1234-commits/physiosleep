# Literature comparison from the supplied Excel

[Back to judge guide](../docs/JUDGE_GUIDE.md) · [Local matched results](../docs/RESULTS.md) · [Paper catalog](PAPERS.md)

**All 56 experiment rows from 22 of the workbook's 31 papers are shown below.** These are attributed workbook values, not locally rerun or independently table-verified results. Numbers can be compared descriptively only: cohorts, Wake cropping, channels, pretraining and aggregation differ. They cannot determine our audit margin.

Download [56 experiment rows as CSV](experiments.csv), [31 papers as CSV](papers.csv), or [structured source rows and provenance](workbook-comparison.json). Every row retains its sheet and row number, protocol, inputs, caveat and source URL. Blank cells are shown as —, never zero. The original workbook remains unchanged; its SHA-256 is `b4ed35cd01d6ff839c1ebe495a99fc55ceae1900af67a2ed87824d8a133909c4`.

## How to read the comparison

- Accuracy and Macro-F1 columns are percentages; kappa is on the original unit scale.
- PhysioSleep's best local **development** values are 90.738% accuracy, 78.978% Macro-F1 and 0.83225 kappa on 119 full recordings / 60 people, using EEG+EOG and no external pretrained weights. They belong in the [local comparison table](../docs/RESULTS.md), not a rank against these papers.
- The workbook's cross-study subtraction columns are intentionally not promoted to effect estimates. Its rounded project inputs are superseded by our exact saved confusion counts.
- YASA's 87.46 is a median nightly accuracy on NSRR, not pooled Sleep-EDF accuracy or Macro-F1. TinySleepNet rows carry the workbook's secondary-table attribution. Pretrained variants retain their separate rows.
- The [earlier source note](README.md) records the narrowly checked YASA aggregation. All rows here retain the conservative **workbook transcription, primary numeric row not independently verified** status.

![Workbook EDF context; unmatched protocols](../reports/judge-guide/literature_context.png)

## Small Sleep-EDF protocols

| Source row | Model | Protocol | Inputs | Accuracy % | Macro-F1 % | Kappa | Caveat / source |
|---|---|---|---|---:|---:|---:|---|
| P02 · 5 | DeepSleepNet | EDF-20; subject-wise CV | EEG | 82 | 76.9 | 0.76 | Fpz-Cz; ±30 min wake window · [source](https://arxiv.org/abs/1703.04046) |
| P04 · 6 | SleepEEGNet | EDF-2013/20; 20-fold CV | EEG | 84.26 | 79.66 | 0.79 | Fpz-Cz; original Table IV · [source](https://arxiv.org/html/1903.02108v1) |
| P05 · 7 | IITNet L=10 | SleepEDF small cohort | EEG | 83.9 | 77.6 | 0.78 | Abstract best sequence length L=10 · [source](https://arxiv.org/abs/1902.06562) |
| P06 · 8 | U-Time | EDF-39; 20-fold CV | EEG | — | 79 | — | Global mean of class F1; rounded to 2 decimals in original paper · [source](https://arxiv.org/html/1910.11162v1) |
| P07 · 9 | TinySleepNet | EDF-20 | EEG | 85.4 | 80.5 | 0.8 | Reported baseline in SalientSleepNet Table 1 / DeepSleepNet-Lite Table VIII · [source](https://arxiv.org/pdf/2108.10600) |
| P08 · 10 | AttnSleep (single epoch) | EDF-20 | EEG | 84.4 | 78.1 | 0.79 | Main model, Table V · [source](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf) |
| P08 · 11 | AttnSleep (3 epochs) | EDF-20 | EEG | 85.6 | 80.9 | 0.8 | Three-epoch variant, Table VI · [source](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf) |
| P09 · 12 | XSleepNet2 | EDF small; ±30 min | EEG + EOG | 86.4 | 80.9 | 0.813 | Two modalities, Table II · [source](https://arxiv.org/pdf/2007.05492) |
| P11 · 13 | L-SeqSleepNet scratch | EDF-20; LOSO; 5 repetitions | EEG | 86.3 | 79.3 | 0.813 | Mean across repeated CV experiments, not median across nights · [source](https://ar5iv.labs.arxiv.org/html/2301.03441) |
| P11 · 14 | L-SeqSleepNet + SHHS pretraining | EDF-20; LOSO; 5 repetitions | EEG | 88.6 | 82.9 | 0.845 | External pretraining; not scratch · [source](https://ar5iv.labs.arxiv.org/html/2301.03441) |
| P12 · 15 | SleePyCo author repository | EDF-2013 SC / 20 | EEG | 86.8 | 81.2 | 0.82 | Value from official repository; distinguish from expanded EDF paper table · [source](https://github.com/gist-ailab/SleePyCo) |
| P13 · 16 | CatBoost scratch | SC20 | EEG + EOG | 86 | 79.7 | 0.807 | No EMG; no direct-transfer initialization · [source](https://arxiv.org/pdf/2207.07753) |
| P13 · 17 | CatBoost direct transfer | SC20 | EEG + EOG + EMG | 86.6 | 81 | 0.816 | Table 3 DT variant; NOT scratch · [source](https://arxiv.org/pdf/2207.07753) |
| P14 · 18 | ZleepAnlystNet | EDF-2013/20 | EEG | 87.02 | 82.09 | 0.8221 | Fpz-Cz · [source](https://www.nature.com/articles/s41598-024-60796-y) |
| P15 · 19 | DeepSleepNet-Lite | EDF-2013/20; ±30 min | EEG | 84 | 78 | 0.78 | All epochs; no uncertainty rejection · [source](https://arxiv.org/pdf/2108.10600) |
| P23 · 20 | SalientSleepNet | EDF-39; 20-fold cross-subject | EEG + EOG | 87.5 | 83 | — | Table 1; kappa not reported there · [source](https://www.ijcai.org/proceedings/2021/0360.pdf) |
| P25 · 21 | Regularized SeqSleepNet | EDF-20; LOSO; 5 repetitions | EEG | 86.2 | 79.3 | 0.811 | Temporal regularization gamma=1e-4 · [source](https://api.unil.ch/iris/server/api/core/bitstreams/4c6f4de9-d179-4888-8311-a6ce23042f7d/content) |
| P30 · 22 | TransSleep — 2022 preprint | EDF; Fpz-Cz | EEG | 86.1 | 81.7 | — | Preprint Table 2; do not silently substitute final-publication values · [source](https://arxiv.org/pdf/2203.12590) |

## Expanded Sleep-EDF protocols

| Source row | Model | Protocol | Inputs | Accuracy % | Macro-F1 % | Kappa | Caveat / source |
|---|---|---|---|---:|---:|---:|---|
| P04 · 5 | SleepEEGNet | EDF-2018; 10-fold CV | EEG | 80.03 | 73.55 | 0.73 | Original Table IV; Fpz-Cz · [source](https://arxiv.org/html/1903.02108v1) |
| P06 · 6 | U-Time | EDF-153; 10-fold CV | EEG | — | 76 | — | Global macro F1, rounded original Table 2 · [source](https://arxiv.org/html/1910.11162v1) |
| P07 · 7 | TinySleepNet | Expanded Sleep-EDF | EEG | 83.1 | 78.1 | 0.77 | Reported baseline: DeepSleepNet-Lite Table VIII · [source](https://arxiv.org/pdf/2108.10600) |
| P08 · 8 | AttnSleep (single epoch) | EDF-78 | EEG | 81.3 | 75.1 | 0.74 | Main model Table V · [source](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf) |
| P08 · 9 | AttnSleep (3 epochs) | EDF-78 | EEG | 82.9 | 78.1 | 0.77 | Three-epoch variant Table VI · [source](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf) |
| P09 · 10 | XSleepNet2 | Expanded EDF; ±30 min | EEG + EOG | 84 | 78.7 | 0.778 | Two modalities; Table II · [source](https://arxiv.org/pdf/2007.05492) |
| P10 · 11 | SleepTransformer scratch | Expanded EDF | EEG | 81.4 | 74.3 | 0.743 | Without SHHS pretraining · [source](https://arxiv.org/pdf/2105.11043) |
| P10 · 12 | SleepTransformer + SHHS | Expanded EDF | EEG | 84.9 | 78.8 | 0.789 | With SHHS pretraining · [source](https://arxiv.org/pdf/2105.11043) |
| P12 · 13 | SleePyCo | Expanded EDF (paper lists 79 subjects) | EEG | 84.6 | 79 | 0.787 | Table 4; record exclusions must be aligned · [source](https://arxiv.org/pdf/2209.09452) |
| P13 · 14 | CatBoost scratch | SC78 | EEG + EOG | 83 | 77.2 | 0.763 | No EMG, Table 3 · [source](https://arxiv.org/pdf/2207.07753) |
| P13 · 15 | CatBoost scratch + EMG | SC78 | EEG + EOG + EMG | 83.1 | 77.5 | 0.766 | Three signals, Table 3 · [source](https://arxiv.org/pdf/2207.07753) |
| P15 · 16 | DeepSleepNet-Lite | EDF-2018; ±30 min | EEG | 80.3 | 75.2 | 0.73 | All epochs; uncertainty-rejected subset excluded · [source](https://arxiv.org/pdf/2108.10600) |
| P23 · 17 | SalientSleepNet | EDF-153; 20-fold cross-subject | EEG + EOG | 84.1 | 79.5 | — | Table 1; not the EDF-39 result · [source](https://www.ijcai.org/proceedings/2021/0360.pdf) |

## Other cohorts and aggregation types

| Source row | Model | Protocol | Inputs | Accuracy % | Macro-F1 % | Kappa | Caveat / source |
|---|---|---|---|---:|---:|---:|---|
| P01 · 5 | YASA full | NSRR / 585 held-out nights | EEG + EOG + EMG + age/sex | 87.46 | — | 0.8188 | Nightwise median, NOT pooled. Exact kappa from supplied Figure 1. · [source](https://elifesciences.org/articles/70092) |
| P01 · 6 | YASA EEG+EOG ablation | NSRR / same 585 nights | EEG + EOG | 86.92 | — | 0.809 | Nightwise median; modality-matched but population and split differ. · [source](https://elifesciences.org/articles/70092) |
| P02 · 7 | DeepSleepNet | MASS | EEG | 86.2 | 81.7 | 0.8 | Pooled metrics; different cohort from EDF. · [source](https://arxiv.org/abs/1703.04046) |
| P03 · 8 | SeqSleepNet original | MASS; 200 participants | EEG + EOG + EMG | 87.1 | — | — | Original abstract accuracy; MF1/kappa not imported without table verification. · [source](https://arxiv.org/abs/1809.10932) |
| P05 · 9 | IITNet L=9 | MASS | EEG | 86.5 | 80.7 | 0.8 | Best L=9 in original abstract. · [source](https://arxiv.org/abs/1902.06562) |
| P05 · 10 | IITNet L=10 | SHHS | EEG | 86.7 | 79.8 | 0.81 | Best L=10 in original abstract. · [source](https://arxiv.org/abs/1902.06562) |
| P09 · 11 | XSleepNet2 two signals | SHHS | EEG + EOG | 88.8 | 81.8 | 0.843 | Table II; pooled. · [source](https://arxiv.org/pdf/2007.05492) |
| P09 · 12 | XSleepNet2 three signals | MASS | EEG + EOG + EMG | 87.6 | 83.8 | 0.823 | Table II; pooled. · [source](https://arxiv.org/pdf/2007.05492) |
| P10 · 13 | SleepTransformer | SHHS | EEG | 87.7 | 80.1 | 0.828 | Pooled, not external-cohort zero-shot result. · [source](https://arxiv.org/pdf/2105.11043) |
| P12 · 14 | SleePyCo | MASS | EEG | 86.8 | 82.5 | 0.811 | Original Table 4. · [source](https://arxiv.org/pdf/2209.09452) |
| P12 · 15 | SleePyCo | Physio2018 | EEG | 80.9 | 78.9 | 0.737 | Original Table 4. · [source](https://arxiv.org/pdf/2209.09452) |
| P12 · 16 | SleePyCo | SHHS | EEG | 87.9 | 80.7 | 0.83 | Original Table 4. · [source](https://arxiv.org/pdf/2209.09452) |
| P16 · 17 | SimpleSleepNet | DOD-H | Multichannel PSG | 89.9 | — | 0.846 | Mean recording accuracy/kappa; reported F1=89.9 not entered as macro. · [source](https://arxiv.org/pdf/1911.03221) |
| P16 · 18 | SimpleSleepNet | DOD-O | Multichannel PSG | 88.7 | — | 0.823 | Mean recording accuracy/kappa; reported F1=88.3 not entered as macro. · [source](https://arxiv.org/pdf/1911.03221) |
| P19 · 19 | PFTSleep | Held-out test | Multichannel full-night PSG | — | — | 0.81 | Summary-reported kappa; do not equate with EDF test. · [source](https://pubmed.ncbi.nlm.nih.gov/40080690/) |
| P19 · 20 | PFTSleep zero-shot | APPLES external | Multichannel full-night PSG | — | — | 0.59 | External-cohort kappa. · [source](https://pubmed.ncbi.nlm.nih.gov/40080690/) |
| P19 · 21 | PFTSleep zero-shot | MESA external | Multichannel full-night PSG | — | — | 0.6 | Before dataset-specific fine-tuning. · [source](https://pubmed.ncbi.nlm.nih.gov/40080690/) |
| P19 · 22 | PFTSleep zero-shot | MrOS visit 2 external | Multichannel full-night PSG | — | — | 0.75 | External-cohort kappa. · [source](https://pubmed.ncbi.nlm.nih.gov/40080690/) |
| P19 · 23 | PFTSleep fine-tuned | MESA external / adapted head | Multichannel full-night PSG | — | — | 0.76 | Fine-tuning result is not zero-shot. · [source](https://pubmed.ncbi.nlm.nih.gov/40080690/) |
| P20 · 24 | SOMNUS / SLEEPYLAND | SEDF-SC OOD | EEG/EOG ensemble + channel vote | 92 | 75.4 | 0.835 | Pooled test recordings; Supplementary Table 3 arXiv v2. · [source](https://arxiv.org/html/2506.08574v2) |
| P20 · 25 | SOMNUS / SLEEPYLAND | SEDF-ST OOD | EEG/EOG ensemble + channel vote | 83.8 | 77.5 | 0.767 | Pooled test recordings; Supplementary Table 3 arXiv v2. · [source](https://arxiv.org/html/2506.08574v2) |
| P20 · 26 | SOMNUS / SLEEPYLAND | DOD-H OOD | EEG/EOG ensemble + channel vote | 90.3 | 85.7 | 0.861 | Pooled test recordings; Supplementary Table 3 arXiv v2. · [source](https://arxiv.org/html/2506.08574v2) |
| P20 · 27 | SOMNUS / SLEEPYLAND | DOD-O OOD | EEG/EOG ensemble + channel vote | 87.8 | 81.6 | 0.821 | Pooled test recordings; Supplementary Table 3 arXiv v2. · [source](https://arxiv.org/html/2506.08574v2) |
| P24 · 28 | Independent validation of YASA | 75 adults / 483 nights | EEG + EOG + EMG | 82.9 | — | — | Overall agreement as reported; lights-out/lights-on; not Figure 1 median. · [source](https://link.springer.com/article/10.1007/s44470-026-00059-x) |
| P30 · 29 | TransSleep — 2022 preprint | MASS | EEG | 87.4 | 82.6 | — | Preprint Table 2, not necessarily final-publication score. · [source](https://arxiv.org/pdf/2203.12590) |
