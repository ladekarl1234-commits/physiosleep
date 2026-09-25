# Catalog of the 31 supplied papers

[Numeric comparisons](COMPARISON.md) · [CSV with original relevance and limitation notes](papers.csv)

Titles, versions, modalities and dataset descriptions below are transcribed from the owner workbook. A listed paper is not a reproduced baseline or a verified numerical source. The nine catalog-only papers have no experiment row in the 56-row extract.

| ID / workbook row | Paper / model | Version | Inputs | Dataset / evaluation |
|---|---|---|---|---|
| P01 / 5 | [YASA — An open-source, high-performance tool for automated sleep staging](https://elifesciences.org/articles/70092) | 2021 | EEG + EOG + EMG; age/sex; EEG+EOG ablation | NSRR test 1: 585 nights; DOD external test |
| P02 / 6 | [DeepSleepNet](https://arxiv.org/abs/1703.04046) | 2017 | Single EEG | Sleep-EDF-20; MASS |
| P03 / 7 | [SeqSleepNet](https://arxiv.org/abs/1809.10932) | 2019 | EEG + EOG + EMG; alternative configurations | MASS; subsequent EDF evaluations |
| P04 / 8 | [SleepEEGNet](https://arxiv.org/html/1903.02108v1) | 2019 | Single EEG | Sleep-EDF 2013 and 2018 |
| P05 / 9 | [IITNet](https://arxiv.org/abs/1902.06562) | 2020; preprint 2019 | Single EEG | Sleep-EDF; MASS; SHHS |
| P06 / 10 | [U-Time](https://arxiv.org/html/1910.11162v1) | 2019 | Single EEG; multimodal ablations | EDF-39; EDF-153; five additional cohorts |
| P07 / 11 | [TinySleepNet](https://github.com/akaraspt/tinysleepnet) | 2020 | Single EEG | Sleep-EDF 2013 and 2018; other cohorts |
| P08 / 12 | [AttnSleep — An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf) | 2021 | Single EEG | EDF-20; EDF-78; selected SHHS |
| P09 / 13 | [XSleepNet: Multi-View Sequential Model for Automatic Sleep Staging](https://arxiv.org/pdf/2007.05492) | 2021 online / 2022 volume | EEG; EEG+EOG; EEG+EOG+EMG | EDF; MASS; Physio2018; SHHS |
| P10 / 14 | [SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification](https://arxiv.org/pdf/2105.11043) | 2022 | Single EEG | SHHS; expanded EDF |
| P11 / 15 | [L-SeqSleepNet: Whole-cycle Long Sequence Modelling for Automatic Sleep Staging](https://ar5iv.labs.arxiv.org/html/2301.03441) | 2023 | Single EEG | SHHS; EDF-20 |
| P12 / 16 | [SleePyCo: Automatic sleep scoring with feature pyramid and contrastive learning](https://arxiv.org/pdf/2209.09452) | 2024; preprint 2022 | Single EEG | Expanded EDF; MASS; Physio2018; SHHS |
| P13 / 17 | [Do not sleep on traditional machine learning: Simple and interpretable techniques are competitive to deep learning for sleep scoring](https://arxiv.org/pdf/2207.07753) | 2023 | EEG / EOG / EMG combinations | SC20; SC78; ST; MASS |
| P14 / 18 | [ZleepAnlystNet](https://www.nature.com/articles/s41598-024-60796-y) | 2024 | Single EEG | Sleep-EDF 2013 and 2018 |
| P15 / 19 | [DeepSleepNet-Lite](https://arxiv.org/pdf/2108.10600) | 2021 preprint / 2022 publication | Single EEG | Sleep-EDF 2013 and 2018 |
| P16 / 20 | [Dreem Open Datasets: Multi-Scored Sleep Datasets to Compare Human and Automated Sleep Staging — SimpleSleepNet](https://arxiv.org/pdf/1911.03221) | 2020 | Multichannel PSG; single-channel ablation | DOD-H; DOD-O; multiple human scorers |
| P17 / 21 | [RobustSleepNet: Transfer learning for automated sleep staging at scale](https://arxiv.org/abs/2101.02452) | 2021 | Variable EEG/EOG montage | Eight datasets; leave-one-dataset-out |
| P18 / 22 | [U-Sleep: resilient high-frequency sleep staging](https://www.nature.com/articles/s41746-021-00440-5) | 2021 | EEG + EOG; channel-pair aggregation | Large multi-cohort training; unseen cohorts |
| P19 / 23 | [PFTSleep — A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages](https://pubmed.ncbi.nlm.nih.gov/40080690/) | 2025 | Full-night multichannel PSG | Held-out; APPLES; MESA; MrOS |
| P20 / 24 | [SLEEPYLAND: trust begins with fair evaluation of automatic sleep staging models — SOMNUS](https://arxiv.org/html/2506.08574v2) | 2025 online; arXiv v2 | EEG/EOG ensembles; channel-derivation voting | 17 in-domain cohorts; OOD including SEDF and DOD |
| P21 / 25 | [SleepFM — A multimodal sleep foundation model for disease prediction](https://www.nature.com/articles/s41591-025-04133-4) | 2026 | Multimodal PSG | Multiple cohorts; external validation |
| P22 / 26 | [SleepDG: Generalizable Sleep Staging via Multi-Level Domain Alignment](https://arxiv.org/html/2401.05363v3) | 2024 | Multimodal sleep signals | Five datasets; domain generalization |
| P23 / 27 | [SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://www.ijcai.org/proceedings/2021/0360.pdf) | 2021 | EEG + EOG | EDF-39; EDF-153 |
| P24 / 28 | [YASA automated sleep staging performance across seven nights of normal sleep and sleep restriction](https://link.springer.com/article/10.1007/s44470-026-00059-x) | 2026 | EEG + EOG + EMG | 75 adults; 483 nights; independent laboratory |
| P25 / 29 | [Improving Automatic Sleep Staging Via Temporal Smoothness Regularization](https://api.unil.ch/iris/server/api/core/bitstreams/4c6f4de9-d179-4888-8311-a6ce23042f7d/content) | 2023 | Single EEG | EDF-20; subject-wise CV |
| P26 / 30 | [A convolutional neural network for sleep stage scoring from raw single-channel EEG — Sors et al.](https://www.sciencedirect.com/science/article/pii/S1746809417302847) | 2018 | Single EEG | SHHS |
| P27 / 31 | [A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series — Chambon et al.](https://arxiv.org/abs/1707.03321) | 2018; preprint 2017 | EEG + EOG + EMG | Multimodal PSG |
| P28 / 32 | [Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy — Stephansen et al.](https://www.nature.com/articles/s41467-018-07229-3) | 2018 | Multimodal PSG | Multiple cohorts |
| P29 / 33 | [GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification](https://www.ijcai.org/proceedings/2020/184) | 2020 | Multiple EEG channels | MASS-SS3 |
| P30 / 34 | [TransSleep: Transitioning-Aware Attention-Based Deep Neural Network for Sleep Staging](https://arxiv.org/pdf/2203.12590) | 2023 publication; numerical source: 2022 preprint | Single EEG | EDF; MASS |
| P31 / 35 | [AISleep: Automated and interpretable sleep staging](https://pubmed.ncbi.nlm.nih.gov/41472827/) | 2025 | Single EEG | Healthy public cohorts; clinical cohorts |
