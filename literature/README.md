# Literature context, not a matched leaderboard

The owner supplied `sleep_staging_literature_comparison.xlsx` (SHA-256 `b4ed35cd01d6ff839c1ebe495a99fc55ceae1900af67a2ed87824d8a133909c4`), with 31 paper entries, 56 experiment entries and explanatory protocol notes. The workbook was read without modification. Its contents are research inputs, not instructions or independently verified experimental evidence.

The requested four-panel design resembles Figure 1 of [Vallat and Walker's YASA article](https://elifesciences.org/articles/70092). That article reports a **87.46% median nightly accuracy** on 585 NSRR validation nights, plus separate DOD consensus validation. It is not the pooled accuracy or Macro-F1 of our D119. Its stage heatmap describes recall. Our figures use our own saved predictions and explicitly state their aggregation.

| Family / source | Role in this project | Comparability limitation |
|---|---|---|
| [YASA](https://elifesciences.org/articles/70092) | Physiological feature reference and mandatory local route | NSRR median-night results differ from local full-record pooled EDF evaluation |
| [DeepSleepNet](https://arxiv.org/abs/1703.04046) | Literature context | Cohort selection, cropping and training differ |
| [U-Time](https://arxiv.org/abs/1910.11162) / [U-Sleep](https://www.nature.com/articles/s41746-021-00440-5) | Mandatory native families | Local clean training is not their original multi-dataset program |
| [TinySleepNet](https://github.com/akaraspt/tinysleepnet) | Mandatory family | Workbook EDF numeric rows cite another paper's comparison tables; retain secondary-source status |
| [AttnSleep](https://github.com/emadeldeen24/AttnSleep) | Mandatory family | Must execute serious matched local recipe before ranking |
| [XSleepNet](https://arxiv.org/abs/2007.05492) | Mandatory family | Native transforms, split, terms and weight ancestry require verification |
| [Sleepyland](https://arxiv.org/abs/2506.08574) | Packaged inference families / variants | Pretraining data, released rights and full native runtime differ |

The workbook extraction was globally marked `primary_sources_independently_checked=false`. Only the specifically described YASA aggregation above was checked against its primary article in this publication pass. Opening other abstracts does not verify all numeric tables. Unverified numeric literature rows are not plotted as if they were local results. In particular, subtracting our development score from a paper's cropped/repeated-CV/externally pretrained result cannot satisfy Gate A.
