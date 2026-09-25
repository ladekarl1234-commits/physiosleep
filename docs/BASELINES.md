# Required comparator registry

**Looking for what actually ran? Open the [dated development / test job ledger](EXPERIMENTS.md) and [local results](RESULTS.md).** The frozen table below records required confirmation slots, separately from development work.

The frozen registry is a preregistration snapshot, not a live training dashboard. Its `NOT_RUN` slots indicate no completed confirmatory execution. Native YASA development work and bounded Sleepyland/YASA compatibility evidence do not complete the eleven-way benchmark. Exact JSON pins and the full variant set are in [the frozen registry](../evidence/baseline-registry-frozen.json).

| Mandatory slot | Upstream | Revision | Frozen slot status |
|---|---|---|---|
| `yasa_native` | [raphaelvallat/yasa](https://github.com/raphaelvallat/yasa/tree/e581bf452097e01b493b3072964e206ae9b01dc4) | `e581bf45` | NOT_RUN |
| `msa_cnn_official` | [sgoerttler/MSA-CNN](https://github.com/sgoerttler/MSA-CNN/tree/f01455eb0396f790571b165fd0350b105476f043) | `f01455eb` | NOT_RUN |
| `attnsleep_official` | [emadeldeen24/AttnSleep](https://github.com/emadeldeen24/AttnSleep/tree/6b4d2665884628c8a7bb09f36589a8ec0992f8e2) | `6b4d2665` | NOT_RUN |
| `xsleepnet_official` | [pquochuy/XSleepNet](https://github.com/pquochuy/XSleepNet/tree/7a2248e78cf8f12ae6d2ae5041bb30ac3133bf29) | `7a2248e7` | NOT_RUN |
| `tinysleepnet_official` | [akaraspt/tinysleepnet](https://github.com/akaraspt/tinysleepnet/tree/70f45cff92ac0bf0a718e522bc27cca2c63ceff7) | `70f45cff` | NOT_RUN |
| `utime_official` | [perslev/U-Time](https://github.com/perslev/U-Time/tree/7fc4cbf79e5454661f1c0d0768886e4dfbe9d42a) | `7fc4cbf7` | NOT_RUN |
| `usleep_official` | [perslev/U-Time](https://github.com/perslev/U-Time/tree/7fc4cbf79e5454661f1c0d0768886e4dfbe9d42a) | `7fc4cbf7` | NOT_RUN |
| `sleepyland_yasa` | [biomedical-signal-processing/sleepyland](https://github.com/biomedical-signal-processing/sleepyland/tree/db420a5c1d3329304907d14f9b3b314fa6605aa9) | `db420a5c` | NOT_RUN |
| `sleepyland_usleep` | [biomedical-signal-processing/sleepyland](https://github.com/biomedical-signal-processing/sleepyland/tree/db420a5c1d3329304907d14f9b3b314fa6605aa9) | `db420a5c` | NOT_RUN |
| `sleepyland_deepresnet` | [biomedical-signal-processing/sleepyland](https://github.com/biomedical-signal-processing/sleepyland/tree/db420a5c1d3329304907d14f9b3b314fa6605aa9) | `db420a5c` | NOT_RUN |
| `sleepyland_sleeptransformer` | [biomedical-signal-processing/sleepyland](https://github.com/biomedical-signal-processing/sleepyland/tree/db420a5c1d3329304907d14f9b3b314fa6605aa9) | `db420a5c` | NOT_RUN |

## Adequacy and remaining work

- YASA: local clean EEG and EEG+EOG classifiers have complete development fits. Convergence at 400 rounds is not established; released model lineage is separate.
- MSA-CNN: serious 100-epoch recipes, all physiological variants, participant split and named-channel mapping required; license grant unresolved.
- AttnSleep: native 100-epoch route with pinned legacy environment. Real seed-17 training is active at the [dated snapshot](EXPERIMENTS.md); individual epochs/probes do not equal completed serious folds.
- XSleepNet: native one/two-channel routes and MATLAB preprocessing parity required; code-use rights unresolved.
- TinySleepNet: native sequence training and TensorFlow 1.13.1 route; conflicting terms unresolved.
- U-Time / U-Sleep: distinct native architectures/configurations; 128 Hz preprocessing, sampler/loss equivalence and adequate clean fits required. U-Time's fold-0 inner invocation **failed at state commit after 433 updates and validation**. Synthetic restart qualification passed; actual checkpoint recovery and adequate fits remain incomplete. [Dated status and evidence](EXPERIMENTS.md).
- Sleepyland YASA: the repaired string-class facade preserves original saved model bytes and native probabilities in an actual saved-fold test. A clean five-fold development fit is independently verified at 0.783662 Macro-F1 on 119 recordings. All five-fold parity checks passed. Full mandatory-slot adequacy and unchanged-container equivalence must not be assumed.
- Sleepyland deep families: register all 18 family/year/modality variants plus the equal-probability 2024 EEG+EOG ensemble. Released-weight rights/ancestry and Linux/container availability remain unresolved. Clean native training is only an alternative where permitted and demonstrated.

| Blocker | What resolves it |
|---|---|
| Code or weight use rights | Applicable published terms or actual authorization for the intended use |
| Unknown training overlap | Participant-level exclusion evidence or permitted clean local retraining |
| Native runtime parity | Tested supported runtime or numerical/output equivalence for the declared compatibility route |
| Inadequate development margin / precision | Justified development improvements and readiness evidence; do not expose audits to diagnose it |
| Failed A or insufficient independent precision | Fresh independent confirmation people and an explicit sequential protocol; B is not a retry pool |

No literature score, missing comparator, false duplicate or optimistic status flag can waive these requirements.
