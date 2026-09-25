# Claim-to-evidence ledger

| Claim | Evidence | Scope / disallowed inference |
|---|---|---|
| Best development Macro-F1 0.789779 | `evidence/development-metrics.json`; replay script; aggregate figures | Adaptive D60 selection; not audit performance |
| +0.003512 vs fixed control | Matched D119 confusion results in report | Below +0.02; not every mandatory comparator |
| A/B not run | Current confirmation decision and frozen registry | NOT_RUN is not pass or fail |
| Complete development epoch accounting | 276,133 complete / 274,271 valid / 1,862 invalid | Invalid labels never silently compacted |
| Fixed EEG+EOG control score errors 6.37 / 24.71 / 73.31 | [Named systems in score table](RESULTS.md#experimental-score) | Historical control, not latest candidate; eligible 61 windows / 40 people; targets unmet |
| Frozen transition candidate score errors 5.68 / 24.33 / 63.09 | [Score extension](../reports/score-extension/summary.json), [independent verification](../evidence/score-extension-verification.json) | Same 61/40 and charter; all three MAEs fail; paired intervals versus EEG+EOG include zero |
| Sleepyland/YASA five-fold development fit and parity | [Aggregate execution and verification receipt](../evidence/sleepyland-yasa-development.json) | Macro-F1 0.783662; two feature groups, three physiological channels; no unchanged Docker-equivalence or mandatory-slot completion |
| Hardware calculations and generated fault checks | Hardware contracts and source tools | No physical bench or safety/clinical validation |
| Agent judge grades | Frozen rubrics and initial/final reviews | Simulated perspectives, not clinical or competition approval |
| Literature scores | [56 workbook experiment rows](../literature/COMPARISON.md), [31 papers](../literature/PAPERS.md), source identities and URLs | Attributed transcriptions, not independently verified numeric paper rows; different protocols; never head-to-head |
| Native training and software status | [Dated jobs](EXPERIMENTS.md), [CI receipt](../evidence/hosted-ci.json) | Static observations; running epochs, software tests and confirmation are distinct |

Artifact identities are listed in `evidence/local-evidence-identities.json`. Full protected run payloads remain local; public readers can reproduce aggregate arithmetic, not independently inspect those raw predictions from this repository alone.
