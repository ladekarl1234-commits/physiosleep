# Claim-to-evidence ledger

| Claim | Evidence | Scope / disallowed inference |
|---|---|---|
| Best development Macro-F1 0.789779 | `evidence/development-metrics.json`; replay script; aggregate figures | Adaptive D60 selection; not audit performance |
| +0.003512 vs fixed control | Matched D119 confusion results in report | Below +0.02; not every mandatory comparator |
| A/B exploratory tests completed: Macro-F1 0.759022 / 0.798875 | [Saved-results report](../reports/exploratory-audits-v1/report.md), aggregate confusion counts and evaluator source | Single D60 control; formal gates NOT_RUN; consumed cohorts cannot be reused for confirmation |
| Complete development epoch accounting | 276,133 complete / 274,271 valid / 1,862 invalid | Invalid labels never silently compacted |
| Publisher-linked source bytes for all evaluated pairs | [Fresh complete-file rehash summary](../evidence/publisher-content-crosswalk-20260925.json), access-controlled source and receipt hashes | 398/398 manifest files and 394/394 EDFs matched; content identity does not establish timing or scoring correctness |
| Independent annotation/grid crosscheck for all 197 evaluated pairs | [Aggregate crosscheck and private receipt hashes](../evidence/reader-annotation-crosscheck-20260925.json) | 28,529 EDF+ intervals; full projected labels, validity and onset positions matched; PSG waveform calibration was not crosschecked here |
| Fixed EEG+EOG control score errors 6.37 / 24.71 / 73.31 | [Named systems in score table](RESULTS.md#experimental-score) | Historical control, not latest candidate; eligible 61 windows / 40 people; targets unmet |
| Frozen transition candidate score errors 5.68 / 24.33 / 63.09 | [Score extension](../reports/score-extension/summary.json), [independent verification](../evidence/score-extension-verification.json) | Same 61/40 and charter; all three MAEs fail; paired intervals versus EEG+EOG include zero |
| Sleepyland/YASA five-fold development fit and parity | [Aggregate execution and verification receipt](../evidence/sleepyland-yasa-development.json) | Macro-F1 0.783662; two feature groups, three physiological channels; no unchanged Docker-equivalence or mandatory-slot completion |
| Hardware calculations and generated fault checks | [Public six-case replay and historical result hashes](../evidence/hardware-desk-replay.json), [source and command](DEVELOPER_GUIDE.md#replay-the-generated-hardware-cases) | Only six packet/clock integration cases replay publicly; historical filter/clock/packet group counts remain hash-bound; no physical bench or safety/clinical validation |
| Agent judge grades | Frozen rubrics and initial/final reviews | Simulated perspectives, not clinical or competition approval |
| Literature scores | [56 workbook experiment rows](../literature/COMPARISON.md), [31 papers](../literature/PAPERS.md), source identities and URLs | Attributed transcriptions, not independently verified numeric paper rows; different protocols; never head-to-head |
| Native training and software status | [Dated jobs](EXPERIMENTS.md), [CI receipt](../evidence/hosted-ci.json) | Static observations; running epochs, software tests and confirmation are distinct |

Artifact identities are listed in `evidence/local-evidence-identities.json`. Full protected run payloads remain local; public readers can reproduce aggregate arithmetic, not independently inspect those raw predictions from this repository alone.
