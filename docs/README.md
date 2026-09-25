# Documentation

[Repository home](../README.md) · [Judge guide](JUDGE_GUIDE.md)

| Area | Main guide | Supporting evidence |
|---|---|---|
| Read the final measured conclusion | [Final English report](FINAL_REPORT.md) | A/B values, Macro-F1 formula, limits and judge questions |
| Follow each evaluation stage | [Dev/train → Audit 1 → Audit 2](EVALUATION_STAGES.md) | Three-stage figure, model identity, denominators and uncertainty |
| Understand the project | [Pipeline](PIPELINE.md) | Source-module map and leakage boundaries |
| Review what ran | [Experiment and test ledger](EXPERIMENTS.md) | [Public job snapshot](../evidence/job-status.json), hosted CI |
| Assess local performance | [Results](RESULTS.md) | Confusion counts, comparison figures, score intervals |
| Inspect the A/B tests | [Exploratory audit guide](EXPLORATORY_AUDITS.md) | Actual held-person evaluation, exact model identity and confirmation limits |
| Compare to literature | [Protocol-aware comparison](REPOSITORY_COMPARISON.md#published-algorithm-context) | [56 settings](../literature/COMPARISON.md), [31-paper catalog](../literature/PAPERS.md), CSV and source rows |
| Compare upstream code | [Repository comparison](REPOSITORY_COMPARISON.md) | Local difference and stage-recall figures; YASA, SLEEPYLAND, U-Time/U-Sleep and AttnSleep execution status |
| Inspect methodology | [Scientific report](REPORT.md) | [Mandatory baselines](BASELINES.md), [claims](CLAIMS.md) |
| Reproduce public outputs | [Developer guide](DEVELOPER_GUIDE.md) | [Reproducibility](REPRODUCIBILITY.md), source/tests/tools |
| Understand decisions | [ADR index](adr/README.md) | 17 accepted, provisional or superseded decisions |
| Judge the evidence | [Questions and answers](JUDGE_QUESTIONS.md) | [Grades and reviews](JUDGING.md), [fixed-rubric feasibility](ROAD_TO_90.md) |
| Assess hardware | [Hardware guide](HARDWARE.md) | Concepts, calculated budgets and physical prerequisites |
| Check permissions | [Rights](../RIGHTS.md) | [Primary-source recheck](../evidence/rights-recheck-20260925.md) |

Documentation is a dated research snapshot. Software checks, development fits, exploratory test results and formal confirmation are separate evidence states. The original A/B cohorts were retired from confirmation by the owner's 25 September decision. Further baseline testing was ended for this delivery by [ADR 0017](adr/0017-final-exploratory-delivery.md).
