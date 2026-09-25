# PhysioSleep v0.1.0 research snapshot

**25 September 2026 · research prerelease**

This snapshot packages an inspectable Sleep-EDF sleep-staging research workflow, public aggregate evidence and a clear three-stage reviewer route. It is source and documentation, not a trained-model binary or clinical device.

## Measured results

| Stage | Pooled five-class Macro-F1 | Scope |
|---|---:|---|
| Development best pipeline | 0.789779 | 60 people / 119 recordings, five held-person folds; three-seed transition pipeline |
| Audit 1 (A) | 0.759022 | 20 people / 39 recordings; frozen D60 seed-17 single model |
| Audit 2 (B) | 0.798875 | 20 different people / 39 recordings; **same** single model |

Audit results are exploratory. Development model selection and the audit checkpoint differ; the selected development ensemble was not audit-tested. The original A/B cohorts are consumed, and formal +0.02 comparator gates remain **NOT_RUN**. No clinical, device or cross-dataset validation is claimed.

## Included for review

- [Three-stage figure and metric walkthrough](https://github.com/ladekarl1234-commits/physiosleep/blob/v0.1.0/docs/EVALUATION_STAGES.md), including the mapping from dev/train to Audit 1 and Audit 2.
- [Final measured report](https://github.com/ladekarl1234-commits/physiosleep/blob/v0.1.0/docs/FINAL_REPORT.md), class F1, confusion matrices, uncertainty and limitations.
- [Upstream repository comparison](https://github.com/ladekarl1234-commits/physiosleep/blob/v0.1.0/docs/REPOSITORY_COMPARISON.md) distinguishing local measurements from available algorithms and unmatched paper scores.
- [Public verification commands](https://github.com/ladekarl1234-commits/physiosleep/blob/v0.1.0/docs/DEVELOPER_GUIDE.md) for aggregate arithmetic, 99 synthetic tests and publication boundaries.

Original EDFs, participant-linked predictions, private checkpoints and third-party weights are excluded. Public checks can replay aggregate scores and inspect code, but cannot rerun the private model fit or bootstrap inputs from this clone alone. See [rights and exclusions](https://github.com/ladekarl1234-commits/physiosleep/blob/v0.1.0/RIGHTS.md).
