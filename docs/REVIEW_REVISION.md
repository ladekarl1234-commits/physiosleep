# Repository review and changes

[Judge guide](JUDGE_GUIDE.md) Ã‚Â· [Scientific grades](JUDGING.md)

On 25 September 2026 the owner asked for a repository that clearly exposes development/test work, grades, judging guidance, the pipeline and comparisons from the supplied Excel. Two independent simulated reviewer agents examined the existing package before the changes: a sleep-specialist methodologist and a data/ML verifier. The main implementation agent made the revisions; the reviewers did not implement their own findings.

## Findings and responses

| Reviewer finding | Concrete change | Where a judge can verify it |
|---|---|---|
| Project purpose was buried beneath metrics | Explain EEG/EOG input, five-stage output, offline use and research maturity first | [Repository home](../README.md) |
| Frozen NOT_RUN table obscured executed work | Separate completed development fits, running native jobs, failures, software tests and audits in a dated ledger | [Jobs](EXPERIMENTS.md), [snapshot](../evidence/job-status.json) |
| Grades were difficult to find or interpret | Put both grades on the landing page, with review dates and fixed-rubric links | [Judge guide](JUDGE_GUIDE.md), [grading records](JUDGING.md) |
| No end-to-end pipeline/source map | Add PNG/SVG diagram, reference-label boundary graph and source-module table | [Pipeline](PIPELINE.md) |
| Excel content was acknowledged but not exposed | Publish all 56 experiment rows and 31 papers with source rows/URLs, CSV and JSON; plot literature separately | [Comparisons](../literature/COMPARISON.md) |
| Judge navigation was a flat list | Add five-minute route, deeper technical route, linked questions, developer setup and documentation index | [Start here](JUDGE_GUIDE.md), [developer guide](DEVELOPER_GUIDE.md) |
| Score versions and parity claims had drifted | Name historical control vs transition candidate; retain 119/60 vs 61/40; update five-fold Sleepyland evidence | [Results](RESULTS.md), [claims](CLAIMS.md) |
| Workflow configuration was presented without an execution receipt | Link an observed hosted run with exact commit, completion and test count | [Hosted CI record](../evidence/hosted-ci.json) |

## Review checks

The specialist re-review accepted all seven original usability findings substantively. It checked 752 exported literature fields against the immutable extraction, preserved missing values, reviewed three new PNGs and verified figure-manifest hashes. It requested two wording corrections in the pipeline: distinguish a mean of seed probabilities from uniform probabilities, and distinguish summary agreement evaluation from inference. Both were applied. Its final layout note was fixed by shortening the summary box title.

The data/ML verification record and final closure status are linked from [judging](JUDGING.md). Public checks validate exact aggregates, tracked-file boundaries, local links and bound source/figure identities. Browser inspection verifies the actual published navigation and rendering; none of those checks opens the audit cohorts or retrains a baseline.

## What this revision does not change

At the earlier documentation-review cutoff, scientific grades were **sleep specialist 71/100 and data/ML 59/100**. No numerical presentation grade is substituted for them. At that review cutoff formal Audit A/B gates remained NOT_RUN; [subsequent exploratory tests](EXPLORATORY_AUDITS.md) are now complete. The required +0.02 margin is unmet, score targets are unmet and the device is unvalidated. Better documentation makes the evidence easier to assess; it cannot supply missing experiments or clinical measurements.

The subsequent completed exploratory A/B tests were separately reviewed: specialist 73/100, ML 58/100. See [the current regrade and rationale](JUDGING.md).
