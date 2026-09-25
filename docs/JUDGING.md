# Independent simulated judging and revisions

**[Judge walkthrough](JUDGE_GUIDE.md) · [Repository review and fixes](REVIEW_REVISION.md) · [Current development/test jobs](EXPERIMENTS.md)**

Two separate reviewer agents examined the evidence from sleep-specialist and data/ML perspectives. They are simulated judges, not licensed clinician attestations or official competition judges. Rubrics were fixed before edits and retained for rescoring.

| Perspective | Initial score | Latest reviewed score | Meaning |
|---|---:|---:|---|
| Sleep specialist | 69/100 | 71/100 | Documentation and selected visual review; no added clinical/device/confirmation credit |
| Data / ML / reproducibility | 53/100 | 59/100 | New fit artifact verification, public reproducibility and publication checks |

The [initial specialist rubric](../evidence/judging/sleep-specialist-initial.json), [specialist rescore](../evidence/judging/sleep-specialist-final.json) [initial ML rubric](../evidence/judging/ml-initial.json) and [final ML rubric](../evidence/judging/ml-final.json) preserve individual criteria. Scores grade the available evidence, not a claim that this project meets its benchmark.

## Improvements made after review

- Put the unmet +0.02 margin, incomplete eleven-way benchmark and untouched audits beside headline results.
- Expose the score-eligible 61/119 recordings and 40/60 people, failed component targets, bias/P90, descriptive uncertainty and 18-configuration sensitivity.
- Define SPT, WASO, all-Wake behavior and missing independent night boundaries; distinguish R&K mapping from AASM rescoring.
- Separate literature medians/pooled results and secondary sources from actual matched local experiments.
- Publish anonymous aggregate figure data, exact confusion-count replay, pinned environments, synthetic tests and source identities.
- Repair the public test dependency/fixture omissions and rerun all 92 tests in a fresh local environment.
- Correct the two-view versus three-seed figure mapping and assert exact ensemble membership.
- Separate concept art, generated engineering checks and physical device evidence; document unresolved EOG geometry and Cz access.

The next substantial score gains require new evidence: adequate mandatory baselines, unchanged confirmation gates, actual external full-training replay, independent sleep-boundary/clinical validity and paired physical-device qualification. These cannot be earned by changing the rubric, hiding failures or polishing pictures.

## Repository usability review, 25 September 2026

The owner requested clearer documentation, visible jobs/grades, a pipeline, a judging route and numeric Excel comparisons. A specialist methodologist and data/ML verifier independently identified the gaps, then reviewed the revised package. The [finding-to-fix record](REVIEW_REVISION.md) documents their feedback and the changes. This is a separate review of discoverability and evidence traceability, with no automatic numerical scientific regrade.

- [Data/ML second review](../evidence/judging/repository-ml-review.md): all seven initial navigation/traceability findings resolved; original workbook and exported values checked.
- [Data/ML machine-readable receipt](../evidence/judging/repository-ml-review.json): counts, missingness, scope and release conditions at review time.
- [Specialist review record](../evidence/judging/repository-specialist-review.md): U01–U07 accepted; final nonblocking layout note corrected by the editor.
