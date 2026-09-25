# Independent simulated judging and revisions

**[Judge walkthrough](JUDGE_GUIDE.md) Â· [Repository review and fixes](REVIEW_REVISION.md) Â· [Current development/test jobs](EXPERIMENTS.md)**

Two separate reviewer agents examined the evidence from sleep-specialist and data/ML perspectives. They are simulated judges, not licensed clinician attestations or official competition judges. Rubrics were fixed before edits and retained for rescoring.

The owner has renewed the request for a score above 90 while keeping mandatory-baseline testing stopped. The [criterion-by-criterion feasibility check](ROAD_TO_90.md) shows optimistic ceilings of 77/100 for data/ML and 83/100 for the sleep specialist under those evidence constraints. The specialist gained one point for test-linked sensitivity cases and one for publicly replayable generated hardware evidence; data/ML gained points for a complete publisher-byte crosswalk, an independent all-pair reader/grid crosscheck and a source-bound development-candidate lineage review. These do not supply comparator, physical-device or confirmation evidence.

| Perspective | Initial score | Latest reviewed score | Meaning |
|---|---:|---:|---|
| Sleep specialist | 69/100 | **75/100** | Held-person evidence, scoped uncertainty, reviewed sensitivity and generated hardware replay; no clinical/device/confirmation credit |
| Data / ML / reproducibility | 53/100 | **61/100** | Source-byte, 197-pair reader/grid and development-candidate lineage checks reviewed; consumed holdouts remain |

The [initial specialist rubric](../evidence/judging/sleep-specialist-initial.json), [exploratory specialist review](../evidence/judging/sleep-specialist-exploratory.json), [sensitivity regrade](../evidence/judging/sleep-specialist-sensitivity-regrade.json), [hardware-replay regrade](../evidence/judging/sleep-specialist-hardware-replay-regrade.json), [initial ML rubric](../evidence/judging/ml-initial.json), [exploratory ML review](../evidence/judging/ml-exploratory.json), [publisher-provenance regrade](../evidence/judging/ml-publisher-provenance-regrade.json), [reader/grid regrade](../evidence/judging/ml-reader-annotation-regrade.json) and [candidate-lineage regrade](../evidence/judging/ml-candidate-lineage-regrade.json) preserve the criteria and dated changes. Scores grade the available evidence, not a claim that this project meets its benchmark.

## Regrade after completed exploratory A/B tests

The [specialist regrade](../evidence/judging/sleep-specialist-exploratory.json) is **71 → 73** under the unchanged 19-criterion rubric. The [ML regrade](../evidence/judging/ml-exploratory.json) is **59 → 58** under its unchanged 15-criterion rubric. The specialist added two points for additional participant-disjoint evidence and explicit clustered uncertainty. The ML reviewer removed the remaining holdout-preservation point because both reserved cohorts were consumed. Formal confirmation still earns zero ML points. No score was inflated to meet a target.

Both A/B results passed [independent local saved-array verification](../evidence/exploratory-audits-verification.json). This is not an external full-training replay. Older review records below are retained as historical evidence.

## Regrade after score sensitivity review

The [specialist review receipt](../evidence/judging/sleep-specialist-sensitivity-regrade.json) records **73 to 74** under the same rubric, changing only `sensitivity_failure_cases` from 4/5 to 5/5. The reviewer independently checked the boundary and invariance examples against the public synthetic tests, source and timing contract. This adds interpretability credit, not clinical, score-accuracy, comparator, confirmation or device credit. The data/ML reviewer rechecked the documentation diff and retained **58/100** because no ML criterion gained new experimental evidence.

## Regrade after complete source-byte crosswalk

The [data/ML review receipt](../evidence/judging/ml-publisher-provenance-regrade.json) records **58 to 59** under the same rubric, changing only `data_content_provenance` from 6/7 to 7/7. A separate data engineer freshly hashed all 398 manifest files (8,715,189,781 bytes), matched the publisher manifest and cross-checked all 394 evaluated EDFs against the protected readiness record. The reviewer inspected the public summary, private execution receipt and source, but did not independently repeat the full byte scan or HTTPS fetch. The [public crosswalk](../evidence/publisher-content-crosswalk-20260925.json) contains counts and hashes only. This adds source-identity credit, not model, annotation, comparator, confirmation or clinical credit.

## Regrade after generated hardware replay

The [specialist review receipt](../evidence/judging/sleep-specialist-hardware-replay-regrade.json) records **74 to 75**, changing only `accessible_traceable_provenance` from 4/5 to 5/5. A separate read-only specialist reviewer matched six private historical result/review hashes and all six public case summaries, reran the generated packet/clock checks, and confirmed that an altered valid-sample count is rejected. [Inputs, command and limits](HARDWARE.md) are public. The reviewer assigned no acquisition, real-device, staging, score-fidelity or clinical credit.

## Regrade after all-pair reader and grid crosscheck

The [data/ML review receipt](../evidence/judging/ml-reader-annotation-regrade.json) records **59 to 60**, changing only `data_reader_annotation_timing` from 6/7 to 7/7. A bounded independent TAL decoder and boundary-partition projector checked all 197 original Hypnograms, including seven MNE compatibility cases, against established readers and every frozen truth grid. A separate verifier reran the command, reproduced the exact private receipt hash and rejected a malformed synthetic TAL. The [public aggregate](../evidence/reader-annotation-crosscheck-20260925.json) contains only counts and hashes. This adds reader/epoch-accounting credit, not model, comparator, whole-night, clinical or confirmation credit.

## Regrade after development candidate lineage review

The [data/ML review receipt](../evidence/judging/ml-candidate-lineage-regrade.json) records **60 to 61**, changing only `ml_candidate_development` from 7/8 to 8/8. A private read-only [crosswalk aggregate](../evidence/model-lineage-crosswalk-20260925.json) ties existing source, split, derived, 15-fold, three-seed, transition and separate D60 audit-refit artifacts to prior independent metric receipts. The methodologist separately checked the completed six-configuration screen's result, config and plan hashes. The crosswalk's seven injected mismatches were rejected on independent rerun. Development selection used all 60 participants; the best development ensemble was not the model evaluated on A/B. Complete immutable artifact ancestry stays **6/7** because some transition and audit bytes lack one independently anchored freeze map. No model was retrained or rescored.

## Improvements made after the earlier review

- Put the unmet +0.02 margin, incomplete eleven-way benchmark and the then-untouched audits beside headline results; current exploratory results are now separately reported.
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

- [Historical data/ML repository review](../evidence/judging/repository-ml-review.md): all seven initial navigation/traceability findings resolved at its **pre-audit cutoff**. Its 59/100 ML, 71/100 specialist and Audit A/B `NOT_RUN` statements describe that earlier date, not the current 61/75 grades above.
- [Data/ML machine-readable receipt](../evidence/judging/repository-ml-review.json): counts, missingness, scope and release conditions at review time.
- [Specialist review record](../evidence/judging/repository-specialist-review.md): U01â€“U07 accepted; final nonblocking layout note corrected by the editor.
