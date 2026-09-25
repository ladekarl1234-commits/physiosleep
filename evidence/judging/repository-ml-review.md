# Final public-repository usability review — data/ML judge

**Disposition: PASS for the seven presentation findings in the initial review, subject to staging and the owner's publication scanner.** This is a review of whether a judge can find and trace evidence. It does not change the frozen scientific grades: simulated data/ML 59/100 and specialist 71/100. Audit A/B remain NOT_RUN; development performance is not benchmark confirmation.

## Findings first

No blocking discrepancy was found in the revised public literature, local comparison, score table or guide links. The remaining release dependency at review time is packaging: `git status --short` in `publication/physiosleep` still showed unstaged modifications (including `README.md`, `docs/JUDGE_QUESTIONS.md`, `docs/PIPELINE.md`, `evidence/judging/ml-final.json`, `reports/judge-guide/manifest.json`, the pipeline images and `tools/build_judge_figures.py`). The owner must stage final bytes and run the publication scanner; this review is not a push or hosted check of those bytes. The hosted CI receipt records a successful run for commit `e824bc84dba4aed3cc552d9ca1e31e0dd043ca47`, not the current revised tree.

## Seven initial findings rechecked

| Initial finding | Final judgment | Evidence |
|---|---|---|
| 1. Visible job ledger | Resolved for reviewer navigation | `docs/EXPERIMENTS.md` distinguishes completed D fits, AttnSleep `RUNNING_AT_SNAPSHOT` 23/100, U-Time failed after 433 updates, synthetic/public checks and A/B `NOT_RUN`. It links a dated `evidence/job-status.json`; no live status is implied. |
| 2. Claim identities | Resolved | `docs/CLAIMS.md` now names earlier EEG+EOG score errors separately from transition score extension, and five-fold Sleepyland parity. `evidence/local-evidence-identities.json` includes score and Sleepyland verification receipts and explicitly says hashes do not recover protected evidence. |
| 3. Judge route and grades | Resolved | README opens `docs/JUDGE_GUIDE.md`, `docs/JUDGING.md`, results, limitations and rubric; simulated 53→59 and 69→71 are labeled scientific reviews, not official or clinical scores. |
| 4. Pipeline and runnable boundary | Resolved | `docs/PIPELINE.md` maps source modules and invariants from source intake through gates, separates reference annotations from signal-only inference, and states protected training cannot replay from a plain clone. `docs/DEVELOPER_GUIDE.md` gives public check commands. |
| 5. Configured versus observed jobs | Resolved with explicit historical CI limit | `docs/EXPERIMENTS.md` separates software tests, aggregate replay, private D fits and untouched audits. `evidence/hosted-ci.json` records success, run URL and exact old commit; guide states later documentation is not covered by that run. No current hosted execution was independently observed here. |
| 6. Literature workbook | Resolved as an attributed transcription | Public JSON/CSV/Markdown expose 31 papers and all 56 workbook experiment settings, with row identity, protocol, inputs, numeric values, caveat and URL. `literature/COMPARISON.md` labels percentages versus kappa, blanks, secondary sources, mismatched cohorts and no head-to-head rank. Primary numeric paper tables were not independently checked. |
| 7. Consolidated local results | Resolved | `docs/RESULTS.md` gives nine D119/60 staging rows, five D61/40 score systems and three endpoints, N1 limitation, adverse two-view result, fixed control and failed targets, with aggregate/figure links. |

## Independent bounded checks performed

- Loaded the original read-only Excel at `sleep_staging_literature_comparison.xlsx` with bundled Python/openpyxl (`PYTHONUTF8=1`); SHA-256 `b4ed35cd01d6ff839c1ebe495a99fc55ceae1900af67a2ed87824d8a133909c4` equals the public provenance value. Compared every exported paper's eight workbook cells and every experiment's seven core cells plus caveat and URL against its named sheet/row. Counts by sheet: 31 papers, 18 small EDF, 13 expanded EDF, 25 other settings. No cell discrepancy. Seven accuracy, eleven Macro-F1 and eight kappa cells are blank in both source and export; no experiment has all three metrics blank.
- Compared all 31 paper CSV and 56 experiment CSV records to JSON; only Python boolean spelling (`False` in CSV) differs from JSON boolean syntax, with equivalent value. Checked all 31 paper Markdown entries and all 56 experiment Markdown entries within their respective sheet sections, including numeric values, source URLs and caveats. No mismatch. The Markdown source key must include its section because paper ID and row numbers can recur across workbook sheets.
- Compared the first eight public local models against `reports/publication-figures/summary.json`, the ninth Sleepyland model against `evidence/sleepyland-yasa-development.json`, all nine CSV rows and control deltas against `reports/judge-guide/local-comparison.json`. Compared all fifteen score CSV rows (five systems × score/TST/WASO) against `reports/score-extension/summary.json`, including MAE, bias, P90, MAE interval and 61/40 denominators. Checked displayed `docs/RESULTS.md` table values to published precision. No mismatch.
- Verified all thirteen paths/hashes in `reports/judge-guide/manifest.json` and 31/56 source workbook bindings. Checked local target existence for links and images in README, six new guide pages and both literature Markdown pages: no missing targets. Checked the public ML review JSON for machine-specific path strings; `python tools/check_publication.py` is now the public command and `public_sanitization` records the original byte hash.

## Not performed and evidence limits

No raw EDF, participant-linked prediction, private model, audit holdout, training run, new external numeric-paper verification, live CI re-run or complete public test suite was accessed or executed for this review. The hosted receipt and prior test counts are inspected records, not a fresh invocation by this verifier. The seven judgments concern usability only; they do not improve the scientific rubric or resolve baseline, rights, calibration, clinical or physical-device gaps.

Public derivative: the machine-specific workbook path was removed; scientific scores and judgments are unchanged.
