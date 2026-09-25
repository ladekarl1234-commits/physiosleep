# Questions a critical judge should ask

[Guided evidence tour](JUDGE_GUIDE.md) · [Current results](RESULTS.md) · [Dated jobs](EXPERIMENTS.md)

**Have you beaten every mandatory baseline by two Macro-F1 points?** No. Best development gain is +0.003512 over one fixed control; the matched mandatory benchmark and both audits are incomplete.

**Why report 90.7% accuracy if N1 is weak?** We also report fixed-class Macro-F1 and the full confusion matrix. Wake accounts for 63.27% of valid epochs. Best-system N1 F1 is 0.4722; accuracy is not a substitute.

**Did people or nights leak across folds?** All nights follow their participant, and learned components use only fold-training people. Recipe selection uses development results, so D performance remains conditional on selection. Audits are procedural on a shared host; strong external access isolation is not claimed.

**What changed about A and B?** The owner explicitly authorized immediate exploratory evaluation of both cohorts on 25 September 2026. Their original confirmation reservation was retired. [ADR 0016](adr/0016-exploratory-audits.md) records that decision; a measured single-model score still cannot repair missing matched comparators or establish the required +0.02 margin.

**Are confidence and decoded-stage probabilities the same?** No. The transition decoder changes hard stages while saved probabilities remain classifier emissions. Confidence plots label their source and do not call them decoder posterior probabilities or calibrated clinical confidence.

**Is this a new model architecture?** The current strongest pipeline combines existing native physiological features, locally trained gradient boosting, seed ensembling and a train-only transition prior. Its present contribution is evaluated integration and reproducible contracts; new-architecture or state-of-the-art claims are unsupported.

**Does the sleep score measure health or restorative sleep?** No. It is a two-component unvalidated recording-window index. Agreement with the same formula on reference stages is not clinical construct validation. Only 61/119 nights and 40/60 people are score-eligible; all tested sensor models miss the planned joint error targets.

**Are two channels or two electrodes sufficient?** Not established. Among the original three sensor configurations (EOG, Fpz−Cz EEG and EEG+EOG), the EEG+EOG control had the highest staging Macro-F1. The full subset study and B remain incomplete. Two bipolar measurements conditionally need four sensing contacts plus a bias contact. Fpz−Cz requires Cz access; EOG geometry/polarity are still unresolved.

**Is the device built or ready for human use?** No. Illustrations are concept art, budgets are calculations, and fault checks use generated signals. Physical bench, safety and paired reference validation are pending.

**Can the evaluated pipeline operate in real time?** Not as evaluated. Whole-record normalization and centered context support offline analysis. Online operation needs a separately frozen causal pipeline and its own validation.

**Can anyone reproduce everything from this repository?** Public arithmetic, figures and synthetic tests are replayable. Exact training also needs controlled data, frozen manifests, source versions and model lineage that are intentionally not uploaded. No fresh external full-training replay is claimed.

**Did the agent judges certify it?** No. Their grades are simulated independent reviews under fixed rubrics. Documentation fixes can improve transparency but cannot create missing confirmation, clinical or physical evidence.
