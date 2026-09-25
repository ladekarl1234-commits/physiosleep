# Evidence required for reviewer scores of 90

**Current feasibility check, 25 September 2026.** The owner renewed the goal of a review score above 90 while explicitly retaining the stop on U-Sleep, Sleepyland and other mandatory-baseline experiments. The [final report](FINAL_REPORT.md) remains the measured conclusion. The former experiment queue below is historical, not permission to restart it.

The owner goal is to earn more than 90 from the existing reviewers, as far as the
evidence allows. Rubrics, the eleven mandatory baseline slots and both audit
criteria stay unchanged. Current independently reviewed grades are 60/100 for
data/ML and 75/100 for the sleep-specialist perspective. Agent grades are not
clinical or competition approval.

## Feasibility under current constraints

The fixed [data/ML rubric](../evidence/judging/ml-exploratory.json) now scores [60/100](../evidence/judging/ml-reader-annotation-regrade.json) after independently reviewed points for complete source-byte provenance and a separate 197-pair annotation/grid crosscheck. Mandatory comparators are 2/12, leaving 10 unavailable points under the stop; formal confirmation is 0/10, and both original holdouts are consumed, leaving the untouched-holdout/official-rules criterion at 0/3. Even **granting every other criterion full credit** gives an optimistic ceiling of **100 - 10 - 10 - 3 = 77/100**. If independent fresh-host full-training replay also remains absent (0/4), the ceiling is **73/100**. These are mathematical bounds, not predicted scores.

The fixed [sleep-specialist rubric](../evidence/judging/sleep-specialist-exploratory.json) now scores [75/100](../evidence/judging/sleep-specialist-hardware-replay-regrade.json) after independently reviewed points for test-linked sensitivity interpretation and public replay of six generated hardware cases. Matched comparison/confirmation remains 1/8, clinical/subjective validity 0/5 and paired physical-device qualification 0/5. Holding those at their current evidence-backed values gives an optimistic ceiling of **100 - 7 - 5 - 5 = 83/100**. Failed score-component targets, uncertain whole-night boundaries and same-source population limit the realistic result further.

The eight remaining specialist points within that ceiling are one each for independently established sleep windows and population applicability, three for actual component fidelity, two for a source-supported contact/montage specification, and one for measured or otherwise qualifying acquisition conversion. The current Sleep-EDF material lacks independently bounded whole nights and a genuinely external test population; the fixed score/component targets fail. The public generated replay adds traceability only. Thus **83 is an algebraic upper bound, not a locally executable target** under the present evidence.

No amount of prose, figures, synthetic checks or regrading can legitimately supply the missing comparison, fresh confirmation, clinical or physical measurements. The specialist's current 75-to-83 gap is mostly those missing measurements; the ML 60-to-77 gap includes native comparator lineage, failed score/precision targets and a fresh-host full-training replay unavailable from this public clone. The sections below are retained so a future authorized program can see what evidence was missing; current work may improve traceability and correct defects without claiming the >90 target has been achieved.

## Former proposed work, in evidence order

The owner has since authorized exploratory evaluation of the original A/B cohorts;
see [the audit guide](EXPLORATORY_AUDITS.md). Those cohorts are retired from
confirmation. The audit scores describe the frozen D60 control, while the steps
below remain requirements for broader benchmark and scientific evidence.

1. Continue the native AttnSleep recipe from its verified epoch-65
   checkpoint. Keep its 100-epoch folds, full grids and fitted ancestry. Checkpoint
   progress or training loss alone cannot earn complete-comparator credit.
2. Preserve U-Time's failed execution and all sealed bytes. Read-only restoration
   and prepared-data checks passed. The corrected pointer implementation passed
   focused tests; the selected replacement is a fresh original-seed fit, pending
   renewed source-bound qualification. No successful continuation fit is claimed.
3. Assess U-Time throughput before extensive continuation. The first invocation
   consumed 12,493.5 seconds. Repeating that wall cost for 80 further epochs would
   be about 278 hours for one inner fold, assuming no improvement after epoch 1.
   This extrapolation motivates equivalent performance work, not fewer native
   epochs or altered stopping rules.
4. Completed and independently verified the already-frozen best decoder's score
   agreement on the original 61 eligible windows / 40 people. Score/TST/WASO MAEs
   are 5.677 points / 24.331 minutes / 63.088 minutes: all miss their targets.
   Paired intervals versus EEG+EOG include zero. Formula, eligibility, controls and
   2,000 common participant draws were unchanged; no fitting or audit exposure.
5. Complete the remaining permitted native families and source-qualified variants.
   Resolve genuine external rights/lineage gaps through actual applicable evidence.
   The 25 September public-source recheck resolved none of the existing blockers.
   U-Sleep's 119 development signal records are now prepared and the first inner
   fold's 95 inputs are bound; this is [data readiness](../evidence/usleep-development-preparation.json),
   not a trained comparator or a grade increase. The integrated epoch lifecycle
   still needs actual qualification before training can start.
6. Meet development adequacy, finalist robustness and precision readiness. Obtain
   fresh confirmation participants, freeze a prospective protocol and exact
   systems, then apply the unchanged superiority criteria. The consumed original
   A/B cohorts cannot be reused as untouched confirmation.

## External evidence that code or prose cannot create

- Applicable MSA-CNN grant, clarification of TinySleepNet terms, intended-use and
  weight rights where needed, and released-model training-cohort exclusion evidence.
- Fresh independent confirmation participants: both original holdouts are now
  retired by owner authorization. B cannot become a retry pool.
- Independently established observation boundaries for whole-night score claims,
  and clinical/subjective criterion evidence for clinical construct validity.
- Actual board/instrument access, authorized safety/bench work and paired device
  measurements before physical qualification can receive credit.
- Independent full-training replay; public CI verifies synthetic code and aggregate
  arithmetic, not the protected-data experiment.

No reviewer is asked to change a rubric or award missing evidence. Regrade only
after completed, independently checked milestones. Record adverse results and
unchanged scores alongside improvements. The former baseline queue remains
stopped by the owner's explicit answer; current work is limited to evidence
checks, corrections and reproducibility improvements that do not reopen it.
