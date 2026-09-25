# ADR 0016: Owner-authorized exploratory evaluation of both audits

Date: 25 September 2026. Status: accepted. Supersedes the reservation decision in [ADR 0015](0015-audit-order.md), without changing the benchmark criteria in [ADR 0004](0004-metric-gate.md).

## Context

The mandatory benchmark was incomplete. The owner requested immediate A/B scores and explicitly chose “Run both now as exploratory evaluations” after being informed that this consumes the two reserved holdouts and requires fresh confirmation participants.

## Decision

Evaluate the available all-development seed-17 EEG+EOG YASA-feature LightGBM control, with no audit-informed refitting or selection. Freeze its exact checkpoint, configuration, implementation, authorization and metric recipe before inference. Commit all 78 recordings' signal-only predictions and successful execution receipts before opening either phase's reference labels.

Keep the original participant allocation, full 30-second epoch grid, invalid-reference mask and fixed five-class pooled Macro-F1. Each phase has 39 recordings from 20 participants. Report separate descriptive participant-cluster 95% intervals from 10,000 cohort-stratified resamples.

## Consequences

The measurements are genuine held-out exploratory results for this control. They do not test the best development ensemble, complete the mandatory baseline comparison, prove the +0.02 margin, or pass either formal gate. Both cohorts lose their status as untouched confirmation sets. Future confirmation needs fresh participants.

The execution ledger distinguishes retirement, actual reference exposure, successful evaluation and formal gate status. A successful exploratory run cannot be promoted to a gate pass by editing a status file.

## Alternatives considered

Preserving the holdouts until all baselines were ready would retain the original confirmation design, but the owner explicitly replaced that choice. Calling the immediate results confirmatory would misrepresent the incomplete comparator and readiness requirements.
