# ADR 0017: Final delivery from completed exploratory audits

Date: 25 September 2026. Status: accepted by explicit owner instruction. This decision supersedes the active experiment queue in the historical [reviewer roadmap](../ROAD_TO_90.md); it does not change the metric or formal gate in [ADR 0004](0004-metric-gate.md) or the audit exposure recorded by [ADR 0016](0016-exploratory-audits.md).

## Context

The owner requested an end to U-Sleep, Sleepyland and other unfinished baseline testing, and requested the final A/B Macro-F1 report, the English formula and reasoning, and a review-driven improvement pass. Both A and B already have complete exploratory scores for one frozen D60 seed-17 control. The eleven mandatory comparator slots, formal +0.02 point margin and simultaneous paired intervals do not have completion evidence.

## Decision

Stop further baseline and model experiments for this delivery. Publish a final report with the verified exploratory A/B values, model identity, class-level errors, uncertainty method, exact pooled Macro-F1 definition and limits of inference. Improve inaccurate or stale documentation and run public arithmetic, software and publication checks. Preserve failed/incomplete job evidence and original formal protocol. Do not infer that stopping work makes a historical gate pass.

## Consequences

The deliverable is a measured same-dataset exploratory evaluation and an inspectable research package. Audit A and B are consumed; future formal confirmation needs fresh people and a prospective protocol. The original gate remains `NOT_RUN`, and no claim of universal baseline superiority, clinical utility or working hardware follows. Simulated reviewer grades remain evidence grades under their original rubrics; documentation improvements do not automatically rescore missing experiments.
