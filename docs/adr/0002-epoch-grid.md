# ADR 0002: Fixed full-record epoch timeline

Date: 2026-09-24

Status: **Accepted**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Use half-open 30-second PSG epochs; retain invalid locations; emit every complete epoch and let only the evaluator mask reference-invalid labels.

## Alternatives and consequences

Label-guided cropping and invalid-epoch compaction change the task and denominator. Full recordings define a different, Wake-heavy estimand; no difficulty ranking is established.

## Verification and revisit condition

Timing, prediction-grid and mask mismatch rejection tests. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
