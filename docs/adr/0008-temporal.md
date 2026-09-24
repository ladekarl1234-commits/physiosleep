# ADR 0008: Train-only temporal priors and emission semantics

Date: 2026-09-24

Status: **Accepted**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Fit priors within training participants; handle gaps/boundaries; zero weights recover argmax. Preserve original emissions separately from decoded stages.

## Alternatives and consequences

Calling emissions decoder posterior probabilities would misstate uncertainty. Whole-record centered features imply offline use.

## Verification and revisit condition

Exhaustive short-sequence tests and saved hard-label comparisons. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
