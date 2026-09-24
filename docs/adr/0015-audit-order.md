# ADR 0015: Preserve confirmation after explicit owner clarification

Date: 2026-09-24

Status: **Accepted; latest owner decision**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Complete mandatory baselines and readiness, then Audit A; reserve Audit B for the frozen reduced model after A passes.

## Alternatives and consequences

The earlier request to test both immediately was clarified. Premature exposure, audit seed selection and B-as-retry are rejected.

## Verification and revisit condition

Exposure ledger and unchanged gate checks; both currently NOT_RUN. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
