# ADR 0004: Exact five-class Macro-F1 and unchanged superiority gate

Date: 2026-09-24

Status: **Accepted**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Require all eleven slots, exact delta >=1/50 against every comparator, positive simultaneous paired participant intervals and artifact attestation.

## Alternatives and consequences

Average-baseline comparisons, relative gains and rounded thresholds are rejected. Passing superiority CI differs from lower CI >=0.02.

## Verification and revisit condition

Metric and adversarial gate fixtures; actual audits remain NOT_RUN. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
