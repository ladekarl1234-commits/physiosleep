# ADR 0007: Measured feature models before complex candidates

Date: 2026-09-24

Status: **Accepted development choice**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Use native features and local classifiers to establish a complete pipeline; compare fixed seeds, probability ensembles and train-only temporal priors.

## Alternatives and consequences

Development gains are small, N1 weak and convergence unresolved. No new-architecture or state-of-the-art claim follows.

## Verification and revisit condition

Aggregate metrics, saved prediction verification and ablations. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
