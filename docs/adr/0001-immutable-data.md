# ADR 0001: Immutable originals and calibrated data

Date: 2026-09-24

Status: **Accepted**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Preserve original EDFs, spreadsheets and manifests; bind derivatives to source hashes and exact transformations. Resolve reader defects through explicit compatibility derivatives.

## Alternatives and consequences

In-place header repairs were rejected because they break provenance. Costs include derivative storage and stricter validation.

## Verification and revisit condition

Timing and source-identity tests; original content hashes. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
