# ADR 0013: Separate sanitized public repository

Date: 2026-09-24

Status: **Accepted by owner publication request**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Publish an explicit allowlist of source, synthetic tests, aggregate data, figures and reports in a separate Git repository.

## Alternatives and consequences

Original no-Git research workspace stays intact. Data, labels, weights, credentials, private logs and font binaries remain local; full training replay is consequently not included.

## Verification and revisit condition

Actual staged-content scan, checksums, portable links and clean-clone smoke checks. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
