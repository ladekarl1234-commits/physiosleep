# ADR 0005: Pinned native baselines and separate environments

Date: 2026-09-24

Status: **Accepted; execution incomplete**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Use serious upstream recipes, named physiological inputs and isolated runtimes. Adapt split/cropping only through versioned contracts.

## Alternatives and consequences

One combined runtime cannot safely satisfy incompatible legacy dependencies. Missing rights or parity leave a slot blocked.

## Verification and revisit condition

Pinned registry, source/environment hashes and actual native output parity. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
