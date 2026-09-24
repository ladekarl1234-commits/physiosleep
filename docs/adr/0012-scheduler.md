# ADR 0012: Bounded single-job local execution

Date: 2026-09-24

Status: **Accepted**

## Context

The project must preserve reproducibility and distinguish development, confirmation and provisional downstream work. See the [current report](../REPORT.md) for measured evidence.

## Decision

Use one heavy job, resource reservations, PID/birth ownership, atomic records and resumable checkpoints; preserve failed attempts.

## Alternatives and consequences

A time pause does not establish convergence. Paid compute or health-data upload requires separate authorization.

## Verification and revisit condition

Supervisor regression fixtures and terminal attempt receipts. Revisit only through an explicit versioned decision; preserve prior evidence and confirmation exposure history.
