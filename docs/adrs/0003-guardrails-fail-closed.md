# ADR 0003: Guardrails fail closed

**Status:** Accepted

## Context

A guardrail sidecar can be down, slow or wrong. Silently skipping it would remove protection exactly when something is off.

## Decision

`guardrails.fail_mode=closed` blocks when a rail errors or times out; `open` is allowed outside production only (production refuses to start with it). Output defaults to `buffer` mode so a blocked answer never reaches the client.

## Consequences

Safer by default at the price of availability when a sidecar is down, and added latency in `buffer` mode (the answer streams only after the output check). `stream` mode is available for latency-sensitive use, with the documented risk that a prefix may already be sent.
