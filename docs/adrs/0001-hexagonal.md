# ADR 0001: Hexagonal architecture enforced by tooling

**Status:** Accepted

## Context

We need five orchestrators, many backends and optional SDKs without the core depending on any of them.

## Decision

Ports in `domain/ports`, adapters behind lazy registries, and `import-linter` contracts that fail CI when api/application import adapters or domain imports a framework.

## Consequences

Adding an adapter touches one module and one registry line. Cost: more indirection, and the composition root (`container.py`) knows every adapter.
