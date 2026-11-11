# ADR 0002: Bridge every framework to the LLM port

**Status:** Accepted

## Context

Agent frameworks each call models their own way, so retry, breaker, fallback and metering would have to be re-implemented per framework.

## Decision

Each adapter wraps our resilient `LLMClient` as the framework's own model type (for LangGraph, a LangChain chat model) and lets the framework run its native tool loop. Tool calling is part of the port.

## Consequences

Resilience, metering and fallback are uniform, and one contract suite proves every orchestrator emits the same event grammar. Cost: we depend on each framework's extension points, which drift. ADK, MAF, CrewAI and deepagents adapters existed and were removed to keep the template small; the port makes them easy to re-add.
