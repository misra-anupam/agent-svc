# ADR 0002: Bridge every framework to the LLM port

**Status:** Accepted

## Context

LangGraph, ADK, MAF, CrewAI and deepagents each call models their own way, so retry, breaker, fallback and metering would have to be re-implemented five times.

## Decision

Each adapter wraps our resilient `LLMClient` as the framework's own model type (a LangChain chat model, an ADK `BaseLlm`, an MAF chat client, a CrewAI `BaseLLM`) and lets the framework run its native tool loop. Tool calling is part of the port.

## Consequences

Resilience, metering and fallback are uniform, and one contract suite proves all five orchestrators emit the same event grammar. Cost: we depend on each framework's extension points, which drift (CrewAI runs sync in a worker thread; MAF stops silently at its iteration cap, so we enforce the cap ourselves).
