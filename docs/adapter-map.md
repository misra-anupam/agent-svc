# Adapter map

The core only knows **ports**, which are small interfaces in `domain/ports/`. Each concrete tool lives in
an **adapter** that implements one port. Read the map left to right: the core, a group of ports, a port,
then the adapters you can choose for it. Dashed boxes are decorators: adapters that wrap another adapter
of the same port.

<div style="overflow-x:auto">
<a href="../assets/adapter-map.svg"><img src="../assets/adapter-map.svg" style="max-width:none" width="1186" alt="Mind map: the core connects to six groups of ports; each port connects to the adapters that implement it, with decorators shown dashed"></a>
</div>

The diagram scrolls sideways on narrow screens; click it to open the full-size image.

## How to read it

* **Stacks.** The LLM call passes through `Breaker`, `Retry`, `Fallback` and `Metered` before it reaches
  OpenRouter. Every agent run passes through `Traced`, `Cached` and `Guarded` before it reaches LangGraph.
  Tools pass through `Composite` and then `Guarded` (allowlist, schema check, timeout, size cap).
* **Several adapters, one chosen.** Memory, cache, run buffer, tracer and guardrails each list their
  options. A setting such as `AGENT_MEMORY__THREAD_PROVIDER` selects one.
* **Guardrails stack.** Unlike the other ports, several guardrail adapters can run together: pick them
  with `guardrails.enabled_input` and `guardrails.enabled_output`.

## How it is wired

* **Dependencies point inward.** `api` uses `application`, which uses `domain`. Adapters implement
  domain ports and are never imported by the layers above; `import-linter` fails the build if that breaks
  (see [Architecture](architecture.md)).
* **One place picks the adapters.** `container.py` reads `Settings` and asks each port's registry for the
  adapter named there. A registry entry is a lazy `module:attr`, so an unused SDK is never imported.
* **A missing extra fails loudly.** Selecting an adapter whose uv extra is not installed stops startup with
  an error that names the extra to install.
* **Adding one.** See [Add an adapter](how-to/add-adapter.md).
