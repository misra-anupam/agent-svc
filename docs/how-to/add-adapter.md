# Add an adapter

Every swappable concern is a port in `src/agent_svc/domain/ports/` with a registry in
`src/agent_svc/adapters/registries.py`.

```bash
make new-adapter PORT=guardrails NAME=acme      # generates the stubs below
```

1. **Implement the port** in `adapters/<port>/<name>.py` and expose a `build(settings)` factory.
   Import heavy SDKs *inside* the factory or class so the module imports without them.
2. **Register it** (lazy string, plus the uv extra that provides its dependencies):

    ```python
    guardrails.register("acme", "agent_svc.adapters.guardrails.acme:build", extra="acme")
    ```

3. **Add the extra** to `pyproject.toml` and run `uv lock`. If its pins conflict with the lock, leave it
   out of `pyproject.toml` and document `uv pip install ...`.
4. **Add the selector** to the matching `Literal[...]` in `config/settings.py` where one exists.
5. **Test it**: run the shared contract/adversarial suites against it.
    * orchestrators: append the name to `PROVIDERS` in `tests/contract/test_orchestrator_contract.py`
    * guardrails: the adversarial cases in `tests/guardrails/adversarial.jsonl`
    * evaluators: add it to the parametrised list in `tests/unit/test_golden_eval.py`
6. `make check`.

## Third-party packages

An external package can add an adapter without touching this repo, via entry points:

```toml
[project.entry-points."agent_svc.guardrails"]
acme = "acme_rails:build"
```

The registry loads the `agent_svc.<port>` group the first time an unknown name is requested.

## Rules adapters must follow

* Domain and application code never import adapters (`make arch` fails the build).
* A rail/evaluator/store that cannot reach its backend raises; policy (fail closed/open) is applied by
  the caller, not swallowed in the adapter.
* Blocking SDK calls go through `anyio.to_thread.run_sync`.
* Never send user content to a third party by default (disable vendor telemetry).
