# Developer scripts

Two helper scripts live in `scripts/`. Neither is needed to run the service.

## `new_adapter.py`: scaffold an adapter

```bash
make new-adapter PORT=guardrails NAME=acme_pii
```

- `PORT` is one of `guardrails`, `prompts`, `eval`, `tracing`, `cache`. `NAME` is snake_case.
- Creates `src/agent_svc/adapters/<port>/<name>.py`: a stub class plus the `build(settings)` factory the
  registry resolves lazily. It never overwrites an existing file and edits nothing else.
- Prints the manual steps that remain: register it in `adapters/registries.py`, add the settings
  `Literal` (and uv extra if it needs one), add it to the shared test suite, then `make check`.

You still write the port method yourself. See [Add an adapter](add-adapter.md) for the full walkthrough.

## `gen_openapi.py`: regenerate the API reference

```bash
make docs-api                                   # rewrites docs/api.md
uv run python scripts/gen_openapi.py --check    # exit 1 if docs/api.md is stale
```

- Builds the app and renders its OpenAPI schema into [API reference](../api.md). Do not edit `docs/api.md`
  by hand.
- `make docs-build` runs it first, and CI runs `--check`, so an endpoint change without a regenerated
  page fails the build.
