# Golden dataset

One JSON object per line (`*.jsonl`); `#` lines are comments.

| field | meaning |
|---|---|
| `id` | unique |
| `input` | the user message |
| `expected` | substring the answer must contain (case-insensitive) |
| `expected_tools` | exact ordered tool-call trajectory (omit to skip) |
| `expect_error` | error code the run must end with, e.g. `policy_violation` |
| `tags` | filter with `--tags` / `--exclude-tags` |
| `metadata.simulate` | `{"reply", "tools":[{"name","arguments"}]}`: the scripted reply the unit tests serve for this case (`tests/scripted.py`); ignored by `make eval` |
| `metadata.reference` | a full reference answer (free-form; not scored by the built-in evaluators) |

`make eval` replays the cases through the real service stack (guardrails, tools, orchestrator) with
your configured model, so it needs `OPENROUTER_API_KEY`. Cases tagged `simulate-only` assume scripted
replies and are skipped unless you pass `--exclude-tags`. Thresholds live in `thresholds.yaml`; a real
model is non-deterministic, so tune them for it.
