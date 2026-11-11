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
| `metadata.simulate` | `{"reply", "tools":[{"name","arguments"}]}`: what the scripted model does in `--simulate` runs |
| `metadata.reference` | a full reference answer (used by the `adk_eval` ROUGE metric) |

`make eval` replays the cases through the real service stack (guardrails, tools, orchestrator) with a
scripted model, so the gate needs no API key. It guards routing, guardrails and the answer contract,
not model quality: for that run `uv run agent-svc-eval --real --exclude-tags simulate-only` against
your configured model. Thresholds live in `thresholds.yaml`.
