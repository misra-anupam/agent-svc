# Golden dataset and evals

`evals/golden/*.jsonl` holds cases; `evals/golden/thresholds.yaml` holds the gate. Field reference:
[`evals/golden/README.md`](https://github.com/misra-anupam/agent-svc/blob/main/evals/golden/README.md).

```bash
make eval                                            # scripted model, local scorer, no API key
make eval EVAL_ARGS="--evaluator mlflow_genai"       # logs a tagged MLflow run
uv run agent-svc-eval --real --exclude-tags simulate-only   # your configured model
```

Exit code is `1` when any threshold fails, so CI can gate on it. A report JSON is written to
`evals/reports/`.

## What the default gate proves

It replays the cases through the **real stack** (guardrails, tools, orchestrator) with a scripted
model (`metadata.simulate`). That guards tool routing, trajectories, guardrail behaviour and the answer
contract. It does **not** measure model quality: use `--real` for that, ideally on a schedule.

## Metrics

| Metric | Meaning |
|---|---|
| `answer_match` | expected text is in the answer |
| `tool_trajectory` | exact ordered tool calls |
| `error_expectation` | the run ended with the expected error (or none) |
| `unexpected_error_rate` | runs that failed when they should not have |
| `latency_p50_ms`, `latency_p95_ms`, `cost_usd_total`, `tokens_mean` | operational |

A thresholded metric that cannot be computed fails the gate rather than being skipped.

## Prompts

`AGENT_PROMPTS__REF=system@prod` loads the system prompt from the store at startup; `/v1/agent`
reports the version in use. File store layout: `prompts/<name>/<n>.txt` plus `aliases.json`.
