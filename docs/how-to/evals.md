# Golden dataset and evals

`evals/golden/*.jsonl` holds cases; `evals/golden/thresholds.yaml` holds the gate. Field reference:
[`evals/golden/README.md`](https://github.com/misra-anupam/agent-svc/blob/main/evals/golden/README.md).

```bash
make eval                                            # your model (OPENROUTER_API_KEY), MLflow GenAI scorer on sqlite
```

Exit code is `1` when any threshold fails, so CI can gate on it. A report JSON is written to
`evals/reports/`.

## What the default gate proves

It replays the cases through the **real stack** (guardrails, tools, orchestrator) with your configured
model. The CI step runs only when the `OPENROUTER_API_KEY` secret is set. Cases tagged `simulate-only`
are skipped. Real models are non-deterministic, so a flaky gate usually means thresholds that are too tight.

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
reports the version in use. Prompts live in the MLflow Prompt Registry (`AGENT_PROMPTS__MLFLOW_TRACKING_URI`).
