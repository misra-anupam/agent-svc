# API reference

Generated from the OpenAPI schema (`agent-svc` 0.1.0) by `scripts/gen_openapi.py`. A live, interactive version is served at `/docs` outside production.

## `GET /healthz`

Liveness probe

Responses: 200

## `GET /readyz`

Readiness probe

Responses: 200

## `GET /v1/agent`

Agent name and version

Responses: 200

## `DELETE /v1/memory`

Erase all of my user and long-term memory

| parameter | in | required |
|---|---|---|
| `authorization` | header | False |

Responses: 204, 422

## `GET /v1/memory/user`

Read my durable user memory

| parameter | in | required |
|---|---|---|
| `authorization` | header | False |

Responses: 200, 422

## `PUT /v1/memory/user`

Merge facts into user memory

| parameter | in | required |
|---|---|---|
| `authorization` | header | False |

Responses: 200, 422

## `GET /v1/runs/{run_id}/events`

Resume or replay a run's event stream

| parameter | in | required |
|---|---|---|
| `run_id` | path | True |
| `last-event-id` | header | False |
| `authorization` | header | False |

Responses: 200, 422

## `DELETE /v1/threads/{thread_id}`

Delete a thread

| parameter | in | required |
|---|---|---|
| `thread_id` | path | True |
| `authorization` | header | False |

Responses: 204, 422

## `GET /v1/threads/{thread_id}`

Read a thread's history

| parameter | in | required |
|---|---|---|
| `thread_id` | path | True |
| `authorization` | header | False |

Responses: 200, 422

## `POST /v1/threads/{thread_id}/chat`

Chat with the agent (Server-Sent Events)

| parameter | in | required |
|---|---|---|
| `thread_id` | path | True |
| `idempotency-key` | header | False |
| `cache-control` | header | False |
| `authorization` | header | False |

Responses: 200, 422
