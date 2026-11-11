from __future__ import annotations

import json

from agent_svc.domain.models import AgentRequest, Message, Principal


def make_request(text: str = "hello world", run_id: str = "run1") -> AgentRequest:
    return AgentRequest(
        run_id=run_id,
        thread_id="t1",
        principal=Principal(subject="u1"),
        messages=(Message("user", text),),
        agent_name="default",
        agent_version="1.0.0",
    )


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    out: list[tuple[str, dict[str, object]]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((fields["event"], json.loads(fields["data"])))
    return out
