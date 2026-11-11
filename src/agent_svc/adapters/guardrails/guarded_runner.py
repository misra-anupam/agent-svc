"""GuardedRunner: input/output guardrails around any AgentRunner (decorator).

Input: the newest user message is checked before the model sees it (block -> ``policy_violation``,
redact -> the redacted text is what the agent receives). Output, ``buffer`` mode: text deltas are
held until the full answer passes the output rails, so a blocked answer never leaves the service.
``stream`` mode forwards deltas immediately and re-checks as text accumulates (lower latency, but
a prefix may already be out when a rail fires).

Each check is surfaced as a ``guardrail`` step so it shows up in the UI and in traces.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

from agent_svc.adapters.guardrails.chain import ChainOutcome, GuardrailChain
from agent_svc.config.settings import GuardrailSettings
from agent_svc.domain.exceptions import PolicyViolationError
from agent_svc.domain.models import (
    AgentEvent,
    AgentRequest,
    MessageCompleted,
    MessageDelta,
    Principal,
    RunStarted,
    StepFinished,
    StepKind,
    StepStarted,
)
from agent_svc.domain.ports import AgentRunner, GuardrailStage

# (input chain, output chain) for a principal; ``None`` disables guarding for that principal.
PolicyResolver = Callable[[Principal], tuple[GuardrailChain, GuardrailChain] | None]


def _step(stage: GuardrailStage, rid: str, outcome: ChainOutcome, ms: int) -> list[AgentEvent]:
    sid = f"guardrail-{stage.value}-{uuid.uuid4().hex[:8]}"
    rails = ",".join(t.rail for t in outcome.trail)
    return [
        StepStarted(
            run_id=rid,
            step_id=sid,
            kind=StepKind.GUARDRAIL,
            name=stage.value,
            input={"rails": rails},
        ),
        StepFinished(
            run_id=rid,
            step_id=sid,
            ok=not outcome.blocked,
            output_summary=f"{outcome.result.action}: {outcome.result.reason}".rstrip(": "),
            duration_ms=ms,
        ),
    ]


class GuardedRunner:
    def __init__(
        self, inner: AgentRunner, resolve: PolicyResolver, settings: GuardrailSettings
    ) -> None:
        self._inner, self._resolve, self._cfg = inner, resolve, settings

    async def _check(
        self,
        chain: GuardrailChain,
        stage: GuardrailStage,
        text: str,
        request: AgentRequest,
        context: str = "",
    ) -> tuple[ChainOutcome, list[AgentEvent]]:
        started = time.perf_counter()
        outcome = await chain.evaluate(
            text, stage=stage, principal=request.principal, context=context
        )
        ms = int((time.perf_counter() - started) * 1000)
        return outcome, _step(stage, request.run_id, outcome, ms)

    @staticmethod
    def _violation(stage: GuardrailStage, outcome: ChainOutcome) -> PolicyViolationError:
        r = outcome.result
        return PolicyViolationError(
            f"{stage.value} blocked by guardrail '{r.rail}'",
            rail=r.rail,
            stage=stage.value,
            reason=r.reason,
        )

    async def stream(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        chains = self._resolve(request.principal)
        if chains is None:
            async for ev in self._inner.stream(request):
                yield ev
            return
        chain_in, chain_out = chains
        prompt = request.last_user_text

        outcome, events = await self._check(chain_in, GuardrailStage.INPUT, prompt, request)
        if outcome.blocked:
            yield RunStarted(
                run_id=request.run_id,
                thread_id=request.thread_id,
                agent_name=request.agent_name,
                agent_version=request.agent_version,
            )
            for ev in events:
                yield ev
            raise self._violation(GuardrailStage.INPUT, outcome)
        buffered_in = events
        if outcome.result.action == "redact":
            request = _with_user_text(request, outcome.result.text)

        buffer: list[MessageDelta] = []
        seen = ""
        checked_at = 0
        announced = False
        async for ev in self._inner.stream(request):
            if isinstance(ev, RunStarted):
                yield ev
                for e in buffered_in:
                    yield e
                announced = True
                continue
            if not announced:  # inner runner did not announce a run: emit the guard steps first
                for e in buffered_in:
                    yield e
                announced = True
            if isinstance(ev, MessageDelta):
                seen += ev.text
                if self._cfg.output_mode == "stream":
                    if len(seen) - checked_at >= self._cfg.stream_check_chars:
                        checked_at = len(seen)
                        mid, mid_events = await self._check(
                            chain_out, GuardrailStage.OUTPUT, seen, request, prompt
                        )
                        for e in mid_events:
                            yield e
                        if mid.blocked:
                            raise self._violation(GuardrailStage.OUTPUT, mid)
                    yield ev
                else:
                    buffer.append(ev)
                continue
            if isinstance(ev, MessageCompleted):
                text = ev.text or seen
                final, final_events = await self._check(
                    chain_out, GuardrailStage.OUTPUT, text, request, prompt
                )
                for e in final_events:
                    yield e
                if final.blocked:
                    raise self._violation(GuardrailStage.OUTPUT, final)
                if final.result.action == "redact":
                    if self._cfg.output_mode == "buffer":
                        yield MessageDelta(run_id=ev.run_id, text=final.result.text)
                    ev = replace(ev, text=final.result.text)
                else:
                    for d in buffer:
                        yield d
                buffer.clear()
            yield ev


def _with_user_text(request: AgentRequest, text: str) -> AgentRequest:
    msgs = list(request.messages)
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].role == "user":
            msgs[i] = replace(msgs[i], content=text)
            break
    return replace(request, messages=tuple(msgs))
