"""MemoryFacade: composes thread, user and long-term memory into model context."""

from __future__ import annotations

from collections.abc import Sequence

import structlog

from agent_svc.config.settings import MemorySettings
from agent_svc.domain.models import Message, Principal
from agent_svc.domain.ports import LongTermMemory, ThreadMemory, UserMemory

log = structlog.get_logger()


def estimate_tokens(m: Message) -> int:
    return len(m.content) // 4 + 4


class MemoryFacade:
    def __init__(
        self,
        settings: MemorySettings,
        *,
        thread: ThreadMemory | None = None,
        user: UserMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self._s, self.thread, self.user, self.long_term = settings, thread, user, long_term

    @property
    def stateful(self) -> bool:
        """True when the server owns the history (clients then send only the new turn)."""
        return self.thread is not None

    @staticmethod
    def thread_key(principal: Principal, thread_id: str) -> str:
        # Threads are private to their creator: the user id is part of the storage key (IDOR-safe)
        return f"{principal.subject}/{thread_id}"

    async def build_context(
        self, principal: Principal, thread_id: str, new: Sequence[Message]
    ) -> list[Message]:
        history: list[Message] = []
        if self.thread:
            try:
                history = await self.thread.load(
                    principal.tenant, self.thread_key(principal, thread_id)
                )
            except Exception:
                log.warning("thread_memory_load_failed", exc_info=True)

        system = await self._system_context(principal, new)
        budget = self._s.max_context_tokens - sum(map(estimate_tokens, new))
        budget -= estimate_tokens(system) if system else 0
        kept: list[Message] = []
        for m in reversed(history):  # keep the most recent turns that fit
            budget -= estimate_tokens(m)
            if budget < 0:
                break
            kept.append(m)
        kept.reverse()
        return [*([system] if system else []), *kept, *new]

    async def _system_context(self, p: Principal, new: Sequence[Message]) -> Message | None:
        parts: list[str] = []
        try:
            if self.user:
                facts = await self.user.get(p.tenant, p.subject)
                if facts:
                    parts.append(
                        "Known user facts: " + "; ".join(f"{k}={v}" for k, v in facts.items())
                    )
            if self.long_term and new:
                query = next((m.content for m in reversed(new) if m.role == "user"), "")
                hits = await self.long_term.recall(
                    p.tenant, p.subject, query, self._s.long_term_top_k
                )
                if hits:
                    parts.append(
                        "Relevant past memories:\n" + "\n".join(f"- {h.text}" for h in hits)
                    )
        except Exception:
            log.warning("memory_recall_failed", exc_info=True)
        return Message("system", "\n".join(parts)) if parts else None

    async def save_turn(
        self, p: Principal, thread_id: str, new: Sequence[Message], answer: str
    ) -> None:
        try:
            if self.thread:
                await self.thread.append(
                    p.tenant,
                    self.thread_key(p, thread_id),
                    [*new, Message("assistant", answer)],
                )
            if self.long_term and self._s.long_term_write:
                user_text = next((m.content for m in reversed(new) if m.role == "user"), "")
                await self.long_term.remember(
                    p.tenant, p.subject, f"User: {user_text}\nAssistant: {answer}", thread=thread_id
                )
        except Exception:
            log.warning("memory_save_failed", exc_info=True)
