from __future__ import annotations

from typing import Protocol

from agent_svc.domain.models import Principal


class Authenticator(Protocol):
    async def authenticate(
        self, authorization: str | None, *, client_ip: str | None
    ) -> Principal: ...
