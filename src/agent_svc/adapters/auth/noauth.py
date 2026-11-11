from __future__ import annotations

from agent_svc.config.settings import Settings
from agent_svc.domain.models import Principal


class NoAuth:
    """Auth disabled: every caller is an anonymous principal identified by IP."""

    def __init__(self, default_tier: str) -> None:
        self._tier = default_tier

    async def authenticate(self, authorization: str | None, *, client_ip: str | None) -> Principal:
        return Principal(
            subject=f"ip:{client_ip or 'unknown'}",
            tier=self._tier,
            ip=client_ip,
            anonymous=True,
        )


def build(settings: Settings) -> NoAuth:
    return NoAuth(settings.ratelimit.default_tier)
