"""Per-tier guardrail policy: which rails run for which principal tier."""

from __future__ import annotations

from collections.abc import Callable

from agent_svc.adapters.guardrails.chain import GuardrailChain
from agent_svc.config.settings import GuardrailSettings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import Guardrail


class TierPolicyResolver:
    """Builds every referenced rail once, up front (so a missing extra fails at startup)."""

    def __init__(self, cfg: GuardrailSettings, make_rail: Callable[[str], Guardrail]) -> None:
        self._cfg = cfg
        names: set[str] = {*cfg.enabled_input, *cfg.enabled_output}
        for pol in cfg.tier_policy.values():
            names |= {*(pol.input or []), *(pol.output or [])}
        self._rails: dict[str, Guardrail] = {n: make_rail(n) for n in sorted(names)}
        self._chains: dict[tuple[str, ...], GuardrailChain] = {}

    def _chain(self, names: list[str]) -> GuardrailChain:
        key = tuple(names)
        if key not in self._chains:
            self._chains[key] = GuardrailChain(
                [self._rails[n] for n in names],
                fail_mode=self._cfg.fail_mode,
                timeout_seconds=self._cfg.rail_timeout_seconds,
            )
        return self._chains[key]

    def __call__(self, principal: Principal) -> tuple[GuardrailChain, GuardrailChain] | None:
        pol = self._cfg.tier_policy.get(principal.tier)
        if pol is not None and not pol.enabled:
            return None
        inp: list[str] = (
            pol.input if pol and pol.input is not None else list(self._cfg.enabled_input)
        )
        out: list[str] = (
            pol.output if pol and pol.output is not None else list(self._cfg.enabled_output)
        )
        return self._chain(inp), self._chain(out)

    async def aclose(self) -> None:
        for rail in self._rails.values():
            close = getattr(rail, "aclose", None)
            if close is not None:
                await close()
