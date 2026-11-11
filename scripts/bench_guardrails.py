"""Measure per-check latency of the in-process rails over the adversarial dataset.

    PYTHONPATH=. uv run python scripts/bench_guardrails.py [iterations]

HTTP sidecar rails are dominated by network + model time; measure those against your own
deployment (the ``agent_guardrail_seconds`` histogram records them in production).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

from tests.guardrails.dataset import load

from agent_svc.adapters import registries as reg
from agent_svc.config.settings import Settings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailStage


async def bench(name: str, iterations: int) -> tuple[float, float, float] | None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    try:
        rail = reg.guardrails.create(name, settings)
    except Exception as exc:
        print(f"{name:<10} skipped ({type(exc).__name__}: {exc})")
        return None
    cases = [c for c in load() if len(c.text) < 2000]
    samples: list[float] = []
    for _ in range(iterations):
        for c in cases:
            t0 = time.perf_counter()
            await rail.check(c.text, stage=GuardrailStage(c.stage), principal=Principal("bench"))
            samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return statistics.mean(samples), samples[len(samples) // 2], samples[int(len(samples) * 0.95)]


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"{'rail':<10} {'mean ms':>9} {'p50 ms':>9} {'p95 ms':>9}")
    for name in ("builtin", "presidio"):
        if (r := await bench(name, n)) is not None:
            print(f"{name:<10} {r[0]:>9.3f} {r[1]:>9.3f} {r[2]:>9.3f}")


if __name__ == "__main__":
    asyncio.run(main())
