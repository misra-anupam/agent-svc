"""Load smoke: fire concurrent chat requests at a running server and report status codes.

Usage:  uv run python tests/load/limits_smoke.py --url http://localhost:8000 -n 60 -c 15
Shows tier limits (429 rate_limited), budget cutoffs (429 token_budget_exceeded) and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import time

import httpx


async def one(client: httpx.AsyncClient, i: int, headers: dict[str, str]) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    r = await client.post(
        f"/v1/threads/load-{i}/chat",
        json={"messages": [{"role": "user", "content": f"hello {i}"}]},
        headers=headers,
    )
    code = ""
    if r.status_code != 200:
        code = r.json().get("code", "")
    return r.status_code, code, time.perf_counter() - t0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=60, help="total requests")
    ap.add_argument("-c", type=int, default=15, help="concurrency")
    ap.add_argument("--token", default="", help="bearer token (when auth is enabled)")
    a = ap.parse_args()
    headers = {"Authorization": f"Bearer {a.token}"} if a.token else {}
    sem = asyncio.Semaphore(a.c)
    async with httpx.AsyncClient(base_url=a.url, timeout=60) as client:

        async def guarded(i: int) -> tuple[int, str, float]:
            async with sem:
                return await one(client, i, headers)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(a.n)))
        wall = time.perf_counter() - t0

    hist = collections.Counter((s, c) for s, c, _ in results)
    lat = sorted(d for _, _, d in results)
    print(f"{a.n} requests in {wall:.2f}s ({a.n / wall:.1f} rps), concurrency {a.c}")
    for (status, code), n in sorted(hist.items()):
        print(f"  {status} {code or 'ok':<24} x{n}")
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1]
    print(f"  latency p50={p50 * 1000:.0f}ms p95={p95 * 1000:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
