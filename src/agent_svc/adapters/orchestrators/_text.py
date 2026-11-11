from __future__ import annotations

import json
from typing import Any


def summarize(obj: Any, limit: int = 500) -> str:
    """Compact, truncated string form for step input/output shown in the UI."""
    try:
        text = obj if isinstance(obj, str) else json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        text = repr(obj)
    return text if len(text) <= limit else text[: limit - 1] + "…"
