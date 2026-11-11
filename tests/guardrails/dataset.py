"""Adversarial dataset shared by every rail test (OWASP LLM01/02/05/07/10 + benign controls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PATH = Path(__file__).parent / "adversarial.jsonl"


@dataclass(frozen=True)
class Case:
    id: str
    owasp: str
    stage: str
    text: str
    expect: str


def load() -> list[Case]:
    return [Case(**json.loads(line)) for line in PATH.read_text().splitlines() if line.strip()]
