"""Dependency-free rails: injection heuristics, secrets, output handling, prompt-leak, length.

Heuristics are a cheap first line of defence, not a complete one: add a classifier rail (see add-adapter) for real coverage. Text is normalised (NFKC,
zero-width characters stripped, case-folded) before matching so trivial obfuscation does not
bypass the patterns.
"""

# ruff: noqa: E501 - regex tables read better unwrapped
from __future__ import annotations

import re
import unicodedata

from agent_svc.config.settings import GuardrailSettings, Settings
from agent_svc.domain.models import Principal
from agent_svc.domain.ports import GuardrailResult, GuardrailStage

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

_INJECTION = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|preceding) (?:instructions|prompts?|rules|messages)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your) (?:instructions|prompts?|rules|guidelines)",
    r"forget (?:everything|all|your) (?:you|instructions|rules|above)",
    r"(?:reveal|show|print|repeat|output|display|leak|tell me) (?:me )?(?:your |the )?(?:hidden |initial |original |full )?(?:system|developer) (?:prompt|message|instructions)",
    r"\byou are now (?:in )?(?:dan|developer mode|an? unrestricted|an? unfiltered)",
    r"\bdeveloper mode\b.*\b(?:enabled|on|activated)\b",
    r"\bjailbreak(?:ed|ing)?\b",
    r"pretend (?:that )?you (?:have no|are free of|don't have) (?:any )?(?:restrictions|rules|guidelines|filters)",
    r"(?:without|bypass|override|disable) (?:any |all |your )?(?:safety|content|ethical) (?:filters?|guidelines|restrictions|policies)",
    r"<\s*/?\s*(?:system|assistant|im_start|im_end)\s*>",
    r"\[\s*inst\s*\]|<<\s*sys\s*>>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION]

# (label, pattern). Matched on the original text so redaction preserves surrounding content.
_SECRETS = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("api_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
            re.DOTALL,
        ),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
]

# LLM05 (improper output handling): content a downstream renderer could execute or exfiltrate by.
_OUTPUT_HAZARDS = [
    (
        "script_tag",
        re.compile(r"<\s*script\b.*?(?:<\s*/\s*script\s*>|$)", re.IGNORECASE | re.DOTALL),
    ),
    (
        "event_handler",
        re.compile(r"\son(?:error|load|click|mouseover)\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE),
    ),
    ("javascript_url", re.compile(r"javascript\s*:", re.IGNORECASE)),
    # markdown image with data in the query string: renders as a request to an attacker host
    ("image_exfil", re.compile(r"!\[[^\]]*\]\(\s*https?://[^)\s]*\?[^)\s]+\)", re.IGNORECASE)),
]

_LEAK_WINDOW = 40  # a verbatim run of this many prompt characters in an answer is a leak


def _normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return " ".join(folded.split())  # collapse whitespace runs


def _redact(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, list[str]]:
    hits: list[str] = []
    for label, rx in patterns:
        text, n = rx.subn(f"[REDACTED:{label}]", text)
        if n:
            hits.append(label)
    return text, hits


class BuiltinRail:
    name = "builtin"

    def __init__(self, settings: GuardrailSettings, *, system_prompt: str = "") -> None:
        self._max = settings.max_input_chars
        self._prompt = _normalise(system_prompt)

    async def check(
        self, text: str, *, stage: GuardrailStage, principal: Principal, context: str = ""
    ) -> GuardrailResult:
        if stage is GuardrailStage.INPUT:
            return self._input(text)
        return self._output(text)

    def _input(self, text: str) -> GuardrailResult:
        if len(text) > self._max:
            return GuardrailResult(
                "block", rail=self.name, reason=f"input exceeds {self._max} chars"
            )
        norm = _normalise(text)
        for rx in _INJECTION_RE:
            if rx.search(norm):
                return GuardrailResult("block", rail=self.name, reason="prompt_injection")
        cleaned, hits = _redact(text, _SECRETS)
        if hits:
            return GuardrailResult("redact", cleaned, self.name, "secret:" + ",".join(hits))
        return GuardrailResult(text=text, rail=self.name)

    def _output(self, text: str) -> GuardrailResult:
        if self._leaks_prompt(text):
            return GuardrailResult("block", rail=self.name, reason="system_prompt_leak")
        cleaned, hits = _redact(text, [*_SECRETS, *_OUTPUT_HAZARDS])
        if hits:
            return GuardrailResult("redact", cleaned, self.name, "output:" + ",".join(hits))
        return GuardrailResult(text=text, rail=self.name)

    def _leaks_prompt(self, text: str) -> bool:
        p = self._prompt
        if len(p) < _LEAK_WINDOW:
            return False
        out = _normalise(text)
        return any(p[i : i + _LEAK_WINDOW] in out for i in range(0, len(p) - _LEAK_WINDOW + 1, 8))


def build(settings: Settings) -> BuiltinRail:
    return BuiltinRail(settings.guardrails, system_prompt=settings.orchestrator.system_prompt)
