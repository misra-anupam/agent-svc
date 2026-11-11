"""Scaffold an adapter: ``make new-adapter PORT=guardrails NAME=acme``.

Creates ``adapters/<port>/<name>.py`` (a stub with a ``build`` factory) and prints the registry line
and the remaining manual steps (see docs/how-to/add-adapter.md). Refuses to overwrite.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PORTS = {
    "guardrails": ("Guardrail", "check(self, text, *, stage, principal, context='')"),
    "prompts": ("PromptStore", "get(self, name, *, version=None, alias=None)"),
    "eval": ("Evaluator", "evaluate(self, records, *, agent_version, thresholds)"),
    "tracing": ("Tracer", "start_span(self, name, *, parent=None, attributes=None)"),
    "cache": ("Cache", "get(self, key)"),
}
ROOT = Path(__file__).parent.parent / "src" / "agent_svc" / "adapters"

TEMPLATE = '''"""{name} {port} adapter. TODO: implement the {proto} port (see domain/ports)."""

from __future__ import annotations

from agent_svc.config.settings import Settings


class {cls}:
    name = "{name}"

    # TODO: implement `{sig}` exactly as the {proto} protocol declares it.
    # Import heavy SDKs inside __init__ so this module imports without the extra installed.

    def __init__(self, settings: Settings) -> None:
        self._settings = settings


def build(settings: Settings) -> {cls}:
    return {cls}(settings)
'''


def main() -> int:
    if (
        len(sys.argv) != 3
        or sys.argv[1] not in PORTS
        or not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", sys.argv[2])
    ):
        print(f"usage: new_adapter.py <{'|'.join(PORTS)}> <snake_case_name>", file=sys.stderr)
        return 2
    port, name = sys.argv[1:]
    proto, sig = PORTS[port]
    path = ROOT / port / f"{name}.py"
    if path.exists():
        print(f"{path} already exists", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "__init__.py").touch()
    cls = "".join(p.title() for p in name.split("_")) + proto
    path.write_text(TEMPLATE.format(name=name, port=port, proto=proto, cls=cls, sig=sig))
    print(f"created {path.relative_to(ROOT.parent.parent.parent)}")
    print("next:")
    print(
        f'  1. register it in adapters/registries.py: <registry>.register("{name}", '
        f'"agent_svc.adapters.{port}.{name}:build")'
    )
    print("  2. add its uv extra and the settings Literal (docs/how-to/add-adapter.md)")
    print("  3. add it to the matching shared test suite, then `make check`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
