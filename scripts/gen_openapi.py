"""Generate docs/api.md from the app's OpenAPI schema (run: make docs-api).

    uv run python scripts/gen_openapi.py [--check]

``--check`` exits 1 if docs/api.md is out of date (use it in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_svc.config import Settings
from agent_svc.main import create_app

OUT = Path(__file__).parent.parent / "docs" / "api.md"


def render() -> str:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.app.environment = "test"
    spec = create_app(s).openapi()
    lines = [
        "# API reference",
        "",
        "Generated from the OpenAPI schema "
        f"(`{spec['info']['title']}` {spec['info']['version']}) by `scripts/gen_openapi.py`. "
        "A live, interactive version is served at `/docs` outside production.",
        "",
    ]
    for path, ops in sorted(spec["paths"].items()):
        for method, op in sorted(ops.items()):
            lines += [f"## `{method.upper()} {path}`", "", op.get("summary", ""), ""]
            if desc := op.get("description"):
                lines += [desc, ""]
            params = op.get("parameters", [])
            if params:
                lines += ["| parameter | in | required |", "|---|---|---|"]
                lines += [
                    f"| `{p['name']}` | {p['in']} | {p.get('required', False)} |" for p in params
                ]
                lines.append("")
            responses = ", ".join(sorted(op.get("responses", {})))
            lines += [f"Responses: {responses}", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text else 1)
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
