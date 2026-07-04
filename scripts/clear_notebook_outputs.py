#!/usr/bin/env python
"""Strip execution outputs from notebooks before commit (removes path leakage)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def clear_notebook(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        cell["execution_count"] = None
        cell["outputs"] = []
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        print("No notebooks found.")
        return 1
    for path in paths:
        clear_notebook(path)
        print(f"Cleared outputs: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
