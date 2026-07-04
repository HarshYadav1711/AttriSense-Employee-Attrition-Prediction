#!/usr/bin/env python
"""Launch AttriSense with the project virtual environment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"


def _venv_python() -> Path | None:
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else None


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _required_sklearn_version() -> str:
    results_path = ROOT / "models" / "training_results.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if payload.get("sklearn_version"):
            return str(payload["sklearn_version"])
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("scikit-learn=="):
            return line.split("==", 1)[1].strip()
    return "1.9.0"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def _ensure_venv() -> Path:
    python = _venv_python()
    if python is not None:
        return python

    print("Creating project virtual environment in .venv ...")
    _run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    python = _venv_python()
    if python is None:
        raise RuntimeError("Failed to create .venv")

    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    _run([str(python), "-m", "pip", "install", "-e", str(ROOT)])
    return python


def _ensure_sklearn(python: Path) -> None:
    required = _required_sklearn_version()
    probe = subprocess.run(
        [str(python), "-c", "import sklearn; print(sklearn.__version__)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    current = probe.stdout.strip() if probe.returncode == 0 else ""
    if _parse_version(current) >= _parse_version(required):
        return

    print(f"Installing scikit-learn>={required} into .venv (found {current or 'none'}) ...")
    _run([str(python), "-m", "pip", "install", f"scikit-learn=={required}"])


def main() -> int:
    python = _ensure_venv()
    _ensure_sklearn(python)
    cmd = [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app" / "main.py"),
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
