"""Locating an optional vendored gguf-py checkout (used only as an external oracle, never required)."""
import os
import sys
from pathlib import Path

ENV_VAR = "BALOON_GGUF_PY"


def gguf_py(start=None):
    """Return the directory of a gguf-py package, or None.

    Search order: $BALOON_GGUF_PY, then `gguf-py` in `start` and each of its parents (default: this file's directory).
    A directory counts only if it contains a `gguf` package (gguf/__init__.py)."""
    cands = []
    env = os.environ.get(ENV_VAR)
    if env:
        cands.append(Path(env))
    here = Path(start).resolve() if start else Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        cands.append(d / "gguf-py")
    for c in cands:
        if (c / "gguf" / "__init__.py").is_file():
            return c
    return None


def add_gguf_py(start=None):
    """Put the located gguf-py directory first on sys.path (idempotent). Returns the path or None."""
    p = gguf_py(start)
    if p is not None and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
