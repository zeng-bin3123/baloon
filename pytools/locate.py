# port: p17652417b0  # canonical — one definition per TSV symbol
"""Locating an optional vendored gguf-py checkout (used only as an external oracle, never required)."""
import os
import sys
from pathlib import Path

ENV_VAR = "BALOON_GGUF_PY"
