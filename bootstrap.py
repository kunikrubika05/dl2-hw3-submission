"""Notebook entry point: keep implementation helpers out of the user namespace."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "pipeline/tools.py"))["setup_runtime"](
    Path(__file__).resolve().parent,
)
