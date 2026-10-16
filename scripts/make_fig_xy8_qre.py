#!/usr/bin/env python3
"""Compatibility entry point for the canonical paper-side XY8 workflow.

New code should invoke ``code/plotting/make_fig_xy8_qre.py`` directly.  This
wrapper remains so older commands cannot silently execute the retired coarse
grid implementation.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCRIPT = ROOT / "code" / "plotting" / "make_fig_xy8_qre.py"


if __name__ == "__main__":
    runpy.run_path(str(CANONICAL_SCRIPT), run_name="__main__")
