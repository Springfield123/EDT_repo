from __future__ import annotations
from pathlib import Path
from edt.paper import PAPER_ASSETS

print("EDT manuscript assets bundled in this repository:
")
for key, path in PAPER_ASSETS.items():
    print(f"- {key}: {path}")
