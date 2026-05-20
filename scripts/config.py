"""
Load path configuration from config.env in this directory.

Usage in pipeline scripts:
    from config import BASE_DATA           # all scripts
    from config import require, BASE_DATA  # script 07: ESM_DIR = require("SAE_ESM_DIR")
"""

import os
import sys
from pathlib import Path

_cfg = Path(__file__).parent / "config.env"
if _cfg.exists():
    for line in _cfg.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        sys.exit(f"ERROR: {var} not set. Copy config.env.template → config.env and fill in your paths.")
    return val


BASE_DATA = require("SAE_BASE_DATA")
