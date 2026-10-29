"""Paths and business settings. Analyses read config/analysis.yaml, never config/world.yaml."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
# a held-out run points these somewhere else, one directory per world
BUILD = Path(os.environ.get("CNDL_BUILD", ROOT / "build"))
DATA = BUILD / "data"
TRUTH = BUILD / "truth"
WAREHOUSE = BUILD / "warehouse.duckdb"
RESULTS = BUILD / "results"
REPORTS = Path(os.environ.get("CNDL_REPORTS", ROOT / "reports"))


@cache
def analysis() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "analysis.yaml").read_text())


def contracts() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "contracts.yaml").read_text())
