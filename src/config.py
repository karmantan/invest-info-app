from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict:
    chosen = Path(path or os.getenv("INVEST_APP_CONFIG", ROOT / "config/default.yaml"))
    with chosen.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

