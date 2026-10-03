from __future__ import annotations
import copy, os
from pathlib import Path
import pandas as pd
import yaml
from src.config import ROOT

UNIVERSE_PATH = ROOT / "config/tr_etf_universe.csv"
WEB_CONFIG_PATH = ROOT / "config/web.yaml"


def data_dir() -> Path:
    """Persistent location for the database and price cache (a Render disk in production)."""
    path = Path(os.getenv("INVEST_DATA_DIR", ROOT / "state"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_web_config(path: str | Path = WEB_CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_settings(base: dict, overrides: dict | None) -> dict:
    """Deep-merge saved user overrides onto the YAML defaults."""
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict): merged[key] = merge_settings(merged[key], value)
        else: merged[key] = value
    return merged


def load_universe(path: str | Path = UNIVERSE_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["distributing"] = frame.distributing.astype(bool)
    frame["tags"] = frame.tags.fillna("").map(lambda s: [t for t in s.split(";") if t])
    return frame
