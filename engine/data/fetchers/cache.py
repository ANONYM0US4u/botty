import os
from pathlib import Path

import polars as pl


def cache_path(cfg: dict, symbol: str) -> Path:
    return (Path(cfg["storage"]["parquet_dir"]) / "cache" /
            f"{symbol}.parquet")


def load_cached(cfg: dict, symbol: str) -> pl.DataFrame | None:
    p = cache_path(cfg, symbol)
    if not p.exists():
        return None
    try:
        return pl.read_parquet(p)
    except Exception:
        return None


def save_cache(cfg: dict, symbol: str, df: pl.DataFrame) -> None:
    p = cache_path(cfg, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, p)