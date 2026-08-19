"""Pre-warm the parquet bar cache for configured instruments.

Run this once on a machine with network access so the bot can fall back
to cached bars when yfinance/Gate are unavailable:

    .venv\\Scripts\\python.exe scripts\\fetch_data.py [--symbol SYMBOL ...]
"""

import sys

from engine.config import load_config
from engine.data.fetchers.cache import save_cache
from scripts.run_live import make_fetch_bars


def main() -> None:
    cfg = load_config()
    fetch = make_fetch_bars(cfg)
    symbols = sys.argv[1:]
    if not symbols:
        symbols = list(cfg["instruments"]["stocks"]) + \
            list(cfg["instruments"]["crypto"])
    for sym in symbols:
        try:
            df = fetch(sym)
            save_cache(cfg, sym, df)
            print(f"{sym}: {df.height} bars cached")
        except Exception as e:
            print(f"{sym}: FAILED ({e})")


if __name__ == "__main__":
    main()