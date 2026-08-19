import sqlite3
from pathlib import Path
import polars as pl

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, price REAL,
  status TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS fills (
  order_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, ts TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
  ts TEXT PRIMARY KEY, equity REAL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  path TEXT PRIMARY KEY, reward REAL, sharpe REAL, ts TEXT,
  run_id TEXT, model_id TEXT, git_commit TEXT, config_hash TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  ts TEXT, symbol TEXT, action TEXT, probs TEXT, features TEXT, attribution TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  name TEXT, value REAL, ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts);
"""


class DataStore:
    def __init__(self, db_path: str | Path, parquet_dir: str | Path):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def save_bars(self, symbol: str, df: pl.DataFrame, interval_minutes: int) -> None:
        safe = symbol.replace("/", "_")
        dest = self.parquet_dir / f"{safe}_{interval_minutes}m.parquet"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        df.write_parquet(tmp)
        tmp.replace(dest)

    def get_bars(self, symbol: str, start: str | None = None,
                 end: str | None = None) -> pl.DataFrame:
        safe = symbol.replace("/", "_")
        files = sorted(self.parquet_dir.glob(f"{safe}_*m.parquet"))
        if not files:
            return pl.DataFrame()
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        if start:
            df = df.filter(pl.col("time") >= start)
        if end:
            df = df.filter(pl.col("time") <= end)
        return df.sort("time")

    def append_order(self, order: dict) -> None:
        rec = {**order, "ts": order.get("ts", "")}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders VALUES (:id,:symbol,:side,:qty,:price,:status,:ts)",
                rec)

    def append_fill(self, fill: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO fills VALUES (:order_id,:symbol,:side,:qty,:price,:ts)",
                fill)

    def append_equity(self, ts: str, equity: float) -> None:
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO equity_curve VALUES (?,?)", (ts, equity))

    def get_equity(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ts, equity FROM equity_curve ORDER BY ts").fetchall()
        return [{"ts": r[0], "equity": r[1]} for r in rows]

    def get_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM fills ORDER BY ts").fetchall()
        cols = ["order_id", "symbol", "side", "qty", "price", "ts"]
        return [dict(zip(cols, r)) for r in rows]

    def append_checkpoint(self, meta: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES "
                "(:path,:reward,:sharpe,:ts,:run_id,:model_id,:git_commit,:config_hash)",
                meta)

    def get_checkpoints(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY ts DESC").fetchall()
        cols = ["path", "reward", "sharpe", "ts", "run_id", "model_id",
                "git_commit", "config_hash"]
        return [dict(zip(cols, r)) for r in rows]

    def append_decision(self, rec: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO decisions VALUES (:ts,:symbol,:action,:probs,:features,:attribution)",
                rec)

    def get_decisions(self, symbol: str | None = None,
                      limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT ts, symbol, action, probs FROM decisions "
                    "WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                    (symbol, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts, symbol, action, probs FROM decisions "
                    "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "symbol": r[1], "action": r[2], "probs": r[3]}
                for r in rows]

    def append_metric(self, name: str, value: float, ts: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO metrics VALUES (?,?,?)", (name, value, ts))

    def get_metrics(self, name: str, since: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if since:
                rows = conn.execute(
                    "SELECT value, ts FROM metrics WHERE name=? AND ts>=? ORDER BY ts",
                    (name, since)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT value, ts FROM metrics WHERE name=? ORDER BY ts",
                    (name,)).fetchall()
        return [{"value": r[0], "ts": r[1]} for r in rows]