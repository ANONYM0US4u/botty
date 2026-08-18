from dataclasses import dataclass, field
from datetime import datetime, timedelta
import polars as pl

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_SESSION_BREAK = timedelta(hours=6)  # overnight/weekend = expected for NSE


@dataclass
class ValidationReport:
    checks: list[dict] = field(default_factory=list)

    def is_valid(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def failed_checks(self) -> list[str]:
        return [c["check"] for c in self.checks if not c["passed"]]


def _parse_ts(ts) -> datetime | None:
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.strptime(str(ts), _TS_FMT)
    except ValueError:
        return None


def _check_timeframe(bars: pl.DataFrame, interval_min: int,
                     session_breaks_allowed: bool) -> dict:
    if bars.height < 2:
        return {"check": "timeframe_exists", "passed": False,
                "detail": "need >= 2 bars to measure interval"}
    times = [_parse_ts(t) for t in bars["time"]]
    if any(t is None for t in times):
        return {"check": "timeframe_exists", "passed": False,
                "detail": "unparseable timestamps"}
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])
              if b > a]
    in_session = [d for d in deltas
                  if d <= _SESSION_BREAK.total_seconds()]
    if not in_session:
        return {"check": "timeframe_exists", "passed": False,
                "detail": "no intra-session deltas"}
    median = sorted(in_session)[len(in_session) // 2]
    target = interval_min * 60
    ok = abs(median - target) <= target * 0.5
    return {"check": "timeframe_exists", "passed": ok,
            "detail": f"median delta {median}s vs expected {target}s"}


def validate_dataset(bars: pl.DataFrame, symbol: str,
                     expected_interval_minutes: int,
                     date_range: tuple[str, str] | None = None,
                     corporate_actions: dict | None = None,
                     session_breaks_allowed: bool = True) -> ValidationReport:
    report = ValidationReport()

    if bars.height == 0:
        report.checks.append({"check": "range_exists", "passed": False,
                              "detail": "empty dataset"})
        report.checks.append({"check": "timeframe_exists", "passed": False,
                              "detail": "empty dataset"})
        report.checks.append({"check": "no_duplicate_timestamps",
                              "passed": False, "detail": "empty dataset"})
        report.checks.append({"check": "no_future_timestamps",
                              "passed": False, "detail": "empty dataset"})
        report.checks.append({"check": "no_unexpected_gaps",
                              "passed": False, "detail": "empty dataset"})
        report.checks.append({"check": "ohlc_valid", "passed": False,
                              "detail": "empty dataset"})
        report.checks.append({"check": "timezone_normalized",
                              "passed": False, "detail": "empty dataset"})
        report.checks.append({"check": "corporate_action_policy_applied",
                              "passed": True, "detail": "skipped"})
        return report

    times = [_parse_ts(t) for t in bars["time"]]
    parsed = [t for t in times if t is not None]

    if date_range is None:
        report.checks.append({"check": "range_exists", "passed": True,
                              "detail": "no range requested"})
    else:
        want_lo, want_hi = (datetime.strptime(date_range[0], _TS_FMT),
                            datetime.strptime(date_range[1], _TS_FMT))
        got_lo, got_hi = min(parsed), max(parsed)
        ok = got_lo <= want_lo and got_hi >= want_hi
        report.checks.append({"check": "range_exists", "passed": ok,
                              "detail": f"have {got_lo}..{got_hi} want {want_lo}..{want_hi}"})

    report.checks.append(_check_timeframe(bars, expected_interval_minutes,
                                          session_breaks_allowed))

    n = bars.height
    uniq = bars["time"].n_unique()
    report.checks.append({"check": "no_duplicate_timestamps", "passed": uniq == n,
                          "detail": f"{n} rows, {uniq} unique timestamps"})

    now = datetime.now()
    if parsed:
        ok_future = max(parsed) <= now + timedelta(days=1)
        report.checks.append({"check": "no_future_timestamps", "passed": ok_future,
                              "detail": f"max {max(parsed)}"})
    else:
        report.checks.append({"check": "no_future_timestamps", "passed": False,
                              "detail": "no parseable timestamps"})

    gap_ok = True
    gap_detail = "no unexpected gaps"
    if parsed and len(parsed) > 1:
        target = timedelta(minutes=expected_interval_minutes)
        for a, b in zip(parsed, parsed[1:]):
            delta = b - a
            if delta <= target * 2:
                continue
            if session_breaks_allowed and delta >= _SESSION_BREAK:
                continue
            gap_ok = False
            gap_detail = f"gap {delta} at {a} -> {b}"
            break
    report.checks.append({"check": "no_unexpected_gaps", "passed": gap_ok,
                          "detail": gap_detail})

    o = bars["open"].to_list()
    h = bars["high"].to_list()
    l = bars["low"].to_list()
    c = bars["close"].to_list()
    ohlc_ok = all(h[i] >= max(o[i], c[i]) and l[i] <= min(o[i], c[i])
                  and o[i] > 0 and h[i] > 0 and l[i] > 0 and c[i] > 0
                  for i in range(n))
    report.checks.append({"check": "ohlc_valid", "passed": ohlc_ok,
                          "detail": f"{n} bars checked"})

    tz_ok = all(t is not None for t in times)
    report.checks.append({"check": "timezone_normalized", "passed": tz_ok,
                          "detail": "all timestamps tz-naive" if tz_ok
                          else "timezone-aware or unparseable timestamp found"})

    if corporate_actions is None:
        report.checks.append({"check": "corporate_action_policy_applied",
                              "passed": True, "detail": "skipped"})
    else:
        applied = corporate_actions.get("applied", False)
        report.checks.append({"check": "corporate_action_policy_applied",
                              "passed": applied,
                              "detail": corporate_actions.get("note", "")})

    return report