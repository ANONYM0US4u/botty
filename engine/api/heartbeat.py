import threading
import time

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_TIMEOUT_S = 180.0
CLOSE_GRACE_S = 15.0


class Heartbeat:
    """Tracks whether a dashboard page is open.

    Armed once the first heartbeat arrives (bare backend runs never arm it).
    A closing=1 ping (page unload beacon) triggers shutdown after CLOSE_GRACE_S
    unless a heartbeat cancels it (page reload / back navigation).
    Falling silent for HEARTBEAT_TIMEOUT_S also triggers shutdown.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_seen: float | None = None
        self.closing_at: float | None = None

    def touch(self, now: float, closing: bool = False) -> None:
        with self._lock:
            if closing:
                if self.last_seen is not None:
                    self.closing_at = now
            else:
                self.last_seen = now
                self.closing_at = None

    def decide(self, now: float) -> str:
        with self._lock:
            if self.last_seen is None:
                return "alive"
            if self.closing_at is not None:
                return ("shutdown" if now > self.closing_at + CLOSE_GRACE_S
                        else "closing")
            if now - self.last_seen > HEARTBEAT_TIMEOUT_S:
                return "shutdown"
            return "alive"

    def reset(self) -> None:
        with self._lock:
            self.last_seen = None
            self.closing_at = None


def run_watchdog(hb: Heartbeat, on_shutdown) -> None:
    while True:
        time.sleep(2.0)
        if hb.decide(time.time()) == "shutdown":
            on_shutdown()
            return