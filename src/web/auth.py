"""Single-user password gate for the hosted website.

Set APP_PASSWORD in the hosting environment. On Render the app refuses to run
without it, so a deployment can never be public by accident.
"""
from __future__ import annotations
import hashlib, hmac, os, time


def password_matches(candidate: str, expected: str) -> bool:
    digest = lambda v: hashlib.sha256(v.encode("utf-8")).digest()
    return hmac.compare_digest(digest(candidate or ""), digest(expected))


def password_required() -> tuple[bool, str | None]:
    """(login needed, configured password). Raises if a hosted deployment has no password."""
    expected = os.getenv("APP_PASSWORD")
    if expected: return True, expected
    if os.getenv("RENDER") or os.getenv("REQUIRE_APP_PASSWORD"):
        raise RuntimeError("APP_PASSWORD is not set. Add it in the Render dashboard (Environment) and redeploy.")
    return False, None


def failure_delay(failures: int) -> float:
    """Slows down password guessing: 1, 2, 4, 8 … seconds, capped at 30."""
    return float(min(30, 2 ** max(0, failures - 1)))


def throttle(failures: int) -> None:
    time.sleep(failure_delay(failures))


class GlobalLimiter:
    """Across all browser sessions: after too many wrong passwords, refuse logins for a while."""
    def __init__(self, max_failures: int = 20, window_seconds: float = 600, clock=time.monotonic):
        import threading
        self.max_failures = max_failures; self.window = window_seconds; self.clock = clock
        self.failures: list[float] = []; self._lock = threading.Lock()

    def _prune(self) -> None:
        cutoff = self.clock() - self.window
        self.failures = [t for t in self.failures if t > cutoff]

    def locked(self) -> bool:
        with self._lock:
            self._prune(); return len(self.failures) >= self.max_failures

    def record_failure(self) -> None:
        with self._lock: self.failures.append(self.clock()); self._prune()


LIMITER = GlobalLimiter()
