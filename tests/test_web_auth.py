import pytest
from src.web.auth import GlobalLimiter, failure_delay, password_matches, password_required


def test_password_check():
    assert password_matches("correct horse", "correct horse")
    assert not password_matches("wrong", "correct horse") and not password_matches("", "x")


def test_hosted_deployment_requires_password(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False); monkeypatch.setenv("RENDER", "true")
    with pytest.raises(RuntimeError): password_required()
    monkeypatch.setenv("APP_PASSWORD", "pw"); assert password_required() == (True, "pw")
    monkeypatch.delenv("APP_PASSWORD"); monkeypatch.delenv("RENDER"); monkeypatch.delenv("REQUIRE_APP_PASSWORD", raising=False)
    assert password_required() == (False, None)


def test_global_limiter_and_backoff():
    now = [0.0]
    limiter = GlobalLimiter(max_failures=3, window_seconds=60, clock=lambda: now[0])
    for _ in range(3): limiter.record_failure()
    assert limiter.locked()
    now[0] = 61; assert not limiter.locked()
    assert [failure_delay(n) for n in (1, 2, 3, 10)] == [1, 2, 4, 30]
