import pytest
from engine.brokers.dhan import DhanAdapter
from engine.config import get_secret


def test_requires_credentials():
    if not get_secret("DHAN_ACCESS_TOKEN"):
        pytest.skip("DHAN_ACCESS_TOKEN not set")
    a = DhanAdapter(get_secret("DHAN_CLIENT_ID"), get_secret("DHAN_ACCESS_TOKEN"))
    assert a.get_balance() >= 0