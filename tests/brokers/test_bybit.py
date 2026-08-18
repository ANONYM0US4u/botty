import pytest
from engine.brokers.bybit import BybitAdapter
from engine.config import get_secret


def test_requires_credentials():
    if not get_secret("BYBIT_TESTNET_API_KEY"):
        pytest.skip("BYBIT_TESTNET_API_KEY not set")
    a = BybitAdapter(get_secret("BYBIT_TESTNET_API_KEY"), get_secret("BYBIT_TESTNET_API_SECRET"))
    assert a.get_balance() >= 0