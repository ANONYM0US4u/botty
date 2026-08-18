import pytest
from engine.brokers.ccxt_adapter import CcxtAdapter
from engine.config import get_secret


def test_requires_credentials():
    if not get_secret("GATE_TESTNET_API_KEY"):
        pytest.skip("GATE_TESTNET_API_KEY not set")
    a = CcxtAdapter("gate", get_secret("GATE_TESTNET_API_KEY"),
                    get_secret("GATE_TESTNET_API_SECRET"))
    assert a.get_balance() >= 0


def test_invalid_exchange_raises():
    with pytest.raises(ValueError):
        CcxtAdapter("not_an_exchange", "k", "s")