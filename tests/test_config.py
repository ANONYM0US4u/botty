from engine.config import load_config, get_secret


def test_load_config_defaults():
    cfg = load_config()
    assert cfg["market"]["timeframe_minutes"] == 5
    assert cfg["risk"]["daily_loss_limit_pct"] == -3.0
    assert cfg["instruments"]["stocks"] == ["RELIANCE.NS"]
    assert cfg["instruments"]["crypto"] == ["BTCUSDT"]


def test_get_secret_missing_returns_empty():
    assert get_secret("NONEXISTENT_KEY_XYZ") == ""