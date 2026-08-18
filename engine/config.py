from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config/config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_secret(name: str) -> str:
    load_dotenv(_ROOT / "config" / ".env")
    return os.environ.get(name, "")