from engine.config import load_config


def main() -> None:
    cfg = load_config()
    print(f"Trading bot engine ready. Instruments: {cfg['instruments']}")


if __name__ == "__main__":
    main()