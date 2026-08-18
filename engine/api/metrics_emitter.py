import asyncio
import json


class MetricsEmitter:
    def __init__(self):
        self._clients: set = set()

    def register(self, client) -> None:
        self._clients.add(client)

    def unregister(self, client) -> None:
        self._clients.discard(client)

    def emit(self, name: str, value: float) -> None:
        payload = f'{{"name": "{name}", "value": {value}}}'
        for c in list(self._clients):
            try:
                asyncio.create_task(c.send_text(payload))
            except Exception:
                self.unregister(c)

    def emit_json(self, name: str, payload: dict) -> None:
        text = json.dumps({"name": name, "payload": payload})
        for c in list(self._clients):
            try:
                asyncio.create_task(c.send_text(text))
            except Exception:
                self.unregister(c)