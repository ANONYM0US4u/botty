import asyncio
import json


class MetricsEmitter:
    def __init__(self):
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def register(self, client) -> None:
        self._clients.add(client)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def unregister(self, client) -> None:
        self._clients.discard(client)

    def _send(self, client, text: str) -> None:
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                client.send_text(text), self._loop)
        except Exception:
            self.unregister(client)

    def emit(self, name: str, value: float) -> None:
        payload = f'{{"name": "{name}", "value": {value}}}'
        for c in list(self._clients):
            self._send(c, payload)

    def emit_json(self, name: str, payload: dict) -> None:
        text = json.dumps({"name": name, "payload": payload})
        for c in list(self._clients):
            self._send(c, text)