import asyncio
import threading
from engine.api.metrics_emitter import MetricsEmitter


def test_emit_json_sends_payload():
    received = []
    class FakeClient:
        async def send_text(self, text): received.append(text)
    em = MetricsEmitter()
    async def go():
        em.register(FakeClient())
        em.emit_json("probs", {"ts": "t", "probs": [0.1, 0.8, 0.1]})
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert received and '"probs"' in received[0]
    assert '"payload"' in received[0]


def test_emit_json_from_worker_thread_keeps_client():
    received = []
    class FakeClient:
        async def send_text(self, text): received.append(text)
    em = MetricsEmitter()
    async def go():
        em.register(FakeClient())
        ev = threading.Event()
        def worker():
            em.emit_json("probs", {"ts": "t", "probs": [0.5, 0.3, 0.2]})
            ev.set()
        t = threading.Thread(target=worker)
        t.start()
        ev.wait()
        await asyncio.sleep(0.1)
        assert received, "worker-thread emit must reach the client"
        em.emit_json("traits", {"trades": 1})
        await asyncio.sleep(0.05)
        assert len(received) == 2, "client must stay registered"
    asyncio.run(go())