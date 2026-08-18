import asyncio
from engine.api.metrics_emitter import MetricsEmitter


def test_emit_json_sends_payload():
    received = []
    class FakeClient:
        async def send_text(self, text): received.append(text)
    em = MetricsEmitter()
    em.register(FakeClient())
    async def go():
        em.emit_json("probs", {"ts": "t", "probs": [0.1, 0.8, 0.1]})
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert received and '"probs"' in received[0]
    assert '"payload"' in received[0]