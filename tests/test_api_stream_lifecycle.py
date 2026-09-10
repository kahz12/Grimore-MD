import asyncio
import threading

import pytest

pytest.importorskip("starlette")

from grimore.api.app import _iterate_blocking  # noqa: E402


def test_closing_stream_stops_producer_and_closes_iterator():
    produced = []
    closed = threading.Event()

    def source():
        try:
            for i in range(1000):
                produced.append(i)
                yield i
        finally:
            closed.set()

    async def scenario():
        stream = _iterate_blocking(source())
        assert await anext(stream) == 0
        # No further request: the producer must not run ahead of the client.
        await asyncio.sleep(.02)
        assert produced == [0]
        await stream.aclose()
        async def wait_closed():
            while not closed.is_set():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait_closed(), timeout=2)
        assert produced == [0]

    asyncio.run(scenario())


def test_cancellation_during_next_does_not_advance_again(monkeypatch):
    import grimore.api.app as api
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api, "_API_WORKER_SLOTS", slots)
    entered, release, closed = (threading.Event() for _ in range(3))
    advanced = []

    def source():
        try:
            entered.set()
            release.wait(2)
            yield 0
            advanced.append(1)
            yield 1
        finally:
            closed.set()

    async def scenario():
        stream = _iterate_blocking(source())
        request = asyncio.create_task(anext(stream))
        while not entered.is_set():
            await asyncio.sleep(.001)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        async def wait_closed():
            while not closed.is_set():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait_closed(), timeout=2)
        assert advanced == []
        assert slots.acquire(blocking=False)
        slots.release()

    asyncio.run(scenario())
