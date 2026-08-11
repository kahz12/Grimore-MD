"""The circuit breaker stays coherent when several threads use one router.

The daemon indexes on its watcher thread while the CLI or the HTTP API reads,
and both go through the same LLMRouter, so its counters are shared.

What these tests do NOT do is prove the lock is load-bearing on this
interpreter. It is not: a GIL build lost no increments across 160,000
concurrent ``+= 1`` calls even with the switch interval at 1 ns, and every
test here passes with the lock removed. They pin the observable contract --
counts add up, the breaker opens once, a reset clears both fields together --
which is what a future free-threaded build would break first if the
synchronisation were dropped.
"""
import itertools
import threading

from unittest.mock import MagicMock

from grimore.cognition.llm_router import LLMRouter
from grimore.utils.config import CognitionConfig, Config, MemoryConfig, VaultConfig


def _router(threshold=5):
    cfg = Config(
        vault=VaultConfig(path="./v"),
        cognition=CognitionConfig(circuit_failure_threshold=threshold),
        memory=MemoryConfig(db_path=":memory:"),
    )
    router = LLMRouter(cfg)
    router.backend = MagicMock()
    return router


def _hammer(fn, threads=8, per_thread=50):
    barrier = threading.Barrier(threads)

    def run():
        barrier.wait(timeout=30)
        for _ in range(per_thread):
            fn()

    workers = [threading.Thread(target=run, daemon=True) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
        assert not w.is_alive(), "worker did not finish"


class TestBreakerUnderConcurrency:
    def test_every_failure_is_counted(self):
        """400 concurrent failures must count 400. Passes without the lock on
        a GIL build; it is the property a free-threaded one would violate."""
        router = _router(threshold=10_000)   # high, so it never opens and resets
        _hammer(router._record_failure, threads=8, per_thread=50)
        assert router._consecutive_failures == 400

    def test_the_breaker_opens_exactly_once(self):
        router = _router(threshold=5)
        opened = []
        real = router._record_failure

        def watched():
            before = router._open_until
            real()
            if router._open_until != before:
                opened.append(1)

        _hammer(watched, threads=8, per_thread=20)
        assert router._open_until > 0, "the breaker should be open"
        assert len(opened) == 1, f"opened {len(opened)} times, expected once"

    def test_concurrent_success_and_failure_leave_consistent_state(self):
        router = _router(threshold=5)

        # itertools.count is thread-safe on CPython, and a plain int in a
        # closure would need its own synchronisation just to drive the test.
        ticks = itertools.count()

        def mixed():
            if next(ticks) % 3:
                router._record_failure()
            else:
                router._record_success()

        _hammer(mixed, threads=6, per_thread=40)
        # Whatever the interleaving, the state must remain coherent: a
        # non-negative count, and an open_until that is either zero or a real
        # deadline -- never a partially written mix.
        assert router._consecutive_failures >= 0
        assert router._open_until >= 0.0

    def test_a_reset_clears_both_fields_together(self):
        router = _router(threshold=1)
        router._record_failure()
        assert router._open_until > 0
        router._record_success()
        assert router._consecutive_failures == 0
        assert router._open_until == 0.0
