"""I1 regression: build_session pins loopback HTTP backends to the validated
IP and sends the original hostname in the Host header."""
import pytest
import requests
from requests.adapters import HTTPAdapter

from grimore.utils.http import build_session, _LoopbackPinAdapter


def _prep(url, method="GET"):
    return requests.Request(method, url).prepare()


def test_build_session_without_pins_uses_plain_adapter():
    session = build_session()
    assert not isinstance(session.adapters["http://"], _LoopbackPinAdapter)
    assert not isinstance(session.adapters["https://"], _LoopbackPinAdapter)


def test_build_session_with_empty_pins_uses_plain_adapter():
    # loopback_pins returns {} when pinning doesn't apply; that must be a no-op.
    session = build_session(pins={})
    assert not isinstance(session.adapters["http://"], _LoopbackPinAdapter)


def test_build_session_with_pins_mounts_pin_adapter():
    session = build_session(pins={"localhost": ["127.0.0.1"]})
    assert isinstance(session.adapters["http://"], _LoopbackPinAdapter)
    # HTTPS is never pinned (would break SNI / certificate validation).
    assert not isinstance(session.adapters["https://"], _LoopbackPinAdapter)


def test_pin_adapter_rewrites_to_ip_and_preserves_host(monkeypatch):
    captured = {}

    def fake_send(self, request, **kwargs):
        captured["url"] = request.url
        captured["host"] = request.headers.get("Host")
        return "RESP"

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    adapter = _LoopbackPinAdapter({"localhost": ["127.0.0.1"]})
    resp = adapter.send(_prep("http://localhost:11434/api/tags"))

    assert resp == "RESP"
    assert captured["url"] == "http://127.0.0.1:11434/api/tags"
    assert captured["host"] == "localhost:11434"


def test_pin_adapter_brackets_ipv6(monkeypatch):
    captured = {}

    def fake_send(self, request, **kwargs):
        captured["url"] = request.url
        return "RESP"

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    adapter = _LoopbackPinAdapter({"localhost": ["::1"]})
    adapter.send(_prep("http://localhost:11434/api/tags"))

    assert captured["url"] == "http://[::1]:11434/api/tags"


def test_pin_adapter_passes_through_unpinned_host(monkeypatch):
    captured = {}

    def fake_send(self, request, **kwargs):
        captured["url"] = request.url
        return "RESP"

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    adapter = _LoopbackPinAdapter({"localhost": ["127.0.0.1"]})
    adapter.send(_prep("http://other.host:11434/x"))

    # Not in the pin map → URL is left untouched.
    assert captured["url"] == "http://other.host:11434/x"


def test_pin_adapter_falls_back_across_ips(monkeypatch):
    seen = []

    def fake_send(self, request, **kwargs):
        seen.append(request.url)
        if "127.0.0.1" in request.url:
            raise requests.exceptions.ConnectionError("refused")
        return "OK"

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    adapter = _LoopbackPinAdapter({"localhost": ["127.0.0.1", "::1"]})
    resp = adapter.send(_prep("http://localhost:11434/api/tags"))

    assert resp == "OK"
    assert seen == [
        "http://127.0.0.1:11434/api/tags",
        "http://[::1]:11434/api/tags",
    ]


def test_pin_adapter_raises_last_error_when_all_fail(monkeypatch):
    def fake_send(self, request, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    adapter = _LoopbackPinAdapter({"localhost": ["127.0.0.1", "::1"]})

    with pytest.raises(requests.exceptions.ConnectionError):
        adapter.send(_prep("http://localhost:11434/api/tags"))


class TestRetryBudgets:
    """A read timeout on an inference call means "still generating", not
    "transient failure". Retrying it multiplies the wall cost of every
    over-budget call, so the LLM backends opt out of read retries while
    keeping the connect ones (the daemon may start before Ollama is up).
    """

    @staticmethod
    def _retry(session, scheme="http://"):
        return session.adapters[scheme].max_retries

    def test_read_retries_default_to_total(self):
        # Callers that don't care keep the historical behaviour.
        retry = self._retry(build_session(total_retries=2))
        assert retry.read == 2
        assert retry.connect == 2

    def test_read_retries_can_be_disabled_independently(self):
        retry = self._retry(build_session(total_retries=2, read_retries=0))
        assert retry.read == 0
        # Connect retries survive: they're what lets the daemon wait for Ollama.
        assert retry.connect == 2

    def test_total_does_not_mask_a_larger_read_budget(self):
        # urllib3 caps every retry kind at `total`, so a read budget above
        # total_retries would be silently truncated.
        retry = self._retry(build_session(total_retries=1, read_retries=4))
        assert retry.total >= 4

    def test_read_retries_applied_to_both_schemes(self):
        session = build_session(read_retries=0)
        assert self._retry(session, "http://").read == 0
        assert self._retry(session, "https://").read == 0

    def test_pin_adapter_also_carries_the_retry_budget(self):
        session = build_session(pins={"localhost": ["127.0.0.1"]}, read_retries=0)
        assert isinstance(session.adapters["http://"], _LoopbackPinAdapter)
        assert self._retry(session, "http://").read == 0
