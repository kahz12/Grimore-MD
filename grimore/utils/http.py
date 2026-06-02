"""
HTTP Utilities.
This module provides a pre-configured requests session with automatic retries
and backoff logic, optimized for communicating with the local Ollama API.
"""
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class _LoopbackPinAdapter(HTTPAdapter):
    """Connect to a pre-validated loopback IP instead of re-resolving the
    hostname, sending the original host in the ``Host`` header.

    This closes the DNS-rebinding check-vs-use gap (audit I1) for plain-HTTP
    loopback backends: :func:`SecurityGuard.validate_llm_host` resolves and
    approves the host, and the connection then goes straight to the address it
    approved rather than letting the HTTP client re-resolve the name. HTTPS is
    never pinned (it would break SNI / certificate validation), so this adapter
    is only ever mounted on ``http://``.

    The pin map is ``{hostname: [ip, ...]}``; addresses are tried in order so a
    dual-stack loopback host still falls back across families.
    """

    def __init__(self, pins: dict[str, list[str]], *args, **kwargs):
        self._pins = pins
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        parsed = urlparse(request.url)
        ips = self._pins.get(parsed.hostname)
        if not ips:
            return super().send(request, **kwargs)
        # Preserve the original host:port for the Host header so the server
        # still sees the name it expects, even though we dial an IP.
        original_host = parsed.netloc
        last_exc = None
        for ip in ips:
            netloc = f"[{ip}]" if ":" in ip else ip
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            request.url = urlunparse(parsed._replace(netloc=netloc))
            request.headers["Host"] = original_host
            try:
                return super().send(request, **kwargs)
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                continue
        raise last_exc


def build_session(
    total_retries: int = 2,
    backoff: float = 0.5,
    pins: dict[str, list[str]] | None = None,
) -> requests.Session:
    """
    Creates a requests.Session with bounded retries on connection errors and 5xx status codes.
    Includes an exponential backoff factor to be polite to the backend service.

    Retries are enabled for both GET and POST requests, which is crucial for
    reliability when the daemon is waiting for Ollama to become available.

    ``pins`` (a ``{hostname: [ip, ...]}`` map, typically from
    :func:`SecurityGuard.loopback_pins`) pins HTTP requests to those hostnames
    to the given pre-validated loopback IPs — closing the DNS-rebinding TOCTOU
    window (audit I1). When falsy, the session behaves exactly as before.
    """
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        # Retry on common transient errors
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    session = requests.Session()
    # Mount the adapter for both HTTP and HTTPS traffic. When pins are present,
    # HTTP traffic goes through the loopback-pinning adapter; HTTPS is never
    # pinned (SNI/cert), so it always uses the plain adapter.
    if pins:
        session.mount("http://", _LoopbackPinAdapter(pins, max_retries=retry))
    else:
        session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
