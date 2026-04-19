import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(total_retries: int = 2, backoff: float = 0.5) -> requests.Session:
    """
    A requests.Session with bounded retries on connection errors and 5xx,
    plus a small exponential backoff. Idempotent for POST so the daemon
    survives Ollama restarts without spinning forever.
    """
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
