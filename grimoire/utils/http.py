"""
HTTP Utilities.
This module provides a pre-configured requests session with automatic retries
and backoff logic, optimized for communicating with the local Ollama API.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(total_retries: int = 2, backoff: float = 0.5) -> requests.Session:
    """
    Creates a requests.Session with bounded retries on connection errors and 5xx status codes.
    Includes an exponential backoff factor to be polite to the backend service.
    
    Retries are enabled for both GET and POST requests, which is crucial for
    reliability when the daemon is waiting for Ollama to become available.
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
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    # Mount the adapter for both HTTP and HTTPS traffic
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
