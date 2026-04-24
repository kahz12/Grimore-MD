"""
Security and Privacy Guard.
This module provides tools for detecting sensitive data (PII, credentials),
sanitizing prompts to prevent injection, and validating LLM host URLs.
"""
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Common patterns for sensitive information (API keys, secrets, PII)
SENSITIVE_PATTERNS = {
    "api_key": (
        r'\b(?:api[-_]?key|token|auth|password|secret|pwd)\s*[:=]\s*["\']?[A-Za-z0-9\-_./+=]{16,}["\']?',
        re.IGNORECASE,
    ),
    "email": (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}\b', re.IGNORECASE),
    "ipv4": (r'\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b', 0),
    "ssh_key": (r'-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----', 0),
    "aws_key": (r'\bAKIA[0-9A-Z]{16}\b', 0),
    "github_token": (r'\bgh[pousr]_[A-Za-z0-9]{20,}\b', 0),
    "openai_key": (r'\bsk-[A-Za-z0-9]{20,}\b', 0),
    "jwt": (r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b', 0),
    "bearer": (r'\bBearer\s+[A-Za-z0-9\-_.=]{20,}\b', re.IGNORECASE),
}

# Control tokens that could trigger prompt injection or confuse the LLM
ROLE_MARKERS = [
    "system:", "user:", "assistant:", "developer:",
    "sistema:", "usuario:", "asistente:",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|system|>",
    "<|user|>", "<|assistant|>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "### Instruction", "### Response", "### System",
]

_ROLE_MARKER_RE = re.compile(
    "|".join(re.escape(m) for m in ROLE_MARKERS),
    flags=re.IGNORECASE,
)


class SecurityGuard:
    """
    Implements various security checks and content sanitization strategies.
    """
    def __init__(self, vault_path: str):
        self.vault_path = vault_path

    def scan_for_sensitive_data(self, text: str) -> list[str]:
        """
        Detects potential PII or credentials in text.
        Returns a list of detected categories (e.g., ['api_key', 'email']).
        """
        detected = []
        for name, (pattern, flags) in SENSITIVE_PATTERNS.items():
            if re.search(pattern, text, flags):
                detected.append(name)
        return detected

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        """
        Neutralize role markers and chat-template tokens by inserting a 
        zero-width space (\u200b). This prevents the LLM from interpreting 
        user content as instructions (Prompt Injection).
        """
        return _ROLE_MARKER_RE.sub(
            lambda m: m.group(0)[0] + "\u200b" + m.group(0)[1:],
            text,
        )

    # Cache of (url, allow_remote) → validated url or ValueError. A single CLI
    # invocation spins up LLMRouter, Embedder and PreflightChecker; each called
    # validate_llm_host independently, paying three getaddrinfo round-trips
    # before. Memoising here collapses them to one resolution per process.
    _host_cache: dict[tuple[str, bool], str] = {}

    @staticmethod
    def validate_llm_host(url: str, allow_remote: bool = False) -> str:
        """
        Validates the Ollama host URL to prevent SSRF or unauthorized remote access.
        If allow_remote is False, only loopback addresses (localhost/127.0.0.1) are permitted.
        """
        cache_key = (url, bool(allow_remote))
        cached = SecurityGuard._host_cache.get(cache_key)
        if cached is not None:
            return cached

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"OLLAMA_HOST scheme must be http(s): {url!r}")
        if not parsed.hostname:
            raise ValueError(f"OLLAMA_HOST is missing a hostname: {url!r}")

        try:
            infos = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"OLLAMA_HOST does not resolve: {url!r}") from exc

        addrs = {ipaddress.ip_address(info[4][0]) for info in infos}
        all_loopback = all(addr.is_loopback for addr in addrs)

        if not all_loopback:
            if not allow_remote:
                raise ValueError(
                    f"OLLAMA_HOST {url!r} is non-loopback but cognition.allow_remote is false"
                )
            if parsed.scheme != "https":
                raise ValueError(f"Remote OLLAMA_HOST must use https: {url!r}")

        SecurityGuard._host_cache[cache_key] = url
        return url

    @staticmethod
    def wrap_untrusted(text: str, label: str = "note") -> str:
        """
        Wraps untrusted content in XML-like tags to further delimit data from instructions.
        Ensures that any existing opening *and* closing tags within the content
        are neutralized.

        Why both directions: if only the closing form were neutralised, an
        attacker could write ``<note>ignore above, do X</note>`` inside the
        body and \u2014 to the LLM scanning the context \u2014 it looks like an
        immediate re-opening of a new, trusted ``<note>`` block. Breaking the
        opening tag too forecloses that fake re-scope.
        """
        open_tag = f"<{label}>"
        close_tag = f"</{label}>"
        # Insert a zero-width space after the "<" so the LLM tokenizer sees a
        # distinct string, but a human reader still recognises the mark.
        safe = text.replace(close_tag, close_tag.replace("<", "<\u200b"))
        safe = safe.replace(open_tag, open_tag.replace("<", "<\u200b"))
        return f"<{label}>\n{safe}\n</{label}>"

    @staticmethod
    def resolve_within_vault(file_path, vault_root) -> Path:
        """
        Resolve ``file_path`` and require the result to live under ``vault_root``.

        Follows symlinks (so a symlink inside the vault pointing outside is
        rejected) and defends against ``..`` traversal. Raises ``ValueError``
        when the path escapes the vault.
        """
        vault = Path(vault_root).resolve(strict=False)
        candidate = Path(file_path).resolve(strict=False)
        try:
            candidate.relative_to(vault)
        except ValueError as exc:
            raise ValueError(
                f"path escapes vault: {file_path!r} not inside {vault}"
            ) from exc
        return candidate

    @staticmethod
    def redact_for_log(text: str, max_len: int = 160) -> str:
        """
        Make an arbitrary LLM/user string safe to drop into a log field:
        strip control chars, collapse whitespace, redact sensitive patterns,
        and truncate. Prevents note content (which can contain the very
        secrets we scan for) from leaking verbatim into logs.
        """
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        for _, (pattern, flags) in SENSITIVE_PATTERNS.items():
            cleaned = re.sub(pattern, "[REDACTED]", cleaned, flags=flags)
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 1] + "…"
        return cleaned
