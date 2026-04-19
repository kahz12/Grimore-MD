import ipaddress
import re
import socket
from urllib.parse import urlparse

from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Common patterns for sensitive information.
# Email/IPv4 are validated structurally rather than just shape-matched.
SENSITIVE_PATTERNS = {
    "api_key": r'(?i)\b(?:api[-_]?key|token|auth|password|secret|pwd)\s*[:=]\s*["\']?[A-Za-z0-9\-_./+=]{16,}["\']?',
    "email": r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}\b',
    "ipv4": r'\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b',
    "ssh_key": r'-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----',
    "aws_key": r'\bAKIA[0-9A-Z]{16}\b',
    "github_token": r'\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    "openai_key": r'\bsk-[A-Za-z0-9]{20,}\b',
    "jwt": r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b',
    "bearer": r'\bBearer\s+[A-Za-z0-9\-_.=]{20,}\b',
}

# Tokens / role markers that LLMs interpret as control structures.
# Case-insensitive substring replacement; we neutralize them by inserting
# a zero-width-ish separator so the original meaning survives for humans.
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
    def __init__(self, vault_path: str):
        self.vault_path = vault_path

    def scan_for_sensitive_data(self, text: str) -> list[str]:
        """
        Detects potential PII or credentials in text.
        Returns a list of detected categories.
        """
        detected = []
        for name, pattern in SENSITIVE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(name)
        return detected

    def sanitize_prompt(self, text: str) -> str:
        """
        Neutralize role markers and chat-template tokens that could be
        interpreted by the LLM as control structure (prompt injection).
        Replacement breaks the literal token while keeping it readable.
        """
        return _ROLE_MARKER_RE.sub(
            lambda m: m.group(0)[0] + "\u200b" + m.group(0)[1:],
            text,
        )

    @staticmethod
    def validate_llm_host(url: str, allow_remote: bool = False) -> str:
        """
        Validate an Ollama-compatible host URL.
        Rules:
          - scheme must be http or https
          - https required for non-loopback hosts
          - if allow_remote is False, host must resolve to a loopback address
        Returns the URL on success; raises ValueError otherwise.
        """
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

        return url

    @staticmethod
    def wrap_untrusted(text: str, label: str = "note") -> str:
        """
        Wrap untrusted content in a delimited block so the LLM treats it
        strictly as data. Strips any closing delimiter inside the payload.
        """
        close_tag = f"</{label}>"
        safe = text.replace(close_tag, close_tag.replace("<", "<\u200b"))
        return f"<{label}>\n{safe}\n</{label}>"
