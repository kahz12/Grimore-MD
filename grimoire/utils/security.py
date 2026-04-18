import re
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Common patterns for sensitive information
SENSITIVE_PATTERNS = {
    "api_key": r'(?:key|token|auth|password|secret|pwd)[-_a-z0-9]{0,20}[:=]\s*[a-z0-9\-_]{16,}',
    "email": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    "ipv4": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "ssh_key": r'-----BEGIN [A-Z ]+ PRIVATE KEY-----'
}

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
        Simple sanitization to prevent prompt injection 
        from note content.
        """
        # Escaping markers that LLMs use to separate instructions
        sanitized = text.replace("SYSTEM:", "S_YSTEM:")
        sanitized = sanitized.replace("USER:", "U_SER:")
        sanitized = sanitized.replace("ASSISTANT:", "A_SSISTANT:")
        return sanitized
