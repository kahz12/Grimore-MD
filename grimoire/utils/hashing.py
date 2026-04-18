import hashlib
import re

def calculate_content_hash(text: str) -> str:
    """
    Calculates a hash of the clean body text.
    Normalizes whitespace to ensure same content results in same hash.
    """
    # Remove all whitespace and normalize for hashing
    clean_text = "".join(text.split())
    return hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
