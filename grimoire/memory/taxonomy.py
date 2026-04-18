import re
import unicodedata

class Taxonomy:
    def __init__(self, controlled_vocabulary: list[str] = None):
        self.vocabulary = controlled_vocabulary or []
        self.norm_map = {self.normalize(tag): tag for tag in self.vocabulary}

    def normalize(self, tag: str) -> str:
        """
        Normalizes a tag: lowercase, no accents, replaces spaces with hyphens.
        Example: "Ocultismo Clásico" -> "ocultismo-clasico"
        """
        # Lowercase and strip
        tag = tag.lower().strip()
        # Remove accents
        tag = "".join(
            c for c in unicodedata.normalize('NFD', tag)
            if unicodedata.category(c) != 'Mn'
        )
        # Replace spaces and special chars with hyphens
        tag = re.sub(r'[^a-z0-9]+', '-', tag)
        # Remove leading/trailing hyphens
        tag = tag.strip('-')
        return tag

    def reconcile(self, raw_tags: list[str]) -> list[str]:
        """
        Reconciles raw tags from LLM against the controlled vocabulary.
        If a tag's normalized version matches one in the vocabulary, use the vocabulary version.
        """
        reconciled = []
        for raw in raw_tags:
            norm = self.normalize(raw)
            if norm in self.norm_map:
                reconciled.append(self.norm_map[norm])
            else:
                # If not in vocabulary, use the raw tag but maybe normalize it slightly
                # For now, we allow new tags but we should mark them as candidates
                reconciled.append(raw.strip())
        return list(set(reconciled))
