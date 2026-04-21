import re
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


class Taxonomy:
    def __init__(self, controlled_vocabulary: Optional[list[str]] = None):
        self.vocabulary: list[str] = list(controlled_vocabulary or [])
        self.norm_map: dict[str, str] = {
            self.normalize(tag): tag for tag in self.vocabulary
        }

    def normalize(self, tag: str) -> str:
        """
        Normalizes a tag: lowercase, no accents, non-alnum → hyphens.
        Example: "Ocultismo Clásico" → "ocultismo-clasico"
        """
        tag = tag.lower().strip()
        tag = "".join(
            c for c in unicodedata.normalize("NFD", tag)
            if unicodedata.category(c) != "Mn"
        )
        tag = re.sub(r"[^a-z0-9]+", "-", tag)
        return tag.strip("-")

    def reconcile(self, tags: list[str]) -> list[str]:
        """
        Dedupe (preserving first-seen order) and map each tag to its canonical
        vocabulary spelling when the normalized form is registered.
        Inputs are expected to have already been passed through ``normalize``.
        """
        seen: set[str] = set()
        out: list[str] = []
        for tag in tags:
            if not tag:
                continue
            canonical = self.norm_map.get(tag, tag)
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)
        return out


def load_taxonomy_from_vault(vault_path: Path) -> Taxonomy:
    """
    Load the controlled vocabulary from ``<vault>/taxonomy.yml``.

    Expected schema::

        vocabulary:
          - filosofia
          - ocultismo-clasico
          - nihilismo

    Unknown keys are ignored. Errors fall back to an empty taxonomy with a
    warning so a malformed file never blocks ingestion.
    """
    candidate = Path(vault_path) / "taxonomy.yml"
    if not candidate.exists():
        return Taxonomy()

    try:
        raw = candidate.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("taxonomy_load_failed", path=str(candidate), error=str(e))
        return Taxonomy()

    vocab = data.get("vocabulary") if isinstance(data, dict) else None
    if not isinstance(vocab, list):
        logger.warning("taxonomy_no_vocabulary", path=str(candidate))
        return Taxonomy()

    vocabulary = [str(x).strip() for x in vocab if isinstance(x, (str, int))]
    vocabulary = [v for v in vocabulary if v]
    logger.info("taxonomy_loaded", path=str(candidate), size=len(vocabulary))
    return Taxonomy(vocabulary)
