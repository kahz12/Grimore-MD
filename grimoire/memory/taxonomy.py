"""
Tag Taxonomy, Category Tree and vault-level taxonomy loading.

This module exposes two controlled-vocabulary concepts:

* ``Taxonomy``     — flat list of canonical tags (cross-cutting concepts).
* ``CategoryTree`` — hierarchical organisation for notes (Historia · Ciencia · …).

Both live in ``<vault>/taxonomy.yml``; see :func:`load_taxonomy_from_vault`.
"""
from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

from grimoire.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Flat tag vocabulary ────────────────────────────────────────────────────


class Taxonomy:
    """Flat controlled vocabulary for tags. Normalisation + canonical lookup."""

    def __init__(self, controlled_vocabulary: Optional[list[str]] = None):
        self.vocabulary: list[str] = list(controlled_vocabulary or [])
        self.norm_map: dict[str, str] = {
            self.normalize(tag): tag for tag in self.vocabulary
        }

    @staticmethod
    def normalize(tag: str) -> str:
        """Lowercase, strip accents, collapse non-alnum into hyphens."""
        tag = tag.lower().strip()
        tag = "".join(
            c for c in unicodedata.normalize("NFD", tag)
            if unicodedata.category(c) != "Mn"
        )
        tag = re.sub(r"[^a-z0-9]+", "-", tag)
        return tag.strip("-")

    def reconcile(self, tags: list[str]) -> list[str]:
        """Dedupe (preserving first-seen order) and map to canonical spelling."""
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


# ─── Hierarchical category tree ─────────────────────────────────────────────


DEFAULT_ROOTS: tuple[str, ...] = (
    "Historia",
    "Ciencia",
    "Tecnología",
    "Matemáticas",
    "Arte",
)


class CategoryTree:
    """
    Hierarchical taxonomy of categories and subcategories.

    Internally we store the tree as ``dict[canonical_path, list[canonical_path]]``
    mapping each node (``""`` = root) to the *ordered* list of its direct
    children, keyed by the canonical display path (``"Ciencia/Física"``).

    Lookup by user/LLM input (potentially un-accented, lowercase, slash-form)
    goes through :py:meth:`resolve`, which normalises each segment.
    """

    SEP = "/"

    def __init__(self, children: Optional[dict[str, list[str]]] = None):
        self._children: "OrderedDict[str, list[str]]" = OrderedDict()
        if children:
            for path, kids in children.items():
                self._children[path] = list(kids)
        if "" not in self._children:
            self._children[""] = []

    # ─── Construction ────────────────────────────────────────────────────────

    @classmethod
    def with_defaults(cls) -> "CategoryTree":
        tree = cls()
        for root in DEFAULT_ROOTS:
            tree.add(root)
        return tree

    @classmethod
    def from_raw(cls, raw) -> "CategoryTree":
        """
        Parse the ``categories:`` section of taxonomy.yml. Accepts:

        * ``None`` / missing → empty tree.
        * dict: keys are categories; values are either ``None``, ``[]``, a list
          of leaf strings, or a nested dict following the same schema.
        * list of strings: flat root categories with no subcategories.
        """
        tree = cls()
        tree._ingest(raw, parent="")
        return tree

    def _ingest(self, raw, parent: str) -> None:
        if raw is None:
            return
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    self.add(self._join(parent, item.strip()))
                elif isinstance(item, dict):
                    # a single-key dict: {"Física": [...]}
                    for name, children in item.items():
                        full = self._join(parent, str(name).strip())
                        if full.strip(self.SEP):
                            self.add(full)
                            self._ingest(children, full)
            return
        if isinstance(raw, dict):
            for name, children in raw.items():
                full = self._join(parent, str(name).strip())
                if not full.strip(self.SEP):
                    continue
                self.add(full)
                self._ingest(children, full)
            return
        logger.warning("category_entry_unrecognised", value=repr(raw))

    # ─── Queries ─────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return not self._children.get("", [])

    def roots(self) -> list[str]:
        return list(self._children.get("", []))

    def children(self, path: str = "") -> list[str]:
        return list(self._children.get(path, []))

    def has(self, path: str) -> bool:
        return path in self._children

    def paths(self) -> list[str]:
        """Pre-order traversal of every node (excluding the virtual root)."""
        out: list[str] = []

        def walk(node: str) -> None:
            for child in self._children.get(node, []):
                out.append(child)
                walk(child)

        walk("")
        return out

    def resolve(self, raw_path: str) -> Optional[str]:
        """
        Map an arbitrary user/LLM input to a canonical path in the tree.

        Each segment is normalised (lowercase, unaccented) before matching,
        so ``'CIENCIA / fisica'`` resolves to ``'Ciencia/Física'``.
        Returns ``None`` if any segment fails to match.
        """
        if not isinstance(raw_path, str):
            return None
        raw_path = raw_path.strip().strip(self.SEP)
        if not raw_path:
            return None

        node = ""
        canonical_parts: list[str] = []
        for segment in raw_path.split(self.SEP):
            normalised = Taxonomy.normalize(segment)
            if not normalised:
                return None
            match = None
            for child_path in self._children.get(node, []):
                child_name = child_path.rsplit(self.SEP, 1)[-1]
                if Taxonomy.normalize(child_name) == normalised:
                    match = child_path
                    break
            if match is None:
                return None
            canonical_parts.append(match.rsplit(self.SEP, 1)[-1])
            node = match
        return self.SEP.join(canonical_parts)

    # ─── Mutations ───────────────────────────────────────────────────────────

    def add(self, path: str) -> bool:
        """
        Create ``path`` (plus any missing ancestors). Returns True if
        anything new was created, False if the path already existed.
        """
        path = path.strip().strip(self.SEP)
        if not path:
            raise ValueError("category path cannot be empty")

        created = False
        parent = ""
        walked: list[str] = []
        for segment in path.split(self.SEP):
            segment = segment.strip()
            if not segment:
                raise ValueError(f"empty segment in category path: {path!r}")
            walked.append(segment)
            full = self.SEP.join(walked)
            if full not in self._children:
                self._children[full] = []
                self._children.setdefault(parent, [])
                if full not in self._children[parent]:
                    self._children[parent].append(full)
                created = True
            parent = full
        return created

    def remove(self, path: str) -> bool:
        """Remove ``path`` and its whole subtree. Returns True if it existed."""
        path = path.strip().strip(self.SEP)
        if not path or path not in self._children:
            return False

        # Remove node + descendants
        to_drop = [p for p in self._children if p == path or p.startswith(path + self.SEP)]
        for node in to_drop:
            self._children.pop(node, None)

        # Unlink from parent's children list
        parent = path.rsplit(self.SEP, 1)[0] if self.SEP in path else ""
        siblings = self._children.get(parent)
        if siblings and path in siblings:
            siblings.remove(path)
        return True

    # ─── Serialisation ───────────────────────────────────────────────────────

    def to_yaml_dict(self):
        """Serialise back to the nested dict form used in taxonomy.yml."""

        def build(node: str):
            kids = self._children.get(node, [])
            if not kids:
                return []
            out = {}
            for k in kids:
                name = k.rsplit(self.SEP, 1)[-1]
                out[name] = build(k)
            return out

        return build("")

    # ─── Utilities ───────────────────────────────────────────────────────────

    @classmethod
    def _join(cls, parent: str, name: str) -> str:
        if not parent:
            return name
        return f"{parent}{cls.SEP}{name}"


# ─── Combined vault taxonomy loader ─────────────────────────────────────────


@dataclass
class VaultTaxonomy:
    """Bundle of the flat tag vocabulary and the hierarchical category tree."""

    tags: Taxonomy = field(default_factory=Taxonomy)
    categories: CategoryTree = field(default_factory=CategoryTree.with_defaults)


def load_taxonomy_from_vault(vault_path: Path) -> VaultTaxonomy:
    """
    Load ``<vault>/taxonomy.yml``. Expected schema::

        vocabulary:
          - philosophy
          - nihilism

        categories:
          History:
            - Ancient
            - Modern
          Science:
            Physics:
              - Quantum

    A malformed or missing file falls back to ``VaultTaxonomy`` defaults so
    ingestion is never blocked. When no ``categories:`` key is present, the
    tree is seeded with the built-in defaults (History · Science · …).
    """
    candidate = Path(vault_path) / "taxonomy.yml"
    if not candidate.exists():
        return VaultTaxonomy()

    try:
        raw = candidate.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("taxonomy_load_failed", path=str(candidate), error=str(e))
        return VaultTaxonomy()

    if not isinstance(data, dict):
        logger.warning("taxonomy_schema_invalid", path=str(candidate))
        return VaultTaxonomy()

    # Tags
    vocab_raw = data.get("vocabulary")
    if isinstance(vocab_raw, list):
        vocabulary = [str(x).strip() for x in vocab_raw if isinstance(x, (str, int))]
        vocabulary = [v for v in vocabulary if v]
    else:
        vocabulary = []
    tags = Taxonomy(vocabulary)

    # Categories — if the key is present (even if empty) respect the user's
    # explicit wish; only auto-seed defaults when the key is absent.
    if "categories" in data:
        categories = CategoryTree.from_raw(data.get("categories"))
        if categories.is_empty():
            logger.info("taxonomy_categories_empty", path=str(candidate))
    else:
        categories = CategoryTree.with_defaults()

    logger.info(
        "taxonomy_loaded",
        path=str(candidate),
        vocab=len(vocabulary),
        categories=len(categories.paths()),
    )
    return VaultTaxonomy(tags=tags, categories=categories)


def save_taxonomy_to_vault(vault_path: Path, taxonomy: VaultTaxonomy) -> Path:
    """
    Persist the current ``VaultTaxonomy`` back to ``<vault>/taxonomy.yml``.

    Overwrites the file. Uses an atomic write so a crash never leaves a
    half-serialised document behind.
    """
    from grimoire.utils.atomic import atomic_write

    target = Path(vault_path) / "taxonomy.yml"
    payload = {
        "vocabulary": list(taxonomy.tags.vocabulary),
        "categories": taxonomy.categories.to_yaml_dict(),
    }
    body = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    atomic_write(target, lambda fh: fh.write(body.encode("utf-8")), mode="wb")
    logger.info("taxonomy_saved", path=str(target))
    return target
arget
