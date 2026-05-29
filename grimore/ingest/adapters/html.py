"""
HTML adapter — BeautifulSoup over the stdlib ``html.parser`` so the
adapter has no compiled dependency. Users who want faster parsing can
install ``lxml`` (declared as the ``html-fast`` extra) and bs4 will
auto-prefer it; the code is unchanged either way.

Extraction rules:

* Strip ``<script>``, ``<style>``, ``<noscript>`` and pure-navigation
  scaffolding (``<nav>``, ``<header>``, ``<footer>``) so the body text
  doesn't pollute the embedding with menu/footer noise.
* Prefer ``<main>`` or the first ``<article>`` as the content root when
  present; fall back to ``<body>`` and finally the whole document.
* Title fallback: ``<title>`` → first ``<h1>`` → filename stem.
* Sections are seeded at each ``<h1>``-``<h6>`` boundary and carry the
  heading text in ``ExtractedSection.heading``. A future citation layer
  will surface this as ``[[Doc#Heading]]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional, Union

from bs4 import BeautifulSoup, Tag

from grimore.ingest.adapters.base import (
    AdapterOptions,
    ExtractedDocument,
    ExtractedSection,
)
from grimore.ingest.adapters.registry import register
from grimore.utils.hashing import calculate_content_hash, sha256_file
from grimore.utils.logger import get_logger
from grimore.utils.security import SecurityGuard

logger = get_logger(__name__)

_MAX_HTML_BYTES = 10_000_000

_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
# Leaf text containers we lift body text from. A wrapping container
# (e.g. <blockquote><p>…</p></blockquote>) defers to its child to avoid
# double-counting the same text once for each enclosing container.
_LEAF_TEXT_CONTAINERS = ("p", "li", "blockquote", "pre", "td", "th")


def _pick_parser() -> str:
    """Use lxml when available (faster, more lenient), else stdlib.

    bs4 raises ``FeatureNotFound`` rather than degrading silently if we
    request a parser it can't find, so we probe once at import time.
    """
    try:
        BeautifulSoup("<x/>", "lxml")
        return "lxml"
    except Exception:
        return "html.parser"


_PARSER = _pick_parser()


def _select_content_root(soup: BeautifulSoup) -> Tag:
    """Pick the most signal-rich subtree as the document body.

    Order: <main> → <article> → <body> → whole document. Avoids the
    navigation/sidebar noise common in CMS-generated pages.
    """
    for selector in ("main", "article"):
        node = soup.find(selector)
        if isinstance(node, Tag):
            return node
    body = soup.find("body")
    if isinstance(body, Tag):
        return body
    return soup


def _extract_title(soup: BeautifulSoup, root: Tag, fallback: str) -> str:
    """``<title>`` → first ``<h1>`` → filename stem."""
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    h1 = root.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return fallback


def _sectionise(root: Tag) -> list[ExtractedSection]:
    """Walk top-level descendants, breaking on heading boundaries.

    The current heading determines the section a paragraph belongs to;
    text before the first heading is dropped into a leading anonymous
    section so cover paragraphs aren't lost.
    """
    sections: list[ExtractedSection] = []
    current_heading: Optional[str] = None
    current_text: list[str] = []
    order = 0
    # ``id(node)`` of every tag whose text has already been folded in
    # via an enclosing leaf container. Prevents both the
    # <blockquote><p>x</p></blockquote> double-count and the
    # <ul><li>top<ul><li>nested</li></ul></li></ul> drop — the outer
    # leaf grabs the whole subtree once, descendants stay quiet.
    visited: set[int] = set()

    def flush() -> None:
        nonlocal order
        body = "\n\n".join(line for line in (s.strip() for s in current_text) if line)
        if body:
            sections.append(ExtractedSection(
                text=body, heading=current_heading, order=order,
            ))
            order += 1
        current_text.clear()

    for child in root.descendants:
        if not isinstance(child, Tag):
            continue
        if id(child) in visited:
            continue
        name = child.name.lower() if child.name else ""
        if name in _HEADING_TAGS:
            flush()
            current_heading = child.get_text(strip=True) or None
            continue
        if name in _LEAF_TEXT_CONTAINERS:
            text = child.get_text(" ", strip=True)
            if text:
                current_text.append(text)
            # Mark the whole subtree as consumed so nested leaf
            # containers (e.g. <ul> inside <li>) don't re-emit it.
            for desc in child.descendants:
                if isinstance(desc, Tag):
                    visited.add(id(desc))

    flush()
    return sections


class HtmlAdapter:
    extensions: ClassVar[tuple[str, ...]] = ("html", "htm")
    binary: ClassVar[bool] = False
    mutable_frontmatter: ClassVar[bool] = False

    def extract(
        self,
        path: Union[str, Path],
        *,
        options: AdapterOptions,
    ) -> ExtractedDocument:
        file_path = Path(path)
        if options.vault_root is not None:
            SecurityGuard.resolve_within_vault(file_path, options.vault_root)

        try:
            stat = file_path.stat()
        except OSError as e:
            raise ValueError(f"cannot stat {file_path}: {e}") from e

        size = stat.st_size
        if size > _MAX_HTML_BYTES:
            logger.warning(
                "html_too_large", path=str(file_path), size=size, max=_MAX_HTML_BYTES,
            )
            raise ValueError(
                f"html file exceeds {_MAX_HTML_BYTES} bytes: {file_path} ({size} bytes)"
            )

        with open(file_path, "rb") as fh:
            raw = fh.read()

        # bs4 is lenient about encoding — pass raw bytes and let it sniff
        # the meta charset declaration.
        soup = BeautifulSoup(raw, _PARSER)

        for noise in _NOISE_TAGS:
            for node in soup.find_all(noise):
                node.decompose()

        root = _select_content_root(soup)
        title = _extract_title(soup, root, file_path.stem)
        sections = _sectionise(root)
        text = "\n\n".join(s.text for s in sections) or root.get_text(
            "\n", strip=True,
        )

        return ExtractedDocument(
            source_path=file_path,
            format="html",
            title=title,
            text=text,
            content_hash=calculate_content_hash(text),
            file_hash=sha256_file(file_path),
            metadata={},
            sections=sections,
            size_bytes=size,
        )


register(HtmlAdapter())
