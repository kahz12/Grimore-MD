"""
Link Injector.
This module injects semantic connections (wikilinks) into Markdown notes.
It surfaces related content by adding a 'Suggested Connections' section.
"""
import re
from pathlib import Path
from grimoire.utils.atomic import atomic_write
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

# Characters that would let a malicious title escape a wikilink or markdown list item.
_WIKILINK_FORBIDDEN = re.compile(r"[\[\]\|\r\n]+")


def _safe_wikilink_title(raw: str) -> str:
    """Sanitizes a title to ensure it's a valid and safe Markdown wikilink."""
    title = _WIKILINK_FORBIDDEN.sub(" ", str(raw))
    return title.strip()[:200] or "Untitled"


def _safe_reason(raw: str) -> str:
    """Sanitizes the reason text to prevent breaking Markdown formatting."""
    text = re.sub(r"[\r\n]+", " ", str(raw))
    return text.strip()[:300]


class LinkInjector:
    """
    Handles the modification of Markdown files to include semantic links.
    """
    def inject_links(self, file_path: Path, connections: list[dict], dry_run: bool = True):
        """
        Adds or updates a '## 🔗 Suggested Connections' section in the Markdown file.
        The 'connections' parameter is a list of dicts with 'title' and 'reason'.
        """
        if not connections:
            return

        section_header = "## 🔗 Suggested Connections"
        # Build the Markdown list of links
        links_content = "\n".join([
            f"- [[{_safe_wikilink_title(c['title'])}]] — {_safe_reason(c.get('reason', 'Semantic connection found.'))}"
            for c in connections
        ])
        new_section = f"\n\n{section_header}\n{links_content}\n"

        if dry_run:
            logger.info("dry_run_link_injection", path=str(file_path), connections=[c['title'] for c in connections])
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Identify the existing section if it exists, to replace it instead of appending
            pattern = re.compile(rf"\n\n{re.escape(section_header)}.*?(?=\n\n|\Z)", re.DOTALL)
            
            if pattern.search(content):
                # Update existing section
                new_content = pattern.sub(new_section, content)
            else:
                # Append new section to the end of the file
                new_content = content.rstrip() + new_section
            
            # Atomic write to ensure file integrity
            atomic_write(
                file_path,
                lambda fh: fh.write(new_content.encode("utf-8")),
                mode="wb",
            )

            logger.info("links_injected", path=str(file_path))
        except Exception as e:
            logger.error("link_injection_failed", path=str(file_path), error=str(e))
