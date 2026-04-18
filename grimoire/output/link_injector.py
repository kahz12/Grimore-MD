import re
from pathlib import Path
from grimoire.utils.logger import get_logger

logger = get_logger(__name__)

class LinkInjector:
    def inject_links(self, file_path: Path, connections: list[dict], dry_run: bool = True):
        """
        Adds or updates a '## 🔗 Suggested Connections' section in the markdown file.
        Connections is a list of dicts with 'title' and 'reason'.
        """
        if not connections:
            return

        section_header = "## 🔗 Suggested Connections"
        links_content = "\n".join([f"- [[{c['title']}]] — {c.get('reason', 'Semantic connection found.')}" for c in connections])
        new_section = f"\n\n{section_header}\n{links_content}\n"

        if dry_run:
            logger.info("dry_run_link_injection", path=str(file_path), connections=[c['title'] for c in connections])
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if section already exists
            pattern = re.compile(rf"\n\n{re.escape(section_header)}.*?(?=\n\n|\Z)", re.DOTALL)
            
            if pattern.search(content):
                # Update existing section
                new_content = pattern.sub(new_section, content)
            else:
                # Append new section
                new_content = content.rstrip() + new_section
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            logger.info("links_injected", path=str(file_path))
        except Exception as e:
            logger.error("link_injection_failed", path=str(file_path), error=str(e))
