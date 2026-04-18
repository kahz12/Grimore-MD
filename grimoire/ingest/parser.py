import frontmatter
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
from grimoire.utils.hashing import calculate_content_hash

@dataclass
class ParsedNote:
    path: Path
    title: str
    metadata: dict[str, Any]
    content: str
    content_hash: str

class MarkdownParser:
    def parse_file(self, file_path: Path) -> ParsedNote:
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        content = post.content
        metadata = post.metadata
        
        # Determine title: from metadata or first H1 or filename
        title = metadata.get('title')
        if not title:
            h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1)
            else:
                title = file_path.stem

        return ParsedNote(
            path=file_path,
            title=title,
            metadata=metadata,
            content=content,
            content_hash=calculate_content_hash(content)
        )
