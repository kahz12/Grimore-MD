"""
Tag storage: the ``tags`` table, the note↔tag junction rows, and the
frequency/lookup queries behind ``grimore tags`` and the browse screens.
"""
from grimore.memory._base import DbBase



class TagsMixin(DbBase):
    """Tag sync and lookup queries for :class:`Database`."""

    def upsert_tags(self, note_id: int, tag_names: list[str]) -> None:
        """
        Syncs a note's tags with the database.
        Ensures the 'tags' table has the tag names and updates 'note_tags' association.
        """
        with self._get_connection() as conn:
            # Clear old associations
            conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
            for name in tag_names:
                if not name:
                    continue
                # Ensure the tag exists globally
                conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
                row = conn.execute(
                    "SELECT id FROM tags WHERE name = ?", (name,)
                ).fetchone()
                if row is None:
                    continue
                # Associate tag with this note
                conn.execute(
                    "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                    (note_id, row[0]),
                )
            conn.commit()

    def get_tag_frequency(self, limit: int = 50) -> list[tuple[str, int]]:
        """Returns the most used tags and their frequencies."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT t.name, COUNT(nt.note_id) AS freq
                FROM tags t
                JOIN note_tags nt ON nt.tag_id = t.id
                GROUP BY t.id
                ORDER BY freq DESC, t.name ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(name, count) for name, count in rows]

    def get_tag_count(self) -> int:
        """Returns the number of distinct tags currently in use."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT nt.tag_id)
                FROM note_tags nt
                """
            ).fetchone()
        return int(row[0]) if row else 0

    def get_notes_by_tag(self, tag_name: str) -> list[tuple[int, str, str]]:
        """Retrieves all notes that have a specific tag."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT n.id, n.path, n.title
                FROM notes n
                JOIN note_tags nt ON nt.note_id = n.id
                JOIN tags t ON t.id = nt.tag_id
                WHERE t.name = ?
                ORDER BY n.title
                """,
                (tag_name,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def purge_unused_tags(self) -> int:
        """Deletes tags that are not associated with any notes."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tags WHERE id NOT IN (SELECT DISTINCT tag_id FROM note_tags)"
            )
            conn.commit()
            return cursor.rowcount or 0
