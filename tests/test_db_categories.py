from grimoire.memory.db import Database


def _make_db(tmp_path):
    return Database(str(tmp_path / "grimoire.db"))


def _add_note(db: Database, path: str, title: str) -> int:
    return db.upsert_note(path, title, content_hash="h-" + path)


class TestCategoryColumn:
    def test_column_is_migrated_on_init(self, tmp_path):
        db = _make_db(tmp_path)
        with db._get_connection() as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        assert "category" in cols

    def test_set_and_clear_category(self, tmp_path):
        db = _make_db(tmp_path)
        nid = _add_note(db, "a.md", "A")
        db.set_note_category(nid, "Ciencia/Física")
        with db._get_connection() as conn:
            row = conn.execute("SELECT category FROM notes WHERE id = ?", (nid,)).fetchone()
        assert row[0] == "Ciencia/Física"

        db.set_note_category(nid, None)
        with db._get_connection() as conn:
            row = conn.execute("SELECT category FROM notes WHERE id = ?", (nid,)).fetchone()
        assert row[0] is None


class TestCategoryFrequency:
    def test_returns_counts_sorted_desc(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        c = _add_note(db, "c.md", "C")
        db.set_note_category(a, "Ciencia")
        db.set_note_category(b, "Ciencia")
        db.set_note_category(c, "Arte")

        freq = db.get_category_frequency()
        assert freq == [("Ciencia", 2), ("Arte", 1)]

    def test_ignores_null_and_empty(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        db.set_note_category(a, "")
        db.set_note_category(b, None)

        assert db.get_category_frequency() == []


class TestCountNotesUnderCategory:
    def test_direct_and_descendants(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        c = _add_note(db, "c.md", "C")
        db.set_note_category(a, "Ciencia")
        db.set_note_category(b, "Ciencia/Física")
        db.set_note_category(c, "Arte")

        assert db.count_notes_under_category("Ciencia") == 2
        assert db.count_notes_under_category("Ciencia/Física") == 1
        assert db.count_notes_under_category("Arte") == 1
        assert db.count_notes_under_category("Matemáticas") == 0

    def test_empty_path_returns_zero(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.count_notes_under_category("") == 0


class TestGetNotesByCategory:
    def test_recursive_includes_descendants(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        db.set_note_category(a, "Ciencia")
        db.set_note_category(b, "Ciencia/Física")

        rows = db.get_notes_by_category("Ciencia", recursive=True)
        titles = {r[2] for r in rows}
        assert titles == {"A", "B"}

    def test_flat_only_exact(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        db.set_note_category(a, "Ciencia")
        db.set_note_category(b, "Ciencia/Física")

        rows = db.get_notes_by_category("Ciencia", recursive=False)
        titles = {r[2] for r in rows}
        assert titles == {"A"}
