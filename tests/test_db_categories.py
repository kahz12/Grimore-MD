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
        db.set_note_category(nid, "Science/Physics")
        with db._get_connection() as conn:
            row = conn.execute("SELECT category FROM notes WHERE id = ?", (nid,)).fetchone()
        assert row[0] == "Science/Physics"

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
        db.set_note_category(a, "Science")
        db.set_note_category(b, "Science")
        db.set_note_category(c, "Art")

        freq = db.get_category_frequency()
        assert freq == [("Science", 2), ("Art", 1)]

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
        db.set_note_category(a, "Science")
        db.set_note_category(b, "Science/Physics")
        db.set_note_category(c, "Art")

        assert db.count_notes_under_category("Science") == 2
        assert db.count_notes_under_category("Science/Physics") == 1
        assert db.count_notes_under_category("Art") == 1
        assert db.count_notes_under_category("Mathematics") == 0

    def test_empty_path_returns_zero(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.count_notes_under_category("") == 0

    def test_underscore_in_category_does_not_match_other_chars(self, tmp_path):
        # B-04: a literal "_" must not behave like a SQL wildcard.
        db = _make_db(tmp_path)
        legit = _add_note(db, "a.md", "A")
        sneaky = _add_note(db, "b.md", "B")
        db.set_note_category(legit, "50_off/sub")          # belongs under "50_off"
        db.set_note_category(sneaky, "50aoff/sub")         # would falsely match if "_" stayed wild

        assert db.count_notes_under_category("50_off") == 1

    def test_percent_in_category_does_not_match_anything(self, tmp_path):
        # B-04: a literal "%" must not behave like a SQL wildcard.
        db = _make_db(tmp_path)
        legit = _add_note(db, "a.md", "A")
        unrelated = _add_note(db, "b.md", "B")
        db.set_note_category(legit, "100%/sub")
        db.set_note_category(unrelated, "100xyz/sub")

        assert db.count_notes_under_category("100%") == 1


class TestGetNotesByCategory:
    def test_recursive_includes_descendants(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        db.set_note_category(a, "Science")
        db.set_note_category(b, "Science/Physics")

        rows = db.get_notes_by_category("Science", recursive=True)
        titles = {r[2] for r in rows}
        assert titles == {"A", "B"}

    def test_flat_only_exact(self, tmp_path):
        db = _make_db(tmp_path)
        a = _add_note(db, "a.md", "A")
        b = _add_note(db, "b.md", "B")
        db.set_note_category(a, "Science")
        db.set_note_category(b, "Science/Physics")

        rows = db.get_notes_by_category("Science", recursive=False)
        titles = {r[2] for r in rows}
        assert titles == {"A"}

    def test_recursive_treats_underscore_literally(self, tmp_path):
        # B-04: same fix path through get_notes_by_category.
        db = _make_db(tmp_path)
        legit = _add_note(db, "a.md", "Legit")
        sneaky = _add_note(db, "b.md", "Sneaky")
        db.set_note_category(legit, "50_off/sub")
        db.set_note_category(sneaky, "50aoff/sub")

        titles = {r[2] for r in db.get_notes_by_category("50_off", recursive=True)}
        assert titles == {"Legit"}
