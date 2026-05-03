from grimoire.cognition.tagger import (
    _render_category_menu,
    _sanitize_category,
    _sanitize_summary,
    _sanitize_tag,
)
from grimoire.memory.taxonomy import CategoryTree, Taxonomy


class TestSanitizeTag:
    def test_normalizes_via_taxonomy(self):
        tax = Taxonomy()
        assert _sanitize_tag("Classical Occultism", tax) == "classical-occultism"

    def test_strips_hash_prefix(self):
        tax = Taxonomy()
        assert _sanitize_tag("#philosophy", tax) == "philosophy"

    def test_rejects_non_string(self):
        tax = Taxonomy()
        assert _sanitize_tag(123, tax) is None
        assert _sanitize_tag(None, tax) is None
        assert _sanitize_tag(["a"], tax) is None

    def test_rejects_empty_after_normalization(self):
        tax = Taxonomy()
        # a string with only punctuation normalizes to empty
        assert _sanitize_tag("!!!", tax) is None

    def test_rejects_overlong_tag(self):
        tax = Taxonomy()
        long = "a" * 200
        assert _sanitize_tag(long, tax) is None

    def test_maps_to_canonical_from_taxonomy(self):
        tax = Taxonomy(["Classical Occultism"])
        # sanitize returns normalized form; reconciliation to canonical
        # happens later in Tagger.tag_note via taxonomy.reconcile
        assert _sanitize_tag("classical occultism", tax) == "classical-occultism"


class TestSanitizeSummary:
    def test_strips_control_chars(self):
        assert _sanitize_summary("hello\x00\x07world") == "helloworld"

    def test_collapses_whitespace(self):
        assert _sanitize_summary("a   b\n\n\tc") == "a b c"

    def test_truncates_to_300(self):
        assert len(_sanitize_summary("x" * 500)) == 300

    def test_non_string_returns_empty(self):
        assert _sanitize_summary(None) == ""
        assert _sanitize_summary(42) == ""
        assert _sanitize_summary(["list"]) == ""

    def test_preserves_short_summary(self):
        assert _sanitize_summary("a short note") == "a short note"


class TestSanitizeCategory:
    def test_resolves_match(self):
        tree = CategoryTree()
        tree.add("Science/Physics")
        assert _sanitize_category("science/physics", tree) == "Science/Physics"

    def test_unknown_returns_none(self):
        tree = CategoryTree()
        tree.add("Art")
        assert _sanitize_category("Mathematics", tree) is None

    def test_non_string_returns_none(self):
        tree = CategoryTree()
        assert _sanitize_category(None, tree) is None
        assert _sanitize_category(42, tree) is None


class TestRenderCategoryMenu:
    def test_empty_tree_has_sentinel(self):
        tree = CategoryTree()
        assert "no categories" in _render_category_menu(tree).lower()

    def test_lists_all_paths(self):
        tree = CategoryTree()
        tree.add("Science/Physics")
        tree.add("Art")
        menu = _render_category_menu(tree)
        assert "- Science" in menu
        assert "- Science/Physics" in menu
        assert "- Art" in menu

    def test_truncates_to_max_paths(self):
        tree = CategoryTree()
        for i in range(10):
            tree.add(f"Node{i}")
        menu = _render_category_menu(tree, max_paths=3)
        assert menu.count("\n") == 2  # 3 bullets → 2 newlines
