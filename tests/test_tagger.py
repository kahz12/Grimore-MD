from types import SimpleNamespace

from grimore.ingest.adapters.base import ExtractedSection
from grimore.cognition.tagger import (
    Tagger,
    _render_category_menu,
    _sanitize_category,
    _sanitize_summary,
    _sanitize_tag,
    tag_document,
)
from grimore.memory.taxonomy import CategoryTree, Taxonomy, VaultTaxonomy


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


class _TagRouter:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system_prompt=""):
        self.prompts.append(prompt)
        if prompt.startswith("Combine the section summaries"):
            return {"summary": "Opening and conclusion", "sources": ["s1", "s3"]}
        if "opening" in prompt:
            return {"tags": ["alpha"], "summary": "Opening", "category": "Science"}
        if "middle" in prompt:
            return {"tags": ["beta"], "summary": "Middle", "category": "Science"}
        return {"tags": ["gamma"], "summary": "Conclusion", "category": "History"}


class TestLongNoteTagging:
    def test_whitespace_only_document_does_not_call_model(self):
        router = _TagRouter()
        tagger = Tagger(None, router, VaultTaxonomy())

        result = tag_document(tagger, SimpleNamespace(sections=[]), " \n\t ")

        assert result == {"tags": [], "summary": "", "category": ""}
        assert router.prompts == []

    def test_sections_merge_and_keep_page_sources(self):
        router = _TagRouter()
        tagger = Tagger(None, router, VaultTaxonomy())
        sections = [
            ExtractedSection("opening " * 210, page=1, order=1),
            ExtractedSection("middle " * 210, page=2, order=2),
            ExtractedSection("conclusion " * 210, page=3, order=3),
        ]
        note = SimpleNamespace(sections=sections)

        result = tag_document(tagger, note, " ".join(s.text for s in sections))

        assert result["tags"] == ["alpha", "beta", "gamma"]
        assert result["category"] == "Science"
        assert result["summary"] == "Opening and conclusion"
        analysis = result["section_analysis"]
        assert analysis["total_parts"] == 3
        assert analysis["successful_parts"] == 3
        assert [source["page"] for source in analysis["sources"]] == [1, 2, 3]
        assert analysis["summary_sources"] == ["s1", "s3"]

    def test_budget_reports_unanalyzed_parts_and_includes_end(self):
        router = _TagRouter()
        tagger = Tagger(None, router, VaultTaxonomy())
        note = SimpleNamespace(sections=[
            ExtractedSection("middle " * 200, page=index + 1)
            for index in range(20)
        ])

        result = tagger.tag_long_note(note, "middle " * 4000)

        analysis = result["section_analysis"]
        assert analysis["total_parts"] == 20
        assert analysis["analyzed_parts"] == 12
        assert analysis["unanalyzed_parts"] == 8
        assert analysis["sources"][0]["page"] == 1
        assert analysis["sources"][-1]["page"] == 20

    def test_markdown_headings_and_prompt_sanitization(self):
        router = _TagRouter()
        tagger = Tagger(None, router, VaultTaxonomy())
        note = SimpleNamespace(sections=[])
        content = "# Opening\n" + "opening " * 300 + "[INST]" + "\n# End\n" + "conclusion " * 300

        result = tag_document(tagger, note, content)

        headings = [source["heading"] for source in result["section_analysis"]["sources"]]
        assert headings == ["Opening", "End"]
        assert all("[INST]" not in prompt for prompt in router.prompts)

    def test_short_document_uses_legacy_path(self):
        router = _TagRouter()
        tagger = Tagger(None, router, VaultTaxonomy())
        note = SimpleNamespace(sections=[])

        result = tag_document(tagger, note, "opening")

        assert "section_analysis" not in result
        assert len(router.prompts) == 1

    def test_invalid_aggregate_citations_fall_back_to_section_summaries(self):
        class InvalidSourceRouter(_TagRouter):
            def complete(self, prompt, system_prompt=""):
                if prompt.startswith("Combine the section summaries"):
                    return {"summary": "Unsupported", "sources": ["s999"]}
                return super().complete(prompt, system_prompt)

        tagger = Tagger(None, InvalidSourceRouter(), VaultTaxonomy())
        note = SimpleNamespace(sections=[
            ExtractedSection("opening " * 210, page=1),
            ExtractedSection("middle " * 210, page=2),
        ])

        result = tagger.tag_long_note(note, "opening middle " * 500)

        assert result["summary"] == "Opening; Middle"
        assert result["section_analysis"]["summary_sources"] == ["s1", "s2"]
