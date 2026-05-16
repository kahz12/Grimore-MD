import pytest

from grimore.memory.taxonomy import (
    CategoryTree,
    DEFAULT_ROOTS,
    Taxonomy,
    VaultTaxonomy,
    load_taxonomy_from_vault,
    save_taxonomy_to_vault,
)


class TestNormalize:
    def test_lowercases(self):
        assert Taxonomy().normalize("Philosophy") == "philosophy"

    def test_strips_accents(self):
        assert Taxonomy().normalize("Classical Occultism") == "classical-occultism"

    def test_non_alnum_to_hyphens(self):
        assert Taxonomy().normalize("foo/bar_baz!qux") == "foo-bar-baz-qux"

    def test_trims_surrounding_hyphens(self):
        assert Taxonomy().normalize("  --hello--  ") == "hello"

    def test_collapses_runs(self):
        assert Taxonomy().normalize("hello    world") == "hello-world"

    def test_keeps_digits(self):
        assert Taxonomy().normalize("GPT-4 Review") == "gpt-4-review"


class TestReconcile:
    def test_empty(self):
        assert Taxonomy().reconcile([]) == []

    def test_maps_to_canonical(self):
        tax = Taxonomy(["Classical Occultism"])
        # normalized form → canonical spelling
        assert tax.reconcile(["classical-occultism"]) == ["Classical Occultism"]

    def test_unknown_kept_verbatim(self):
        tax = Taxonomy(["philosophy"])
        assert tax.reconcile(["nihilism"]) == ["nihilism"]

    def test_preserves_first_seen_order(self):
        tax = Taxonomy()
        assert tax.reconcile(["b", "a", "c"]) == ["b", "a", "c"]

    def test_deduplicates_after_canonicalization(self):
        tax = Taxonomy(["Philosophy"])
        # both map to canonical "Philosophy"; only one should survive
        out = tax.reconcile(["philosophy", "Philosophy"])
        assert out == ["Philosophy"]

    def test_skips_empty_tags(self):
        assert Taxonomy().reconcile(["", "a", ""]) == ["a"]


class TestLoadTaxonomyFromVault:
    def test_missing_file_returns_empty(self, tmp_path):
        vt = load_taxonomy_from_vault(tmp_path)
        assert isinstance(vt, VaultTaxonomy)
        assert vt.tags.vocabulary == []
        # When no file is present defaults should seed the roots.
        assert set(vt.categories.roots()) == set(DEFAULT_ROOTS)

    def test_loads_valid_yaml(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text(
            "vocabulary:\n  - philosophy\n  - nihilism\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == ["philosophy", "nihilism"]

    def test_malformed_yaml_returns_empty(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text("vocabulary: [unclosed\n")
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == []

    def test_missing_vocabulary_key_returns_empty(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text("other: value\n")
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == []

    def test_filters_non_string_entries(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text(
            "vocabulary:\n  - philosophy\n  - null\n  - \"\"\n  - nihilism\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == ["philosophy", "nihilism"]

    def test_categories_from_nested_yaml(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text(
            "categories:\n"
            "  Science:\n"
            "    Physics:\n"
            "      - Quantum\n"
            "  Art: []\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.categories.has("Science")
        assert vt.categories.has("Science/Physics")
        assert vt.categories.has("Science/Physics/Quantum")
        assert vt.categories.has("Art")

    def test_categories_key_present_but_empty_respects_user(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text("categories:\n")
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.categories.is_empty()


class TestCategoryTree:
    def test_with_defaults_seeds_all_roots(self):
        tree = CategoryTree.with_defaults()
        assert set(tree.roots()) == set(DEFAULT_ROOTS)

    def test_add_creates_ancestors(self):
        tree = CategoryTree()
        assert tree.add("Science/Physics/Quantum") is True
        assert tree.has("Science")
        assert tree.has("Science/Physics")
        assert tree.has("Science/Physics/Quantum")

    def test_add_existing_returns_false(self):
        tree = CategoryTree()
        tree.add("Art")
        assert tree.add("Art") is False

    def test_add_rejects_empty_path(self):
        tree = CategoryTree()
        with pytest.raises(ValueError):
            tree.add("   ")

    def test_remove_drops_subtree(self):
        tree = CategoryTree()
        tree.add("Science/Physics/Quantum")
        tree.add("Science/Biology")
        assert tree.remove("Science/Physics") is True
        assert not tree.has("Science/Physics")
        assert not tree.has("Science/Physics/Quantum")
        assert tree.has("Science/Biology")

    def test_remove_missing_returns_false(self):
        assert CategoryTree().remove("Nope") is False

    def test_resolve_normalises_input(self):
        tree = CategoryTree()
        tree.add("Science/Physics")
        assert tree.resolve("science / physics") == "Science/Physics"
        assert tree.resolve("SCIENCE/PHYSICS") == "Science/Physics"

    def test_resolve_unknown_returns_none(self):
        tree = CategoryTree()
        tree.add("Art")
        assert tree.resolve("Science") is None

    def test_paths_preorder(self):
        tree = CategoryTree()
        tree.add("A/B/C")
        tree.add("A/D")
        paths = tree.paths()
        # A appears before its descendants; B before C.
        assert paths.index("A") < paths.index("A/B")
        assert paths.index("A/B") < paths.index("A/B/C")
        assert "A/D" in paths

    def test_to_yaml_dict_roundtrips(self, tmp_path):
        tree = CategoryTree()
        tree.add("Science/Physics/Quantum")
        tree.add("Art")
        yaml_dict = tree.to_yaml_dict()
        assert "Science" in yaml_dict
        assert "Physics" in yaml_dict["Science"]
        assert "Quantum" in yaml_dict["Science"]["Physics"]
        assert yaml_dict["Art"] == []


class TestSaveTaxonomyToVault:
    def test_roundtrip_preserves_tree(self, tmp_path):
        vt = VaultTaxonomy()
        vt.categories = CategoryTree()
        vt.categories.add("History/Modern")
        vt.categories.add("Science/Physics/Quantum")
        vt.tags = Taxonomy(["philosophy"])

        save_taxonomy_to_vault(tmp_path, vt)

        reloaded = load_taxonomy_from_vault(tmp_path)
        assert reloaded.tags.vocabulary == ["philosophy"]
        assert reloaded.categories.has("History/Modern")
        assert reloaded.categories.has("Science/Physics/Quantum")
