from pathlib import Path

import pytest

from grimoire.memory.taxonomy import (
    CategoryTree,
    DEFAULT_ROOTS,
    Taxonomy,
    VaultTaxonomy,
    load_taxonomy_from_vault,
    save_taxonomy_to_vault,
)


class TestNormalize:
    def test_lowercases(self):
        assert Taxonomy().normalize("Filosofia") == "filosofia"

    def test_strips_accents(self):
        assert Taxonomy().normalize("Ocultismo Clásico") == "ocultismo-clasico"

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
        tax = Taxonomy(["Filosofia"])
        # both map to canonical "Filosofia"; only one should survive
        out = tax.reconcile(["filosofia", "Filosofia"])
        assert out == ["Filosofia"]

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
            "vocabulary:\n  - filosofia\n  - nihilismo\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == ["filosofia", "nihilismo"]

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
            "vocabulary:\n  - filosofia\n  - null\n  - \"\"\n  - nihilismo\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.tags.vocabulary == ["filosofia", "nihilismo"]

    def test_categories_from_nested_yaml(self, tmp_path):
        (tmp_path / "taxonomy.yml").write_text(
            "categories:\n"
            "  Ciencia:\n"
            "    Física:\n"
            "      - Cuántica\n"
            "  Arte: []\n"
        )
        vt = load_taxonomy_from_vault(tmp_path)
        assert vt.categories.has("Ciencia")
        assert vt.categories.has("Ciencia/Física")
        assert vt.categories.has("Ciencia/Física/Cuántica")
        assert vt.categories.has("Arte")

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
        assert tree.add("Ciencia/Física/Cuántica") is True
        assert tree.has("Ciencia")
        assert tree.has("Ciencia/Física")
        assert tree.has("Ciencia/Física/Cuántica")

    def test_add_existing_returns_false(self):
        tree = CategoryTree()
        tree.add("Arte")
        assert tree.add("Arte") is False

    def test_add_rejects_empty_path(self):
        tree = CategoryTree()
        with pytest.raises(ValueError):
            tree.add("   ")

    def test_remove_drops_subtree(self):
        tree = CategoryTree()
        tree.add("Ciencia/Física/Cuántica")
        tree.add("Ciencia/Biología")
        assert tree.remove("Ciencia/Física") is True
        assert not tree.has("Ciencia/Física")
        assert not tree.has("Ciencia/Física/Cuántica")
        assert tree.has("Ciencia/Biología")

    def test_remove_missing_returns_false(self):
        assert CategoryTree().remove("Nope") is False

    def test_resolve_normalises_input(self):
        tree = CategoryTree()
        tree.add("Ciencia/Física")
        assert tree.resolve("ciencia / fisica") == "Ciencia/Física"
        assert tree.resolve("CIENCIA/FÍSICA") == "Ciencia/Física"

    def test_resolve_unknown_returns_none(self):
        tree = CategoryTree()
        tree.add("Arte")
        assert tree.resolve("Ciencia") is None

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
        tree.add("Ciencia/Física/Cuántica")
        tree.add("Arte")
        yaml_dict = tree.to_yaml_dict()
        assert "Ciencia" in yaml_dict
        assert "Física" in yaml_dict["Ciencia"]
        assert "Cuántica" in yaml_dict["Ciencia"]["Física"]
        assert yaml_dict["Arte"] == []


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
)
ca")
)
