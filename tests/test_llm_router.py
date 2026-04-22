from grimoire.cognition.llm_router import _extract_json_object


class TestExtractJsonObject:
    def test_direct_parse(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_block(self):
        text = 'some prose\n```json\n{"tags": ["a", "b"]}\n```\nmore prose'
        assert _extract_json_object(text) == {"tags": ["a", "b"]}

    def test_unlabeled_fence(self):
        text = '```\n{"x": 2}\n```'
        assert _extract_json_object(text) == {"x": 2}

    def test_bracket_balanced_fallback(self):
        text = 'here is the answer: {"a": 1, "b": {"c": 2}} and then trailing noise'
        assert _extract_json_object(text) == {"a": 1, "b": {"c": 2}}

    def test_braces_inside_strings_do_not_confuse(self):
        text = '{"key": "value with } brace inside"}'
        assert _extract_json_object(text) == {"key": "value with } brace inside"}

    def test_escaped_quotes_respected(self):
        text = '{"k": "he said \\"hi\\""}'
        assert _extract_json_object(text) == {"k": 'he said "hi"'}

    def test_returns_none_on_empty(self):
        assert _extract_json_object("") is None

    def test_returns_none_on_no_braces(self):
        assert _extract_json_object("just plain prose here") is None

    def test_returns_none_for_non_dict_top_level(self):
        # A JSON array is not a dict — this function only returns dicts
        assert _extract_json_object("[1, 2, 3]") is None

    def test_returns_none_on_malformed(self):
        assert _extract_json_object("{not valid json at all") is None

    def test_prefers_direct_parse_when_clean(self):
        text = '{"outer": {"inner": 1}}'
        assert _extract_json_object(text) == {"outer": {"inner": 1}}
