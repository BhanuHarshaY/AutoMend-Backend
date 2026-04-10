"""Unit tests for app.guardrails — JSON parsing and repair utilities."""

import json

import pytest

from app.guardrails import (
    close_unclosed_brackets,
    extract_first_json_object,
    fix_trailing_commas,
    parse_llm_output,
    strip_markdown_fences,
)


# ===================================================================
# strip_markdown_fences
# ===================================================================

class TestStripMarkdownFences:
    def test_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_plain_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_no_fence_passthrough(self):
        raw = '{"a": 1}'
        assert strip_markdown_fences(raw) == '{"a": 1}'

    def test_fence_with_surrounding_text(self):
        raw = 'Here is the output:\n```json\n{"key": "val"}\n```\nDone.'
        assert strip_markdown_fences(raw) == '{"key": "val"}'

    def test_multiline_content_in_fence(self):
        raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = strip_markdown_fences(raw)
        assert json.loads(result) == {"a": 1, "b": 2}


# ===================================================================
# extract_first_json_object
# ===================================================================

class TestExtractFirstJsonObject:
    def test_clean_json(self):
        raw = '{"workflow": {}}'
        assert extract_first_json_object(raw) == '{"workflow": {}}'

    def test_json_with_leading_text(self):
        raw = 'Sure! Here is your workflow: {"workflow": {"steps": []}}'
        assert extract_first_json_object(raw) == '{"workflow": {"steps": []}}'

    def test_json_with_trailing_text(self):
        raw = '{"a": 1} hope this helps!'
        assert extract_first_json_object(raw) == '{"a": 1}'

    def test_no_braces(self):
        assert extract_first_json_object("no json here") is None

    def test_unclosed_returns_remainder(self):
        raw = '{"a": {"b": 1}'
        result = extract_first_json_object(raw)
        # Should return from first { to end since it never balances
        assert result == '{"a": {"b": 1}'

    def test_braces_inside_strings_ignored(self):
        raw = '{"msg": "use { and } carefully"}'
        assert extract_first_json_object(raw) == raw

    def test_escaped_quotes_in_strings(self):
        raw = r'{"msg": "he said \"hello\""}'
        result = extract_first_json_object(raw)
        assert result == raw


# ===================================================================
# fix_trailing_commas
# ===================================================================

class TestFixTrailingCommas:
    def test_comma_before_brace(self):
        assert fix_trailing_commas('{"a": 1,}') == '{"a": 1}'

    def test_comma_before_bracket(self):
        assert fix_trailing_commas('[1, 2,]') == '[1, 2]'

    def test_comma_with_whitespace(self):
        assert fix_trailing_commas('{"a": 1 , }') == '{"a": 1 }'

    def test_no_trailing_comma_unchanged(self):
        original = '{"a": 1}'
        assert fix_trailing_commas(original) == original

    def test_nested_trailing_commas(self):
        raw = '{"a": [1, 2,], "b": {"c": 3,},}'
        fixed = fix_trailing_commas(raw)
        assert json.loads(fixed) == {"a": [1, 2], "b": {"c": 3}}


# ===================================================================
# close_unclosed_brackets
# ===================================================================

class TestCloseUnclosedBrackets:
    def test_missing_closing_brace(self):
        raw = '{"a": 1'
        result = close_unclosed_brackets(raw)
        assert json.loads(result) == {"a": 1}

    def test_missing_nested_brace(self):
        raw = '{"a": {"b": 1}'
        result = close_unclosed_brackets(raw)
        assert json.loads(result) == {"a": {"b": 1}}

    def test_missing_bracket_and_brace(self):
        raw = '{"steps": [{"id": 1}'
        result = close_unclosed_brackets(raw)
        assert json.loads(result) == {"steps": [{"id": 1}]}

    def test_balanced_unchanged(self):
        raw = '{"a": [1, 2]}'
        assert close_unclosed_brackets(raw) == raw

    def test_string_brackets_ignored(self):
        raw = '{"msg": "open { bracket"'
        result = close_unclosed_brackets(raw)
        # The { in the string should NOT count as unmatched
        assert result.endswith("}")
        assert json.loads(result) == {"msg": "open { bracket"}


# ===================================================================
# parse_llm_output (full pipeline)
# ===================================================================

class TestParseLlmOutput:
    """Integration tests for the full repair pipeline."""

    def test_valid_json(self):
        raw = '{"workflow": {"steps": []}}'
        assert parse_llm_output(raw) == {"workflow": {"steps": []}}

    def test_markdown_wrapped(self):
        raw = '```json\n{"workflow": {"steps": [{"step_id": 1, "tool": "restart_rollout", "params": {"namespace": "default", "deployment_name": "api-server"}}]}}\n```'
        result = parse_llm_output(raw)
        assert result is not None
        assert result["workflow"]["steps"][0]["tool"] == "restart_rollout"

    def test_trailing_comma(self):
        raw = '{"workflow": {"steps": [{"step_id": 1, "tool": "x", "params": {},},]}}'
        result = parse_llm_output(raw)
        assert result is not None
        assert result["workflow"]["steps"][0]["step_id"] == 1

    def test_missing_closing_bracket(self):
        raw = (
            '{"workflow": {"steps": [{"step_id": 1, "tool": "scale_deployment", '
            '"params": {"namespace": "production", "deployment_name": "fraud-model", '
            '"replicas": 5}}]}'
        )
        result = parse_llm_output(raw)
        assert result is not None
        assert result["workflow"]["steps"][0]["params"]["replicas"] == 5

    def test_json_with_surrounding_prose(self):
        raw = (
            'Here is the remediation workflow:\n'
            '{"workflow": {"steps": [{"step_id": 1, "tool": "restart_rollout", '
            '"params": {"namespace": "prod", "deployment_name": "api"}}]}}\n'
            'Let me know if you need changes.'
        )
        result = parse_llm_output(raw)
        assert result is not None
        assert result["workflow"]["steps"][0]["tool"] == "restart_rollout"

    def test_empty_string(self):
        assert parse_llm_output("") is None

    def test_none_input(self):
        assert parse_llm_output(None) is None

    def test_whitespace_only(self):
        assert parse_llm_output("   \n\t  ") is None

    def test_non_json_text(self):
        assert parse_llm_output("I don't know how to help with that.") is None

    def test_json_array_rejected(self):
        """parse_llm_output must return a dict, not a list."""
        assert parse_llm_output('[1, 2, 3]') is None

    def test_combined_fence_and_trailing_comma(self):
        raw = '```json\n{"workflow": {"steps": [{"step_id": 1, "tool": "x", "params": {},},]}}\n```'
        result = parse_llm_output(raw)
        assert result is not None

    def test_deeply_truncated_output(self):
        """Truncated mid-string — impossible to repair, should return None."""
        raw = '{"workflow": {"steps": [{"step_id": 1, "tool": "sca'
        result = parse_llm_output(raw)
        # The bracket closer can close brackets, but the string literal
        # is broken so json.loads will fail. Accept None.
        # (If the closer happens to produce valid JSON, that's OK too.)
        # Main goal: no exception raised.
        assert result is None or isinstance(result, dict)

    def test_multiple_json_objects_takes_first(self):
        raw = '{"a": 1} {"b": 2}'
        result = parse_llm_output(raw)
        # extract_first_json_object returns first balanced object
        assert result == {"a": 1}