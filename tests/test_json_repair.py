from __future__ import annotations

import pytest

from backend.llm.json_repair import JSONRepairError, parse_json, parse_string_array


def test_plain_json_and_fences() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure, here you go:\n[1, 2, 3]\ntrailing') == [1, 2, 3]


def test_truncated_string_is_closed() -> None:
    assert parse_json('[{"text": "hello') == [{"text": "hello"}]
    assert parse_json('{"summary": "a cat', ) == {"summary": "a cat"}


def test_truncated_array_is_closed() -> None:
    assert parse_json('["one", "two"') == ["one", "two"]
    assert parse_json('{"messages": ["W", "LUL"') == {"messages": ["W", "LUL"]}


def test_truncated_number_or_key_is_dropped_not_fatal() -> None:
    first = parse_json('{"summary": "abc", "action": "walks", "confidence": 0.')
    assert first["summary"] == "abc"
    assert first["action"] == "walks"
    assert "confidence" not in first

    second = parse_json('{"summary": "abc", "action": "walks", "scene_change": tr')
    assert second == {"summary": "abc", "action": "walks"}


def test_string_array_from_object() -> None:
    assert parse_string_array('{"messages": ["W", "LUL"]}') == ["W", "LUL"]


def test_empty_or_garbage_raises() -> None:
    with pytest.raises(JSONRepairError):
        parse_json("no json here")
    with pytest.raises(JSONRepairError):
        parse_string_array('{"messages": "not a list"}')
