"""Forgiving JSON parsing for local-model output.

The model is asked for JSON; local models still wrap it in fences, add a
sentence before it, or stop mid-string. This module strips fences, extracts the
first structured value, and closes unterminated brackets/strings. It is not a
license to accept garbage: callers still validate the parsed shape.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)(?:```|$)", re.IGNORECASE | re.DOTALL)
_START = re.compile(r"[\[{]")


class JSONRepairError(ValueError):
    pass


def strip_code_fences(text: str) -> str:
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_start(text: str) -> int:
    match = _START.search(text)
    return match.start() if match else -1


def _close_unterminated(candidate: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack:
                break
            expected = "[" if char == "]" else "{"
            if stack[-1] != expected:
                break
            stack.pop()

    repaired = candidate
    if in_string:
        if escaped:
            repaired = repaired[:-1]
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "]" if opener == "[" else "}"
    return repaired


def _raw_decode(candidate: str) -> Any:
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(candidate)
    return value


def parse_json(text: str) -> Any:
    """Parse the first JSON value in *text*, repairing common truncation."""
    candidate = strip_code_fences(text)
    start = _find_start(candidate)
    if start < 0:
        raise JSONRepairError("no JSON object or array found")
    candidate = candidate[start:]

    try:
        return _raw_decode(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = _close_unterminated(candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # A completion cap can cut anywhere, including in the middle of a number
    # or a key. Drop the last structural member and close the object instead of
    # failing the whole observation. Defaults in the caller fill omitted keys.
    for index in reversed(_structural_commas(candidate)):
        prefix = candidate[:index].rstrip()
        if not prefix:
            continue
        try:
            return json.loads(_close_unterminated(prefix))
        except json.JSONDecodeError:
            continue

    raise JSONRepairError("could not repair JSON")


def _structural_commas(text: str) -> list[int]:
    """Comma positions outside strings, in source order."""
    positions: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            positions.append(index)
    return positions


def parse_object(text: str) -> dict[str, Any]:
    value = parse_json(text)
    if not isinstance(value, dict):
        raise JSONRepairError(f"expected object, got {type(value).__name__}")
    return value


def parse_string_array(text: str) -> list[str]:
    value = parse_json(text)
    if isinstance(value, dict):
        for key in ("messages", "output", "chat", "items", "replies"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                value = candidate
                break
    if not isinstance(value, list):
        raise JSONRepairError(f"expected array, got {type(value).__name__}")
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            for key in ("text", "message", "content"):
                if isinstance(item.get(key), str):
                    result.append(item[key])
                    break
        else:
            raise JSONRepairError(f"array item has unsupported type {type(item).__name__}")
    return result
