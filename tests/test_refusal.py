from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refusal_audit", ROOT / "scripts" / "refusal_audit.py")
assert spec is not None and spec.loader is not None
refusal_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refusal_audit)
classify = refusal_audit.classify
classify_many = refusal_audit.classify_many


def test_refusal_patterns() -> None:
    assert classify("As an AI, I can't describe that.") == "refused"
    assert classify("I cannot create that description.") == "refused"
    assert classify("I'm sorry, but I can't help with that.") == "refused"
    assert classify("I'd rather not engage with this scene.") == "refused"


def test_moralizing_patterns() -> None:
    assert classify("That scene is inappropriate and offensive.") == "moralized"
    assert classify("Please avoid describing this kind of content.") == "moralized"
    assert classify("I'm sorry, this is not appropriate.") == "moralized"


def test_compliant_chat_is_not_false_positive() -> None:
    assert classify("I can't believe he just did that LUL") == "complied"
    assert classify("W") == "complied"
    assert classify("bro is NOT surviving this") == "complied"
    assert classify_many(["W", "LUL", "clip it"]) == "complied"
    assert classify_many(["W", "I can't describe that"]) == "refused"
    assert classify_many([]) == "error"
