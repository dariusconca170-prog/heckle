from __future__ import annotations

from backend.llm.profile import load_profile


def test_9b_profile_command() -> None:
    profile = load_profile("9b")
    command = profile.command("llama-server")
    assert "-np" in command and command[command.index("-np") + 1] == "1"
    assert "-ngl" in command and command[command.index("-ngl") + 1] == "99"
    assert "-ctk" in command and command[command.index("-ctk") + 1] == "q8_0"
    assert "--jinja" in command
    assert '{"enable_thinking":false}' in command
    assert profile.n_cpu_moe is None
    assert profile.lookahead_s == 5.5


def test_35b_profile_has_ncmoe() -> None:
    profile = load_profile("35b")
    command = profile.command("llama-server")
    assert "-ncmoe" in command
    assert command[command.index("-ncmoe") + 1] == "23"
    assert profile.lookahead_s == 9.0


def test_4b_profile_lookahead() -> None:
    # Updated from the 4B smoke: vision ~1.53 s + batch p95 ~2.41 s + 2 s.
    assert load_profile("4b").lookahead_s == 5.8
