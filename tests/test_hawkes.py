from __future__ import annotations

import math
import random

from backend.dynamics.hawkes import (
    HawkesProcess,
    active_chatters_for_viewers,
    scene_intensity,
)


def test_base_rate_and_hard_clamp() -> None:
    hawkes = HawkesProcess(active_chatters=40, intensity=0.3)
    assert math.isclose(hawkes.rate, 40 * 0.015 * 0.3)
    hawkes.set_context(intensity=5.0)
    assert hawkes.rate <= 8.0
    hawkes.excite(100.0)
    assert hawkes.rate == 8.0


def test_excitation_decays_over_playback_time() -> None:
    hawkes = HawkesProcess(active_chatters=40, intensity=0.3)
    hawkes.excite(0.35)
    expected = 0.35 * math.exp(-0.6)
    hawkes.advance_to(1.0)
    assert math.isclose(hawkes.excitation, expected, rel_tol=1e-6)


def test_events_are_ordered_and_respect_span() -> None:
    hawkes = HawkesProcess(active_chatters=80, intensity=5.0)
    events = hawkes.next_events(12, rng=random.Random(7), max_span_s=6.0)
    assert events
    assert events == sorted(events)
    assert all(0.0 <= event <= 6.0 for event in events)


def test_cooldown_after_sustained_max_rate() -> None:
    hawkes = HawkesProcess(active_chatters=80, intensity=5.0)
    hawkes.excite(100.0)
    hawkes.record_event()
    assert hawkes.rate == 8.0
    for step in range(1, 13):
        hawkes.advance_to(step * 0.3)
        hawkes.record_event()
    assert hawkes.cooldown_events >= 1
    assert hawkes.cooling_down
    assert hawkes.rate == 0.0


def test_scene_intensity_mapping() -> None:
    assert scene_intensity() >= 0.3
    assert scene_intensity(surprise=1.0, speech_energy=1.0, scene_change=True) <= 5.0
    assert scene_intensity(surprise=1.0) > scene_intensity(surprise=0.1)


def test_active_chatters_is_bounded() -> None:
    assert active_chatters_for_viewers(0) == 8
    assert active_chatters_for_viewers(1200) == 36
    assert active_chatters_for_viewers(1_000_000) == 80
