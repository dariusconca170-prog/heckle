from __future__ import annotations

import pytest

from backend.clock import PlaybackClock


def test_first_report_and_lookahead() -> None:
    clock = PlaybackClock(lookahead_s=5.5, seek_threshold_s=1.25)
    update = clock.update(10.0, playing=True, now=100.0)
    assert update.seek is False
    assert update.discontinuity is False
    assert update.snapshot.t_play == pytest.approx(10.0)
    assert update.snapshot.t_read == pytest.approx(15.5)
    assert update.snapshot.generation == 0


def test_projection_between_reports() -> None:
    clock = PlaybackClock(lookahead_s=5.5)
    clock.update(10.0, playing=True, now=100.0)
    assert clock.snapshot(now=101.0).t_play == pytest.approx(11.0)
    clock.update(11.0, playing=False, now=101.0)
    assert clock.snapshot(now=110.0).t_play == pytest.approx(11.0)


def test_normal_4hz_reports_are_not_seeks() -> None:
    clock = PlaybackClock(lookahead_s=5.5)
    clock.update(10.0, playing=True, now=100.0)
    update = clock.update(10.25, playing=True, now=100.25)
    assert update.seek is False
    assert update.discontinuity is False
    assert update.snapshot.generation == 0


def test_jump_is_seek_and_bumps_generation() -> None:
    clock = PlaybackClock(lookahead_s=5.5, seek_threshold_s=1.25)
    clock.update(10.0, playing=True, now=100.0)
    update = clock.update(30.0, playing=True, now=100.5)
    assert update.seek is True
    assert update.discontinuity is True
    assert update.snapshot.generation == 1
    assert update.snapshot.t_read == pytest.approx(35.5)


def test_pause_and_resume_are_discontinuities() -> None:
    clock = PlaybackClock(lookahead_s=5.5)
    clock.update(10.0, playing=True, now=100.0)
    paused = clock.update(10.0, playing=False, now=101.0)
    assert paused.paused is True
    assert paused.discontinuity is True
    resumed = clock.update(10.0, playing=True, now=105.0)
    assert resumed.resumed is True
    assert resumed.discontinuity is True
    assert resumed.snapshot.generation == 2


def test_forced_seek_and_duration_clamp() -> None:
    clock = PlaybackClock(lookahead_s=5.5)
    clock.set_duration(60.0)
    clock.update(10.0, playing=True, now=100.0)
    forced = clock.update(10.0, playing=True, seek=True, now=100.1)
    assert forced.seek is True
    assert forced.snapshot.generation == 1
    update = clock.update(1000.0, playing=True, seek=True, now=100.2)
    assert update.snapshot.t_play == pytest.approx(60.0)
    assert update.snapshot.t_read == pytest.approx(60.0)


def test_reset_increments_generation_and_clears_playhead() -> None:
    clock = PlaybackClock()
    clock.update(12.0, playing=True, now=100.0)
    clock.reset(duration=30.0)
    assert clock.initialized is False
    assert clock.generation >= 1
    assert clock.snapshot(now=101.0).t_play == pytest.approx(0.0)


def test_invalid_values_rejected() -> None:
    clock = PlaybackClock()
    with pytest.raises(ValueError):
        PlaybackClock(lookahead_s=-1)
    with pytest.raises(ValueError):
        PlaybackClock(seek_threshold_s=0)
    with pytest.raises(ValueError):
        clock.update(float("nan"))
