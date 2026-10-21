"""Perception layer: frames + audio -> SceneState / SceneEvent.

M1 implements the visual path: :class:`~backend.perception.frames.FileFrameSource`
extracts a frame at ``t_read``, the observer asks the multimodal model for a
strict ``SceneState``, and :class:`~backend.perception.timeline.SceneTimeline`
keeps the last 90 seconds for chat and debug consumers.
"""

from .frames import FileFrameSource, Frame, FrameError, FrameSource
from .observer import Observation, ObservationError, Observer, scene_from_payload
from .scene import SceneState
from .timeline import SceneTimeline, TimelineEntry

__all__ = [
    "FileFrameSource",
    "Frame",
    "FrameError",
    "FrameSource",
    "Observation",
    "ObservationError",
    "Observer",
    "SceneState",
    "SceneTimeline",
    "TimelineEntry",
    "scene_from_payload",
]
