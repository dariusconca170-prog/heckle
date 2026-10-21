"""Stream furniture that is not part of the chat scheduler (PLAN.md §3.1).

Polls, stream events, clips, microphone and screen capture live here. Each
feature owns its playback-time timers and degrades to a status/metrics field
when its optional dependency is missing.
"""

__all__: list[str] = []
