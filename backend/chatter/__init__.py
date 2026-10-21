"""Chat generation, personas and output filtering (M2)."""

from .guard import MAX_WORDS, check, classify, dedupe_key, template_for, word_count
from .personas import CORE_ROSTER, Persona, PersonaRoster
from .service import BatchResult, ChatMessage, ChatterService

__all__ = [
    "MAX_WORDS",
    "BatchResult",
    "CORE_ROSTER",
    "ChatMessage",
    "ChatterService",
    "Persona",
    "PersonaRoster",
    "check",
    "classify",
    "dedupe_key",
    "template_for",
    "word_count",
]
