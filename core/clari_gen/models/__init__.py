"""Data models for the ambiguity detection system."""

from .ambiguity_types import AmbiguityType, AMBIGUITY_DEFINITIONS
from .query import Query, QueryStatus
from .conversation import Conversation, ConversationTurn

__all__ = [
    "AmbiguityType",
    "AMBIGUITY_DEFINITIONS",
    "Query",
    "QueryStatus",
    "Conversation",
    "ConversationTurn",
]
