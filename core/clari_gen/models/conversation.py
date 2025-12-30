"""Conversation state management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_message_dict(self) -> dict:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """Manages the conversation history for a query."""

    turns: List[ConversationTurn] = field(default_factory=list)

    def add_system_message(self, content: str):
        """Add a system message."""
        self.turns.append(ConversationTurn(role="system", content=content))

    def add_user_message(self, content: str):
        """Add a user message."""
        self.turns.append(ConversationTurn(role="user", content=content))

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.turns.append(ConversationTurn(role="assistant", content=content))

    def to_messages(self) -> List[dict]:
        """Convert conversation to OpenAI messages format."""
        return [turn.to_message_dict() for turn in self.turns]

    def clear(self):
        """Clear conversation history."""
        self.turns.clear()

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message content."""
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.content
        return None
