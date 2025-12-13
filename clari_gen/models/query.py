"""Query data model."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime

from .ambiguity_types import AmbiguityType


class QueryStatus(str, Enum):
    """Status of a query in the pipeline."""

    INITIAL = "INITIAL"
    CHECKING_AMBIGUITY = "CHECKING_AMBIGUITY"
    NOT_AMBIGUOUS = "NOT_AMBIGUOUS"
    AMBIGUOUS = "AMBIGUOUS"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    CLARIFICATION_RECEIVED = "CLARIFICATION_RECEIVED"
    VALIDATING_CLARIFICATION = "VALIDATING_CLARIFICATION"
    CLARIFICATION_INVALID = "CLARIFICATION_INVALID"
    REFORMULATING = "REFORMULATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class Query:
    """Represents a user query and its processing state."""

    # Original query
    original_query: str

    # Status
    status: QueryStatus = QueryStatus.INITIAL

    # Ambiguity classification results
    is_ambiguous: Optional[bool] = None
    ambiguity_types: List[str] = field(default_factory=list)
    ambiguity_reasoning: Optional[str] = None

    # Clarification
    clarifying_question: Optional[str] = None
    user_clarification: Optional[str] = None
    clarification_is_valid: Optional[bool] = None
    clarification_validation_feedback: Optional[str] = None

    # Output
    reformulated_query: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert query to dictionary."""
        return {
            "original_query": self.original_query,
            "status": self.status.value,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_types": self.ambiguity_types,
            "ambiguity_reasoning": self.ambiguity_reasoning,
            "clarifying_question": self.clarifying_question,
            "user_clarification": self.user_clarification,
            "clarification_is_valid": self.clarification_is_valid,
            "clarification_validation_feedback": self.clarification_validation_feedback,
            "reformulated_query": self.reformulated_query,
            "created_at": self.created_at.isoformat(),
            "error_message": self.error_message,
        }

    def get_final_output(self) -> str:
        """Get the final output query (original if not ambiguous, reformulated if ambiguous)."""
        if not self.is_ambiguous:
            return self.original_query
        return self.reformulated_query or self.original_query
