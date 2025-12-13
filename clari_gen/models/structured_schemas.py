"""Pydantic schemas for structured outputs from vLLM models."""

from pydantic import BaseModel, Field
from typing import List


class BinaryDetectionResponse(BaseModel):
    """Schema for binary ambiguity detection response."""

    is_ambiguous: bool = Field(
        description="Whether the query is ambiguous (True) or clear (False)"
    )


class ClarificationResponse(BaseModel):
    """Schema for clarification generation response."""

    original_query: str = Field(description="The original query that was ambiguous")
    ambiguity_types: List[str] = Field(
        description="List of identified ambiguity types (e.g., ['LEXICAL', 'SEMANTIC'])"
    )
    reasoning: str = Field(
        description="Explanation of how the clarifying question will resolve the ambiguity"
    )
    clarifying_question: str = Field(
        description="The generated clarifying question to resolve the ambiguity"
    )


class VanillaClarificationResponse(BaseModel):
    """Schema for vanilla clarification generation response."""

    original_query: str = Field(description="The original query that was ambiguous")
    clarifying_question: str = Field(
        description="The generated clarifying question to resolve the ambiguity"
    )


class AmbiguityClassificationResponse(BaseModel):
    """Schema for ambiguity classification response."""

    ambiguity_types: List[str] = Field(
        description="List of identified ambiguity types (e.g., ['LEXICAL', 'SEMANTIC'])"
    )
    reasoning: str = Field(
        description="Explanation of why these ambiguity types were identified"
    )


class ValidationResponse(BaseModel):
    """Schema for clarification validation response."""

    is_valid: bool = Field(
        description="Whether the clarification is valid and resolves the ambiguity"
    )
    explanation: str = Field(
        description="Explanation of why the clarification is valid or invalid"
    )
