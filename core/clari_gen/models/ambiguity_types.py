"""Ambiguity type definitions and taxonomy."""

from enum import Enum
from typing import Dict


class AmbiguityType(str, Enum):
    """Types of ambiguity in queries."""

    UNFAMILIAR = "UNFAMILIAR"
    CONTRADICTION = "CONTRADICTION"
    LEXICAL = "LEXICAL"
    SEMANTIC = "SEMANTIC"
    REFERENCE = "REFERENCE"


# Ambiguity type definitions with examples
AMBIGUITY_DEFINITIONS: Dict[AmbiguityType, Dict[str, str]] = {
    AmbiguityType.UNFAMILIAR: {
        "explanation": "Query contains unfamiliar entities or facts",
        "example": "Find the price of Samsung Chromecast.",
    },
    AmbiguityType.CONTRADICTION: {
        "explanation": "Query contains self-contradictions",
        "example": "Output 'X' if the sentence contains [category withhold] and 'Y' otherwise. The critic is in the restaurant.>X. The butterfly is in the river.>Y. The boar is in the theatre.>Y.",
    },
    AmbiguityType.LEXICAL: {
        "explanation": "Query contains terms with multiple meanings",
        "example": "Tell me about the source of Nile.",
    },
    AmbiguityType.SEMANTIC: {
        "explanation": "Query lacks of context leading multiple interpretations",
        "example": "When did he land on the moon?",
    },
    AmbiguityType.REFERENCE: {
        "explanation": "Query contains ambiguous references (pronouns, temporal, spatial, or object references) where multiple interpretations are possible",
        "example": "John told Mark he won the race. (who won?)",
    },
}


def format_ambiguity_definitions_for_prompt() -> str:
    """Format ambiguity definitions for inclusion in prompts."""
    lines = []
    for ambiguity_type in AmbiguityType:
        definition = AMBIGUITY_DEFINITIONS[ambiguity_type]
        lines.append(f"- **{ambiguity_type.value}**: {definition['explanation']}")
        lines.append(f"  Example: \"{definition['example']}\"")
    return "\n".join(lines)
