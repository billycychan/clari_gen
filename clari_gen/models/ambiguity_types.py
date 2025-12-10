"""Ambiguity type definitions and taxonomy."""

from enum import Enum
from typing import Dict


class AmbiguityType(str, Enum):
    """Types of ambiguity in queries."""

    UNFAMILIAR = "UNFAMILIAR"
    CONTRADICTION = "CONTRADICTION"
    LEXICAL = "LEXICAL"
    SEMANTIC = "SEMANTIC"
    WHO = "WHO"
    WHEN = "WHEN"
    WHERE = "WHERE"
    WHAT = "WHAT"
    NONE = "NONE"  # Query is not ambiguous


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
    AmbiguityType.WHO: {
        "explanation": "Query output contains confusion due to missing personal elements",
        "example": "Suggest me some gifts for my mother.",
    },
    AmbiguityType.WHEN: {
        "explanation": "Query output contains confusion due to missing temporal elements",
        "example": "How many goals did Argentina score in the World Cup?",
    },
    AmbiguityType.WHERE: {
        "explanation": "Query output contains confusion due to missing spatial elements",
        "example": "Tell me how to reach New York.",
    },
    AmbiguityType.WHAT: {
        "explanation": "Query output contains confusion due to missing task-specific elements",
        "example": "Real name of gwen stacy in spiderman?",
    },
}


def format_ambiguity_definitions_for_prompt() -> str:
    """Format ambiguity definitions for inclusion in prompts."""
    lines = []
    for ambiguity_type in AmbiguityType:
        if ambiguity_type == AmbiguityType.NONE:
            continue
        definition = AMBIGUITY_DEFINITIONS[ambiguity_type]
        lines.append(f"- **{ambiguity_type.value}**: {definition['explanation']}")
        lines.append(f"  Example: \"{definition['example']}\"")
    return "\n".join(lines)
