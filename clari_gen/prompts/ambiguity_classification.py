"""Prompt for classifying ambiguity type using the large model (70B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import AmbiguityClassificationResponse


class AmbiguityClassificationPrompt:
    """Generates prompts for classifying the type of ambiguity."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for ambiguity classification.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at analyzing ambiguous queries in an information-seeking system. Your task is to identify and classify the type of ambiguity in a given query.

Here are the possible ambiguity types:

{ambiguity_definitions}

Your task:
1. Analyze the given query carefully
2. Identify which type(s) of ambiguity apply (there may be multiple types)
3. Provide a clear textual explanation of your reasoning
4. Output your analysis in the specified JSON format

You must respond with a JSON object containing:
- ambiguity_types: a list of applicable ambiguity type strings (e.g., ["LEXICAL", "SEMANTIC"])
- reasoning: your explanation of why these types apply"""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for ambiguity classification.

        Args:
            query: The ambiguous query to classify

        Returns:
            Formatted user prompt
        """
        return f"""Query: "{query}"

What type(s) of ambiguity are present in this query? Respond with a JSON object containing the ambiguity_types list and your reasoning."""

    @staticmethod
    def create_messages(query: str) -> list:
        """Create the full message list for the model.

        Args:
            query: The ambiguous query to classify

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": AmbiguityClassificationPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": AmbiguityClassificationPrompt.create_user_prompt(query),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            AmbiguityClassificationResponse Pydantic model class
        """
        return AmbiguityClassificationResponse

    @staticmethod
    def parse_response(response: str) -> tuple[list[str], str]:
        """Parse the model's structured JSON response.

        Args:
            response: The model's response text containing JSON

        Returns:
            Tuple of (ambiguity_types_list, reasoning)

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # With structured outputs, response should be valid JSON
            parsed = AmbiguityClassificationResponse.model_validate_json(response)

            # Validate types
            valid_types = [
                "UNFAMILIAR",
                "CONTRADICTION",
                "LEXICAL",
                "SEMANTIC",
                "WHO",
                "WHEN",
                "WHERE",
                "WHAT",
            ]

            for ambiguity_type in parsed.ambiguity_types:
                if ambiguity_type not in valid_types:
                    raise ValueError(f"Invalid ambiguity type: {ambiguity_type}")

            return parsed.ambiguity_types, parsed.reasoning

        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
