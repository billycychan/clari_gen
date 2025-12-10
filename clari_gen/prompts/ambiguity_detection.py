"""Prompt for ambiguity detection using the small model (3B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import AmbiguityDetectionResponse


class AmbiguityDetectionPrompt:
    """Generates prompts for detecting if a query is ambiguous."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for ambiguity detection.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at analyzing user queries in an information-seeking system. 
Your task is to determine whether a query is ambiguous.

Here are the possible types of ambiguity to consider:
{ambiguity_definitions}

A query is AMBIGUOUS if it matches one or more ambiguity types.
A query is NOT_AMBIGUOUS only if it matches none.

Your task:
1. Think through the ambiguity types step by step.
2. Explicitly show your chain of thought as you evaluate the query.
3. Identify which ambiguity types (if any) apply.
4. After completing the reasoning process, produce the final output JSON:
- is_ambiguous: boolean (true if ambiguous, false if not)"""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for ambiguity detection.

        Args:
            query: The user's query to analyze

        Returns:
            Formatted user prompt
        """
        return f"""Query: "{query}"

Is this query ambiguous? Respond with a JSON object indicating whether it is ambiguous:"""

    @staticmethod
    def create_messages(query: str) -> list:
        """Create the full message list for the model.

        Args:
            query: The user's query to analyze

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": AmbiguityDetectionPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": AmbiguityDetectionPrompt.create_user_prompt(query),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            AmbiguityDetectionResponse Pydantic model class
        """
        return AmbiguityDetectionResponse

    @staticmethod
    def parse_response(response: str) -> bool:
        """Parse the model's structured JSON response.

        Args:
            response: The model's response text containing JSON

        Returns:
            True if ambiguous, False if not ambiguous

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # With structured outputs, response should be valid JSON
            parsed = AmbiguityDetectionResponse.model_validate_json(response)
            return parsed.is_ambiguous
        except Exception as e:
            raise ValueError(
                f"Could not parse ambiguity detection response: {response}. Error: {e}"
            )
