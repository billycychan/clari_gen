"""Vanilla prompt for generating clarifying questions."""

from ...models.structured_schemas import VanillaClarificationResponse


class ClarificationVanillaPrompt:
    """Generates clarifying questions with ambiguity type guidance but without CoT reasoning."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for AT-standard clarification generation.

        Returns:
            System prompt string
        """
        return f"""You are an expert at analyzing ambiguous user queries and generating clarifying questions for an information-seeking system.

Output ONLY a valid JSON object with the following structure:
{{
    "original_query": "the original query text",
    "clarifying_question": "your generated question"
}}

Do not include any text outside the JSON object."""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for clarification generation.

        Args:
            query: The query to analyze

        Returns:
            Formatted user prompt
        """
        return f"""Given a query in an information-seeking system, generate a clarifying question that you think is most appropriate to gain a better understanding of the user's intent. The ambiguity of a query can be multifaceted, and there are multiple possible ambiguity types.

Query: "{query}"
Output:"""

    @staticmethod
    def create_messages(query: str) -> list:
        """Create the full message list for the model.

        Args:
            query: The query to analyze and generate clarification for

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": ClarificationVanillaPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": ClarificationVanillaPrompt.create_user_prompt(query),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            ClarificationResponse Pydantic model class
        """
        return VanillaClarificationResponse

    @staticmethod
    def parse_response(response: str) -> dict:
        """Parse the model's JSON response using Pydantic.

        Args:
            response: The model's response text containing JSON

        Returns:
            Dictionary with original_query, ambiguity_types, reasoning, clarifying_question

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            parsed = VanillaClarificationResponse.model_validate_json(response)
            return {
                "original_query": parsed.original_query,
                "clarifying_question": parsed.clarifying_question,
            }
        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
