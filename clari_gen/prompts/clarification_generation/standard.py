"""Standard prompt for generating clarifying questions without ambiguity type definitions."""

from ...models.structured_schemas import ClarificationResponse


class ClarificationStandardPrompt:
    """Generates basic clarifying questions without ambiguity type guidance."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for standard clarification generation.

        Returns:
            System prompt string
        """
        return """You are an expert at analyzing ambiguous user queries and generating clarifying questions for an information-seeking system.

Your task:
Generate ONE clear, simple clarifying question that you think is most appropriate to gain a better understanding of the user's intent.

Important:
- Avoid compound questions (don't use "or", "and" to ask multiple things)
- Make it natural and conversational
- Focus on the most critical missing information

Output ONLY a valid JSON object with the following structure:
{
    "original_query": "the original query text",
    "ambiguity_types": ["NONE"],
    "reasoning": "brief explanation of your clarification approach",
    "clarifying_question": "your generated question"
}

Do not include any text outside the JSON object."""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for clarification generation.

        Args:
            query: The query to analyze

        Returns:
            Formatted user prompt
        """
        return f"""Given a query in an information-seeking system, generate a clarifying question that you think is most appropriate to gain a better understanding of the user's intent.

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
                "content": ClarificationStandardPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": ClarificationStandardPrompt.create_user_prompt(query),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            ClarificationResponse Pydantic model class
        """
        return ClarificationResponse

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
            parsed = ClarificationResponse.model_validate_json(response)
            return {
                "original_query": parsed.original_query,
                "ambiguity_types": parsed.ambiguity_types,
                "reasoning": parsed.reasoning,
                "clarifying_question": parsed.clarifying_question,
            }
        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
