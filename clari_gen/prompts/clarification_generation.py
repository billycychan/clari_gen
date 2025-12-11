"""Prompt for generating clarifying questions using the large model (70B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import ClarificationResponse


class ClarificationGenerationPrompt:
    """Generates prompts for creating clarifying questions."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for clarification generation.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at generating clarifying questions for ambiguous user queries in an information-seeking system.

Here are the possible ambiguity types:

{ambiguity_definitions}


Your task and think step by step:
1. Analyze the given query and its identified ambiguity type(s)
2. Explain your reasoning: describe how you plan to clarify the query with your question
3. Generate ONE clear, simple clarifying question that addresses the MOST IMPORTANT missing information

Important:
- Avoid compound questions (don't use "or", "and" to ask multiple things)
- Make it natural and conversational
- Focus on the most critical ambiguity if multiple types are present

Output ONLY a valid JSON object with the following structure:
{{
    "original_query": "the original query text",
    "ambiguity_types": ["LEXICAL", "SEMANTIC"],
    "reasoning": "your explanation of how the question will resolve the ambiguity",
    "clarifying_question": "your generated question"
}}

Do not include any text outside the JSON object."""

    @staticmethod
    def create_user_prompt(
        query: str, ambiguity_types: list[str], reasoning: str
    ) -> str:
        """Create the user prompt for clarification generation.

        Args:
            query: The ambiguous query
            ambiguity_types: The identified types of ambiguity (list)
            reasoning: Explanation of why the query is ambiguous

        Returns:
            Formatted user prompt
        """
        types_str = ", ".join(ambiguity_types)
        return f"""Query: "{query}"

Ambiguity Type(s): {types_str}

Reasoning: {reasoning}

Generate a clarifying question to resolve this ambiguity. Output as JSON only."""

    @staticmethod
    def create_messages(query: str, ambiguity_types: list[str], reasoning: str) -> list:
        """Create the full message list for the model.

        Args:
            query: The ambiguous query
            ambiguity_types: The identified types of ambiguity (list)
            reasoning: Explanation of why the query is ambiguous

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": ClarificationGenerationPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": ClarificationGenerationPrompt.create_user_prompt(
                    query, ambiguity_types, reasoning
                ),
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
            Dictionary with original_query, ambiguity_type, clarifying_question

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # With structured outputs, response should be valid JSON
            parsed = ClarificationResponse.model_validate_json(response)
            return {
                "original_query": parsed.original_query,
                "ambiguity_types": parsed.ambiguity_types,
                "reasoning": parsed.reasoning,
                "clarifying_question": parsed.clarifying_question,
            }
        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
