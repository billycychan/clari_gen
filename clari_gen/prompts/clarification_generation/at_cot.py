"""AT-CoT prompt for generating clarifying questions with ambiguity types and chain-of-thought reasoning."""

from ...models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ...models.structured_schemas import ClarificationResponse


class ClarificationATCoTPrompt:
    """Generates clarifying questions with ambiguity type guidance and chain-of-thought reasoning."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for AT-CoT clarification generation.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at analyzing ambiguous user queries and generating clarifying questions for an information-seeking system.

Here are the possible ambiguity types:

{ambiguity_definitions}

Your task (think step by step):
1. Analyze the given query and identify which ambiguity type(s) apply
2. Explain your reasoning: describe which types of ambiguity apply to the given query
3. Based on these ambiguity types, describe how you plan to clarify the original query
4. Generate ONE clear, simple clarifying question that addresses the MOST IMPORTANT missing information

Important:
- Avoid compound questions (don't use "or", "and" to ask multiple things)
- Make it natural and conversational
- Focus on the most critical ambiguity if multiple types are present
- If you cannot identify any ambiguity, use "NONE" as the ambiguity type

Output ONLY a valid JSON object with the following structure:
{{
    "original_query": "the original query text",
    "ambiguity_types": ["LEXICAL", "SEMANTIC"],
    "reasoning": "your explanation of which types of ambiguity apply and how you plan to clarify the query",
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

Before generating the clarifying question, provide a textual explanation of your reasoning about which types of ambiguity apply to the given query. Based on these ambiguity types, describe how you plan to clarify the original query.

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
                "content": ClarificationATCoTPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": ClarificationATCoTPrompt.create_user_prompt(query),
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
