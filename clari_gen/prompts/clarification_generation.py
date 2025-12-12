"""Prompt for generating clarifying questions using the large model (70B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import ClarificationResponse


class ClarificationGenerationPrompt:
    """Generates prompts for creating clarifying questions with embedded ambiguity classification."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for clarification generation with classification.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at analyzing ambiguous user queries and generating clarifying questions for an information-seeking system.

Here are the possible ambiguity types:

{ambiguity_definitions}

Your task (think step by step):
1. Analyze the given query and identify which ambiguity type(s) apply
2. Explain your reasoning: describe what makes the query ambiguous and how your question will resolve it
3. Generate ONE clear, simple clarifying question that addresses the MOST IMPORTANT missing information

Important:
- Avoid compound questions (don't use "or", "and" to ask multiple things)
- Make it natural and conversational
- Focus on the most critical ambiguity if multiple types are present
- If you cannot identify any ambiguity, use "NONE" as the ambiguity type

Output ONLY a valid JSON object with the following structure:
{{
    "original_query": "the original query text",
    "ambiguity_types": ["LEXICAL", "SEMANTIC"],
    "reasoning": "your explanation of what makes the query ambiguous and how the question will resolve it",
    "clarifying_question": "your generated question"
}}

Do not include any text outside the JSON object."""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for clarification generation with few-shot examples.

        Args:
            query: The query to analyze

        Returns:
            Formatted user prompt with examples
        """
        return f"""Analyze the query, identify its ambiguity type(s), and generate a clarifying question.

Examples:

Example 1:
Query: "Tell me about the source of Nile."
Output: {{
    "original_query": "Tell me about the source of Nile.",
    "ambiguity_types": ["LEXICAL"],
    "reasoning": "The word 'source' has multiple meanings: it could refer to the geographical source (where the river originates) or informational sources (books, articles about the Nile). This lexical ambiguity needs clarification.",
    "clarifying_question": "Are you asking about the geographical origin of the Nile River, or are you looking for informational sources about the Nile?"
}}

Example 2:
Query: "When did he land on the moon?"
Output: {{
    "original_query": "When did he land on the moon?",
    "ambiguity_types": ["REFERENCE"],
    "reasoning": "The pronoun 'he' is an ambiguous reference - it's unclear which person landed on the moon. Multiple astronauts have landed on the moon at different times.",
    "clarifying_question": "Which astronaut are you asking about? For example, Neil Armstrong, Buzz Aldrin, or another astronaut?"
}}

Example 3:
Query: "Find the price of Samsung Chromecast."
Output: {{
    "original_query": "Find the price of Samsung Chromecast.",
    "ambiguity_types": ["UNFAMILIAR"],
    "reasoning": "There is no such product as 'Samsung Chromecast'. Chromecast is made by Google, not Samsung. This appears to be an unfamiliar or non-existent entity.",
    "clarifying_question": "Did you mean the Google Chromecast, or are you looking for a Samsung streaming device?"
}}

Example 4:
Query: "John told Mark he won the race."
Output: {{
    "original_query": "John told Mark he won the race.",
    "ambiguity_types": ["REFERENCE"],
    "reasoning": "The pronoun 'he' could refer to either John or Mark, making it unclear who actually won the race.",
    "clarifying_question": "Who won the race - John or Mark?"
}}

Example 5:
Query: "What is the population of Tokyo in 2023?"
Output: {{
    "original_query": "What is the population of Tokyo in 2023?",
    "ambiguity_types": ["NONE"],
    "reasoning": "This query is clear and unambiguous. It specifies the location (Tokyo) and the time period (2023) with sufficient detail.",
    "clarifying_question": "This query is clear and does not require clarification."
}}

Now analyze this query:
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
                "content": ClarificationGenerationPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": ClarificationGenerationPrompt.create_user_prompt(query),
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
