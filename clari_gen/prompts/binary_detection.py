"""Prompt for binary ambiguity detection using the small model (8B)."""

from ..models.structured_schemas import BinaryDetectionResponse


class BinaryDetectionPrompt:
    """Generates prompts for binary ambiguity detection."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for binary ambiguity detection.

        Returns:
            System prompt string
        """
        return """

You are an expert at detecting ambiguity in user queries for an information-seeking system.

Determine whether the user's query is ambiguous.

Definition:
- A query is clear if a single, well-defined interpretation is the most reasonable, and an answer can be provided without making important assumptions.
- A query is ambiguous if there are two or more distinct, reasonable interpretations that would lead to materially different answers or retrieval results, and the query does not provide enough constraints to choose among them.
- Treat missing critical constraints (e.g., which entity, timeframe, location, version/edition, metric/units, or sense of a polysemous term) as ambiguity when the omission could change the answer.

Decision rule:
- If multiple interpretations are plausible, set "is_ambiguous" to true.
- If one interpretation is clearly dominant and alternatives are unlikely or would not change the answer, set it to false.

Output format:
Return ONLY a valid JSON object exactly in this format:
{
  "is_ambiguous": true or false
}
Do not include any text outside the JSON.
"""

    @staticmethod
    def create_user_prompt_zero_shot(query: str) -> str:
        """Create the user prompt for binary ambiguity detection without examples (zero-shot).

        Args:
            query: The query to analyze

        Returns:
            Formatted user prompt without examples
        """
        return f"""Analyze the following query and determine if it is ambiguous or clear.

Query: "{query}"
Output:"""

    @staticmethod
    def create_user_prompt_few_shot(query: str) -> str:
        """Create the user prompt for binary ambiguity detection with few-shot examples.

        Args:
            query: The query to analyze

        Returns:
            Formatted user prompt with examples
        """
        return f"""Analyze the following query and determine if it is ambiguous or clear.

Examples:

Example 1:
Query: "What is the capital of France?"
Output: {{"is_ambiguous": false}}

Example 2:
Query: "Tell me about the source of Nile."
Output: {{"is_ambiguous": true}}
(Reason: "source" could mean geographical origin or informational source)

Example 3:
Query: "When did he land on the moon?"
Output: {{"is_ambiguous": true}}
(Reason: "he" is an ambiguous reference - which person?)

Example 4:
Query: "Find the price of Samsung Chromecast."
Output: {{"is_ambiguous": true}}
(Reason: Samsung doesn't make Chromecast - unfamiliar/incorrect entity)

Example 5:
Query: "What is the population of Tokyo in 2023?"
Output: {{"is_ambiguous": false}}

Example 6:
Query: "John told Mark he won the race."
Output: {{"is_ambiguous": true}}
(Reason: "he" could refer to John or Mark)

Now analyze this query:
Query: "{query}"
Output:"""

    @staticmethod
    def create_user_prompt(query: str, strategy: str = "zero_shot") -> str:
        """Create the user prompt for binary ambiguity detection.

        Args:
            query: The query to analyze
            strategy: Prompting strategy - "zero_shot" or "few_shot" (default: "few_shot")

        Returns:
            Formatted user prompt
        """
        if strategy == "zero_shot":
            return BinaryDetectionPrompt.create_user_prompt_zero_shot(query)
        elif strategy == "few_shot":
            return BinaryDetectionPrompt.create_user_prompt_few_shot(query)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Use 'zero_shot' or 'few_shot'."
            )

    @staticmethod
    def create_messages(query: str, strategy: str = "few_shot") -> list:
        """Create the full message list for the model.

        Args:
            query: The query to analyze
            strategy: Prompting strategy - "zero_shot" or "few_shot" (default: "few_shot")

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": BinaryDetectionPrompt.create_system_prompt(),
            },
            {
                "role": "user",
                "content": BinaryDetectionPrompt.create_user_prompt(query, strategy),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            BinaryDetectionResponse Pydantic model class
        """
        return BinaryDetectionResponse

    @staticmethod
    def parse_response(response: str) -> dict:
        """Parse the model's JSON response using Pydantic.

        Args:
            response: The model's response text containing JSON

        Returns:
            Dictionary with is_ambiguous

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # With structured outputs, response should be valid JSON
            parsed = BinaryDetectionResponse.model_validate_json(response)
            return {
                "is_ambiguous": parsed.is_ambiguous,
            }
        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
