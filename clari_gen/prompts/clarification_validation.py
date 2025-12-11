"""Prompt for validating user clarifications using the large model (70B)."""

from ..models.structured_schemas import ValidationResponse


class ClarificationValidationPrompt:
    """Generates prompts for validating that a user's clarification resolves the ambiguity."""

    SYSTEM_PROMPT = """You are an expert at evaluating whether a user's clarification adequately resolves an ambiguity in their original query.

Your task and think step by step:
1. Review the original ambiguous query
2. Review the ambiguity type and clarifying question that was asked
3. Review the user's clarification response
4. Determine if the clarification provides sufficient information to resolve the PRIMARY ambiguity

Respond with a JSON object containing:
- is_valid: true or false
- explanation: a brief explanation (one sentence) of your decision"""

    @staticmethod
    def create_user_prompt(
        original_query: str,
        ambiguity_types: list[str],
        clarifying_question: str,
        user_clarification: str,
    ) -> str:
        """Create the user prompt for clarification validation.

        Args:
            original_query: The original ambiguous query
            ambiguity_types: The types of ambiguity identified (list)
            clarifying_question: The question asked to the user
            user_clarification: The user's response

        Returns:
            Formatted user prompt
        """
        types_str = ", ".join(ambiguity_types)
        return f"""Original Query: "{original_query}"

Ambiguity Type(s): {types_str}

Clarifying Question: "{clarifying_question}"

User's Clarification: "{user_clarification}"

Does this clarification adequately resolve the ambiguity? Respond with JSON containing is_valid and explanation."""

    @staticmethod
    def create_messages(
        original_query: str,
        ambiguity_type: str,
        clarifying_question: str,
        user_clarification: str,
    ) -> list:
        """Create the full message list for the model.

        Args:
            original_query: The original ambiguous query
            ambiguity_type: The type of ambiguity identified
            clarifying_question: The question asked to the user
            user_clarification: The user's response

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {"role": "system", "content": ClarificationValidationPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ClarificationValidationPrompt.create_user_prompt(
                    original_query,
                    ambiguity_type,
                    clarifying_question,
                    user_clarification,
                ),
            },
        ]

    @staticmethod
    def get_response_schema():
        """Get the Pydantic schema for structured output.

        Returns:
            ValidationResponse Pydantic model class
        """
        return ValidationResponse

    @staticmethod
    def parse_response(response: str) -> tuple[bool, str]:
        """Parse the model's structured JSON response.

        Args:
            response: The model's response text containing JSON

        Returns:
            Tuple of (is_valid, explanation)

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # With structured outputs, response should be valid JSON
            parsed = ValidationResponse.model_validate_json(response)
            return parsed.is_valid, parsed.explanation
        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
        else:
            raise ValueError(f"Could not parse validation response: {response}")
