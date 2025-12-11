"""Prompt for reformulating queries using the large model (70B)."""


class QueryReformulationPrompt:
    """Generates prompts for reformulating ambiguous queries after clarification."""

    SYSTEM_PROMPT = """You are an expert at reformulating ambiguous queries into clear, unambiguous versions.

Your task and think step by step:
1. Review the original ambiguous query
2. Review the ambiguity type and clarifying question
3. Review the user's clarification
4. Reformulate the query to be clear and unambiguous by incorporating the clarification

The reformulated query should:
- Incorporate the information from the user's clarification
- Be clear, specific, and unambiguous
- Maintain the original intent of the query
- Be a complete, standalone query (not requiring additional context)
- Be natural and well-formed

Respond with ONLY the reformulated query - no extra text, explanations, or formatting."""

    @staticmethod
    def create_user_prompt(
        original_query: str,
        ambiguity_types: list[str],
        clarifying_question: str,
        user_clarification: str,
    ) -> str:
        """Create the user prompt for query reformulation.

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

Reformulate the original query to be clear and unambiguous by incorporating the user's clarification. Output ONLY the reformulated query."""

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
            {"role": "system", "content": QueryReformulationPrompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QueryReformulationPrompt.create_user_prompt(
                    original_query,
                    ambiguity_type,
                    clarifying_question,
                    user_clarification,
                ),
            },
        ]

    @staticmethod
    def parse_response(response: str) -> str:
        """Parse the model's response to extract the reformulated query.

        Args:
            response: The model's response text

        Returns:
            The reformulated query string
        """
        # Remove any extra whitespace and quotes
        reformulated = response.strip()

        # Remove surrounding quotes if present
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]

        return reformulated
