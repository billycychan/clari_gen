"""Prompt for classifying ambiguity type using the small model (8B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import AmbiguityClassificationResponse

ambiguity_definitions = format_ambiguity_definitions_for_prompt()

output_format_sample = f"""
Output ONLY a valid JSON object:
{{
  "ambiguity_types": ["..."],
  "reasoning": "1–2 sentences: the main competing interpretations/facets and why the label(s) apply."
}}
"""

zero_shot_prompt = f"""
You are an expert at detecting ambiguity in user queries for an information-seeking search system.

Ambiguity types and their definitions:
{ambiguity_definitions}

Task:
Given a user query, determine whether it requires clarification.

Rules:
1. Use only the ambiguity types defined above.
2. If the query has exactly one clear, dominant interpretation, set "ambiguity_types" to ["NONE"].
3. If the query has multiple plausible interpretations,  assign the relevant ambiguity types from the definitions above and  set "ambiguity_types" to a list
"""


few_shot_prompt = f"""

{zero_shot_prompt}

### Example A1
Query: "Tell me about Mercury."
Output:
{{
  "ambiguity_types": ["LEXICAL"],
  "reasoning": "The term ‘Mercury’ could refer to the planet, the chemical element, or the mythological figure, resulting in multiple interpretations."
}}

### Example A2
Query: "What is the best way to get there quickly?"
Output:
{{
  "ambiguity_types": ["REFERENCE", "SEMANTIC"],
  "reasoning": "Neither the destination (‘there’) nor the mode of travel is specified, creating multiple plausible interpretations."
}}

### Example A3
Query: "Explain how to treat jaguar issues."
Output:
{{
  "ambiguity_types": ["LEXICAL", "UNFAMILIAR"],
  "reasoning": "‘Jaguar’ may refer to the animal, the automobile brand, or a software system. The phrase ‘issues’ is also undefined, making intent unclear."
}}

### Example C1
Query: "How do I reset a Cisco router to factory settings?"
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "The device type and task are explicit, allowing a direct procedural response."
}}

### Example C2
Query: "Convert 150 kilometers to miles."
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "A clear numerical conversion task with no alternative interpretations."
}}

### Example C3
Query: "List three renewable energy sources."
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "The category and requirement are explicit and permit a straightforward response."
}}

Now classify the next query.
"""

zero_shot_cot_prompt = f"""

{zero_shot_prompt}

Think step by step before answering:
1) Restate the most likely user intent.
2) Enumerate at least two plausible interpretations or facets (if they exist).
3) Map each interpretation to ambiguity type(s) using the definitions.
4) Decide CLEAR vs AMBIGUOUS (clarification needed) using: if 2+ plausible interpretations/facets lead to different answers/results, choose AMBIGUOUS.
5) Write a brief final reasoning (1–2 sentences) summarizing the decisive ambiguity.
"""

few_shot_cot_prompt = f"""

{zero_shot_cot_prompt}
Query: "Tell me about Mercury."
Reasoning:
- Interpret intent: User wants information about “Mercury.”
- Check ambiguity: Could mean planet, chemical element, or mythological figure.
- Decide: Clarification needed.
- Map to type(s): LEXICAL.
Output:
{{
  "ambiguity_types": ["LEXICAL"],
  "reasoning": "The term ‘Mercury’ could refer to the planet, the chemical element, or the mythological figure, resulting in multiple interpretations."
}}

### Example A2
Query: "What is the best way to get there quickly?"
Reasoning:
- Interpret intent: User wants guidance for reaching a destination.
- Check ambiguity: “There” may refer to a physical location, webpage, or conceptual goal.
- Decide: Clarification needed.
- Map to type(s): REFERENCE, SEMANTIC.
Output:
{{
  "ambiguity_types": ["REFERENCE", "SEMANTIC"],
  "reasoning": "Neither the destination (‘there’) nor the mode of travel is specified, creating multiple plausible interpretations."
}}

### Example A3
Query: "Explain how to treat jaguar issues."
Reasoning:
- Interpret intent: User wants help resolving “jaguar issues.”
- Check ambiguity: “Jaguar” could mean an animal, car brand, or software system.
- Decide: Clarification needed.
- Map to type(s): LEXICAL, UNFAMILIAR.
Output:
{{
  "ambiguity_types": ["LEXICAL", "UNFAMILIAR"],
  "reasoning": "‘Jaguar’ may refer to the animal, the automobile brand, or a software system. The phrase ‘issues’ is also undefined, making intent unclear."
}}

### Example C1
Query: "How do I reset a Cisco router to factory settings?"
Reasoning:
- Interpret intent: User wants instructions for factory-resetting a Cisco router.
- Check ambiguity: Model differences exist but do not block giving standard guidance.
- Decide: No clarification needed.
- Map to type(s): NONE.
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "The device type and task are explicit, allowing a direct procedural response."
}}

### Example C2
Query: "Convert 150 kilometers to miles."
Reasoning:
- Interpret intent: User wants a distance converted from kilometers to miles.
- Check ambiguity: None; single clear meaning.
- Decide: No clarification needed.
- Map to type(s): NONE.
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "A clear numerical conversion task with no alternative interpretations."
}}

### Example C3
Query: "List three renewable energy sources."
Reasoning:
- Interpret intent: User wants examples of renewable energy.
- Check ambiguity: Only one straightforward interpretation.
- Decide: No clarification needed.
- Map to type(s): NONE.
Output:
{{
  "ambiguity_types": ["NONE"],
  "reasoning": "The category and requirement are explicit and permit a straightforward response."
}}

Now classify the next query.
"""


class AmbiguityClassificationPrompt:
    """Generates prompts for classifying the type of ambiguity."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for ambiguity classification (Zero-Shot).

        Returns:
            System prompt string
        """
        return zero_shot_prompt

    @staticmethod
    def create_system_prompt_few_shot() -> str:
        """Create few-shot system prompt with examples.

        Returns:
            System prompt string with examples
        """
        return few_shot_prompt

    @staticmethod
    def create_system_prompt_zero_shot_cot() -> str:
        """Create Chain-of-Thought system prompt (zero-shot).

        Returns:
            System prompt string with CoT instructions
        """
        return zero_shot_cot_prompt

    @staticmethod
    def create_system_prompt_few_shot_cot() -> str:
        """Create few-shot Chain-of-Thought system prompt.

        Returns:
            System prompt string with few-shot CoT examples
        """
        return few_shot_cot_prompt

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for ambiguity classification.

        Args:
            query: The ambiguous query to classify

        Returns:
            Formatted user prompt
        """
        return f'Query: "{query}'

    @staticmethod
    def create_messages(query: str, strategy: str = "zero_shot_cot") -> list:
        """Create the full message list for the model.

        Args:
            query: The ambiguous query to classify
            strategy: Prompting strategy to use: "zero_shot", "few_shot", "zero_shot_cot", or "few_shot_cot"

        Returns:
            List of message dicts in OpenAI format
        """
        if strategy == "few_shot":
            system_prompt = (
                AmbiguityClassificationPrompt.create_system_prompt_few_shot()
            )
        elif strategy == "zero_shot":
            system_prompt = AmbiguityClassificationPrompt.create_system_prompt()
        elif strategy == "few_shot_cot":
            system_prompt = (
                AmbiguityClassificationPrompt.create_system_prompt_few_shot_cot()
            )
        else:  # "zero_shot_cot" (default)
            system_prompt = (
                AmbiguityClassificationPrompt.create_system_prompt_zero_shot_cot()
            )

        return [
            {
                "role": "system",
                "content": system_prompt + output_format_sample,
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
                "REFERENCE",
                "NONE",
            ]

            for ambiguity_type in parsed.ambiguity_types:
                if ambiguity_type not in valid_types:
                    raise ValueError(f"Invalid ambiguity type: {ambiguity_type}")

            return parsed.ambiguity_types, parsed.reasoning

        except Exception as e:
            raise ValueError(f"Could not parse structured response: {e}")
