"""Prompt for classifying ambiguity type using the small model (8B)."""

from ..models.ambiguity_types import format_ambiguity_definitions_for_prompt
from ..models.structured_schemas import AmbiguityClassificationResponse


class AmbiguityClassificationPrompt:
    """Generates prompts for classifying the type of ambiguity."""

    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for ambiguity classification.

        Returns:
            System prompt string
        """
        ambiguity_definitions = format_ambiguity_definitions_for_prompt()

        return f"""You are an expert at analyzing queries in an information-seeking system. Your task is to identify and classify the type of ambiguity in a given query.


Here are the possible ambiguity types:


{ambiguity_definitions}


**IMPORTANT**: 
- Only flag ambiguities that are REAL and MEANINGFUL - ambiguities that would prevent accurate interpretation or significantly change the answer
- Minor or insignificant ambiguities should NOT be flagged
- If examples are provided in the query (e.g., "such as X or Y"), the meaning is clarified - do NOT flag as LEXICAL
- Broad terms with commonly accepted meanings (e.g., "older adults" = 65+, "seniors") are NOT ambiguous unless specific precision is required
- If the query is reasonably clear despite minor potential ambiguities, return ["NONE"]
- Default to ["NONE"] unless you find a genuine, substantial ambiguity


**Examples of CLEAR queries (should return ["NONE"]):**
- "What are the evidence-based balance training programs for preventing falls in community-dwelling older adults aged 65 and above?" (specific population, intervention, and setting)
- "What are the non-surgical, evidence-based treatments for chronic low back pain in adults?" (clear population and intervention type)
- "What is the recommended dosage of vitamin D supplementation to reduce fall incidence in postmenopausal women?" (specific intervention, population, and outcome)
- "Which strength training exercises targeting lower extremity muscles are most effective in reducing falls among frail elderly persons?" (specific intervention and population)
- "What are the classic and atypical symptoms of a myocardial infarction in women over the age of 50?" (specific condition, population, and outcomes)


**Examples of AMBIGUOUS queries (one per type):**
- "Where can I find a good screen for eye health?" (LEXICAL - computer monitor vs screening test)
- "How fast should I run to lose weight?" (SEMANTIC - speed? frequency? duration? intensity zone?)
- "I bought a blood pressure cuff and a heart rate monitor. It said my resting heart rate was 95." (REFERENCE - which device is 'it'?)
- "Find the price of Samsung Chromecast." (UNFAMILIAR - Samsung doesn't make Chromecast, Google does)
- "List the best high-fiber foods that are completely free of carbohydrates." (CONTRADICTION - fiber IS a carbohydrate)


Your task are as follows, think step by step, 
1. Analyze the given query carefully
2. Check if the query matches any of the ambiguity types above
3. If it matches one or more types, list them; if it matches none, return ["NONE"]
4. Provide a brief (1-2 sentence) explanation of your reasoning
5. Output your analysis in the specified JSON format


You must respond with a JSON object containing:
- ambiguity_types: a list of applicable ambiguity type strings (e.g., ["LEXICAL", "SEMANTIC"]) or ["NONE"] if not ambiguous
- reasoning: a concise (1-2 sentence) explanation of why these types apply or why the query is clear"""

    @staticmethod
    def create_user_prompt(query: str) -> str:
        """Create the user prompt for ambiguity classification.

        Args:
            query: The ambiguous query to classify

        Returns:
            Formatted user prompt
        """
        return f"""Query: "{query}"

What type(s) of ambiguity are present in this query? If the query is clear and unambiguous, return ["NONE"]. Respond with a JSON object containing the ambiguity_types list and your reasoning."""

    @staticmethod
    def create_messages(query: str) -> list:
        """Create the full message list for the model.

        Args:
            query: The ambiguous query to classify

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": AmbiguityClassificationPrompt.create_system_prompt(),
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
