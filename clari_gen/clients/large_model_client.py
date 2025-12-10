"""Client for the large model (Llama-3.3-70B) - classification, clarification, validation, reformulation."""

from typing import List, Type
from pydantic import BaseModel
import logging

from .base_client import BaseVLLMClient

logger = logging.getLogger(__name__)


class LargeModelClient(BaseVLLMClient):
    """Client for the 70B model used for classification, clarification, validation, and reformulation."""

    # Temperature settings for different tasks
    CLASSIFICATION_TEMPERATURE = 0.3  # Low for consistent classification
    CLARIFICATION_TEMPERATURE = 0.7  # Higher for natural question generation
    VALIDATION_TEMPERATURE = 0.3  # Low for consistent validation
    REFORMULATION_TEMPERATURE = 0.7  # Higher for natural query reformulation

    DEFAULT_MAX_TOKENS = 512

    def __init__(
        self, base_url: str = "http://localhost:8369/v1", api_key: str = "token-abc123"
    ):
        """Initialize the large model client.

        Args:
            base_url: The base URL of the 70B model server
            api_key: API key for authentication
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name="nvidia/Llama-3.3-70B-Instruct-FP8",
        )

    def classify_ambiguity(
        self, messages: List[dict], response_format: Type[BaseModel] = None
    ) -> str:
        """Classify the type of ambiguity in a query.

        Args:
            messages: List of message dicts with system and user prompts
            response_format: Optional Pydantic model for structured output

        Returns:
            The model's response with ambiguity type and reasoning (as JSON if schema provided)

        Raises:
            Exception: If the API call fails
        """
        logger.info("Classifying ambiguity type with 70B model")

        response = self.generate(
            messages=messages,
            temperature=self.CLASSIFICATION_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=0.95,
            response_format=response_format,
        )

        logger.info(f"Classification response: {response[:100]}...")
        return response

    def generate_clarifying_question(
        self, messages: List[dict], response_format: Type[BaseModel] = None
    ) -> str:
        """Generate a clarifying question for an ambiguous query.

        Args:
            messages: List of message dicts with system and user prompts
            response_format: Optional Pydantic model for structured output

        Returns:
            The model's response with JSON containing clarifying question

        Raises:
            Exception: If the API call fails
        """
        logger.info("Generating clarifying question with 70B model")

        response = self.generate(
            messages=messages,
            temperature=self.CLARIFICATION_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=0.95,
            response_format=response_format,
        )

        logger.info(f"Clarification generation response: {response[:100]}...")
        return response

    def validate_clarification(
        self, messages: List[dict], response_format: Type[BaseModel] = None
    ) -> str:
        """Validate that a user's clarification resolves the ambiguity.

        Args:
            messages: List of message dicts with system and user prompts
            response_format: Optional Pydantic model for structured output

        Returns:
            The model's response with validity and explanation (as JSON if schema provided)

        Raises:
            Exception: If the API call fails
        """
        logger.info("Validating user clarification with 70B model")

        response = self.generate(
            messages=messages,
            temperature=self.VALIDATION_TEMPERATURE,
            max_tokens=256,
            top_p=0.95,
            response_format=response_format,
        )

        logger.info(f"Validation response: {response[:100]}...")
        return response

    def reformulate_query(self, messages: List[dict]) -> str:
        """Reformulate a query based on user clarification.

        Args:
            messages: List of message dicts with system and user prompts

        Returns:
            The reformulated query

        Raises:
            Exception: If the API call fails
        """
        logger.info("Reformulating query with 70B model")

        response = self.generate(
            messages=messages,
            temperature=self.REFORMULATION_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=0.95,
        )

        logger.info(f"Reformulation response: {response[:100]}...")
        return response
