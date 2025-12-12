"""Client for the small model (Llama-3.1-8B) - ambiguity classification."""

from typing import List, Optional, Type
import logging
from pydantic import BaseModel

from .base_client import BaseVLLMClient

logger = logging.getLogger(__name__)


class SmallModelClient(BaseVLLMClient):
    """Client for the 8B model used for ambiguity classification."""

    # Default temperature for ambiguity classification (low-medium for consistent classification)
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self, base_url: str = "http://localhost:8368/v1", api_key: str = "token-abc123"
    ):
        """Initialize the small model client.

        Args:
            base_url: The base URL of the 8B model server
            api_key: API key for authentication
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
        )

    def classify_ambiguity(
        self, messages: List[dict], response_format: Optional[Type[BaseModel]] = None
    ) -> str:
        """Classify the type of ambiguity in a query (or return NONE if not ambiguous).

        Args:
            messages: List of message dicts with system and user prompts
            response_format: Optional Pydantic model for structured JSON output using guided_json

        Returns:
            The model's response (JSON string with ambiguity_types and reasoning)

        Raises:
            Exception: If the API call fails
        """
        logger.info("Classifying ambiguity with 8B model")

        response = self.generate(
            messages=messages,
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=0.95,
            response_format=response_format,
        )

        logger.info(f"Ambiguity classification response: {response}")
        return response
