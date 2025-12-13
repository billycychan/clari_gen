"""Base client for vLLM OpenAI-compatible API."""

from typing import List, Optional, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseVLLMClient:
    """Base client for interacting with vLLM servers via OpenAI-compatible API."""

    def __init__(
        self, base_url: str, api_key: str = "token-abc123", model_name: str = ""
    ):
        """Initialize the vLLM client.

        Args:
            base_url: The base URL of the vLLM server (e.g., "http://localhost:8368/v1")
            api_key: API key for authentication (default: "token-abc123")
            model_name: Name of the model being served
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized vLLM client for {model_name} at {base_url}")

    def generate(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> str:
        """Generate a completion from the model.

        Args:
            messages: List of message dicts in OpenAI format
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Optional list of stop sequences
            response_format: Optional Pydantic model for structured JSON output.
                           Uses vLLM's guided_json for guaranteed schema compliance.

        Returns:
            The generated text content

        Raises:
            Exception: If the API call fails
        """
        try:
            logger.debug(
                f"Generating with {self.model_name}: temp={temperature}, max_tokens={max_tokens}"
            )

            # Build kwargs for API call
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            if stop is not None:
                api_kwargs["stop"] = stop

            # Add guided_json for structured output if provided
            if response_format is not None:
                json_schema = response_format.model_json_schema()
                api_kwargs["extra_body"] = {"guided_json": json_schema}
                logger.debug(
                    f"Using vLLM guided_json for schema: {response_format.__name__}"
                )
                logger.debug(f"JSON Schema: {json_schema}")

            response = self.client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content
            logger.debug(f"Generated {len(content)} characters from {self.model_name}")

            return content

        except Exception as e:
            logger.error(f"Error generating from {self.model_name}: {e}")
            raise

    def generate_structured(
        self,
        messages: List[dict],
        response_format: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> T:
        """Generate a structured completion from the model using vLLM's guided_json.

        Args:
            messages: List of message dicts in OpenAI format
            response_format: Pydantic model class for structured output
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Optional list of stop sequences

        Returns:
            Parsed Pydantic model instance

        Raises:
            Exception: If the API call or parsing fails
        """
        content = self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
        )

        # Parse the JSON response into the Pydantic model
        return response_format.model_validate_json(content)

    def test_connection(self) -> bool:
        """Test if the server is accessible.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            models = self.client.models.list()
            logger.info(f"Successfully connected to {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.base_url}: {e}")
            return False
