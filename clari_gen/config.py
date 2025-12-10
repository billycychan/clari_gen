"""Configuration management for the ambiguity detection system."""

import os
from dataclasses import dataclass
from .utils.logger import setup_logger


@dataclass
class ModelConfig:
    """Configuration for model servers."""

    # Small model (8B) - ambiguity detection
    small_model_base_url: str = "http://localhost:8368/v1"
    small_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Large model (70B) - classification, clarification, validation, reformulation
    large_model_base_url: str = "http://localhost:8369/v1"
    large_model_name: str = "nvidia/Llama-3.3-70B-Instruct-FP8"

    # API authentication
    api_key: str = "token-abc123"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            small_model_base_url=os.getenv("SMALL_MODEL_URL", cls.small_model_base_url),
            small_model_name=os.getenv("SMALL_MODEL_NAME", cls.small_model_name),
            large_model_base_url=os.getenv("LARGE_MODEL_URL", cls.large_model_base_url),
            large_model_name=os.getenv("LARGE_MODEL_NAME", cls.large_model_name),
            api_key=os.getenv("VLLM_API_KEY", cls.api_key),
        )


@dataclass
class PipelineConfig:
    """Configuration for the pipeline behavior."""

    max_clarification_attempts: int = 3
    log_level: str = "INFO"
    log_file: str = None

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            max_clarification_attempts=int(
                os.getenv("MAX_CLARIFICATION_ATTEMPTS", cls.max_clarification_attempts)
            ),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_file=os.getenv("LOG_FILE", cls.log_file),
        )


class Config:
    """Main configuration class combining all configs."""

    def __init__(self):
        self.model = ModelConfig.from_env()
        self.pipeline = PipelineConfig.from_env()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup and return configured logger."""
        return setup_logger(
            name="clari_gen",
            level=self.pipeline.log_level,
            log_file=self.pipeline.log_file,
        )

    @classmethod
    def default(cls):
        """Get default configuration."""
        return cls()
