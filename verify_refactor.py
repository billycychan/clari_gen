
import unittest
from unittest.mock import MagicMock, patch
from core.clari_gen.orchestrator.ambiguity_pipeline import AmbiguityPipeline
from core.clari_gen.config import Config, AppConfig, PipelineConfig, ModelConfig

class TestPipelineRefactor(unittest.TestCase):
    def test_init_with_defaults(self):
        """Test initialization with default config (from env/defaults)"""
        print("Testing initialization with default config...")
        pipeline = AmbiguityPipeline()
        self.assertIsInstance(pipeline.config, Config)
        self.assertEqual(pipeline.clarification_strategy, "at_standard") # Default
        print("Default config init verified.")

    def test_init_with_explicit_config(self):
        """Test initialization with a custom config object"""
        print("Testing initialization with explicit config...")
        mock_config = MagicMock(spec=Config)
        
        # Setup nested mocks
        mock_config.pipeline = MagicMock()
        mock_config.model = MagicMock()
        
        mock_config.pipeline.max_clarification_attempts = 5
        mock_config.pipeline.clarification_strategy = "at_cot"
        mock_config.model.small_model_base_url = "http://mock-small"
        mock_config.model.small_model_name = "mock-small-model"
        mock_config.model.large_model_base_url = "http://mock-large"
        mock_config.model.large_model_name = "mock-large-model"
        mock_config.model.api_key = "mock-key" # Add api_key
        
        pipeline = AmbiguityPipeline(config=mock_config)
        mock_config.pipeline.clarification_strategy = "at_cot"
        mock_config.model.small_model_base_url = "http://mock-small"
        mock_config.model.large_model_base_url = "http://mock-large"
        
        pipeline = AmbiguityPipeline(config=mock_config)
        
        self.assertEqual(pipeline.config, mock_config)
        self.assertEqual(pipeline.max_clarification_attempts, 5)
        self.assertEqual(pipeline.clarification_strategy, "at_cot")
        # Check clients used config values (indirectly, by checking attributes if possible, or mocking clients)
        # Since clients are re-instantiated if not passed, we trust they used config.model values 
        # (verification: check if we can inspect client base_url)
        self.assertEqual(pipeline.small_model.base_url, "http://mock-small")
        print("Explicit config init verified.")

    def test_init_with_overrides(self):
        """Test that explicit arguments override config"""
        print("Testing initialization with overrides...")
        mock_config = MagicMock(spec=Config)
        # Setup nested mocks even if not used, to avoid errors if code accesses them
        mock_config.pipeline = MagicMock()
        mock_config.model = MagicMock()
        # Ensure model config has necessary attrs to avoid AttributeError when accessing model_config.*
        mock_config.model.small_model_base_url = "http://default"
        mock_config.model.small_model_name = "default"
        mock_config.model.large_model_base_url = "http://default"
        mock_config.model.large_model_name = "default"
        mock_config.model.api_key = "default"
        
        mock_config.pipeline.max_clarification_attempts = 3
        
        pipeline = AmbiguityPipeline(
            config=mock_config,
            max_clarification_attempts=10,
            clarification_strategy="vanilla"
        )
        
        self.assertEqual(pipeline.max_clarification_attempts, 10)
        self.assertEqual(pipeline.clarification_strategy, "vanilla")
        print("Overrides verified.")

if __name__ == "__main__":
    unittest.main()
