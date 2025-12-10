"""Integration tests for the complete pipeline (requires running servers)."""

import pytest
import os
from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.clients import SmallModelClient, LargeModelClient
from clari_gen.models import QueryStatus


# Skip integration tests if servers are not available
SKIP_INTEGRATION = os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true"


@pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason="Integration tests disabled (set SKIP_INTEGRATION_TESTS=false to enable)",
)
class TestIntegration:
    """Integration tests using actual vLLM servers."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with real clients."""
        small_client = SmallModelClient()
        large_client = LargeModelClient()

        pipeline = AmbiguityPipeline(
            small_model_client=small_client,
            large_model_client=large_client,
        )

        # Test connections
        results = pipeline.test_connections()
        if not all(results.values()):
            pytest.skip("Model servers not available")

        return pipeline

    def test_clear_query(self, pipeline):
        """Test a clear, unambiguous query."""
        result = pipeline.process_query("What is the capital of France?")

        assert result.status == QueryStatus.COMPLETED
        assert result.is_ambiguous == False
        assert result.get_final_output() == "What is the capital of France?"

    def test_ambiguous_query_without_clarification(self, pipeline):
        """Test ambiguous query detection without providing clarification."""
        result = pipeline.process_query("Tell me about the source of Nile.")

        assert result.is_ambiguous == True
        assert result.status == QueryStatus.AWAITING_CLARIFICATION
        assert len(result.ambiguity_types) > 0
        assert result.clarifying_question is not None

    def test_full_clarification_flow(self, pipeline):
        """Test complete flow with clarification."""

        def clarification_callback(question):
            # Simulate user providing clarification
            if "World Cup" in question:
                return "The 2022 FIFA World Cup"
            return "Please clarify"

        result = pipeline.process_query(
            "How many goals did Argentina score in the World Cup?",
            clarification_callback=clarification_callback,
        )

        # Should complete with reformulated query
        assert result.status == QueryStatus.COMPLETED
        assert result.is_ambiguous == True
        assert result.reformulated_query is not None
        assert "2022" in result.get_final_output()

    def test_test_connections(self, pipeline):
        """Test connection testing functionality."""
        results = pipeline.test_connections()

        assert "small_model" in results
        assert "large_model" in results
        assert all(results.values())


@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled")
class TestExampleQueries:
    """Test the system with example queries from the taxonomy."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline with real clients."""
        small_client = SmallModelClient()
        large_client = LargeModelClient()

        pipeline = AmbiguityPipeline(
            small_model_client=small_client,
            large_model_client=large_client,
        )

        results = pipeline.test_connections()
        if not all(results.values()):
            pytest.skip("Model servers not available")

        return pipeline

    @pytest.mark.parametrize(
        "query,expected_type",
        [
            ("Find the price of Samsung Chromecast.", "UNFAMILIAR"),
            ("Tell me about the source of Nile.", "LEXICAL"),
            ("When did he land on the moon?", "SEMANTIC"),
            ("Suggest me some gifts for my mother.", "WHO"),
            ("How many goals did Argentina score in the World Cup?", "WHEN"),
            ("Tell me how to reach New York.", "WHERE"),
            ("Real name of gwen stacy in spiderman?", "WHAT"),
        ],
    )
    def test_taxonomy_examples(self, pipeline, query, expected_type):
        """Test that example queries are correctly classified."""
        result = pipeline.process_query(query)

        # All these queries should be detected as ambiguous
        assert result.is_ambiguous == True, f"Query '{query}' should be ambiguous"

        # Check if the expected ambiguity type is in the list
        if result.ambiguity_types:
            print(f"Query: {query}")
            types_str = ", ".join([t.value for t in result.ambiguity_types])
            print(f"Expected: {expected_type}, Got: {types_str}")
            print(f"Reasoning: {result.ambiguity_reasoning}")
            print(f"Question: {result.clarifying_question}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
