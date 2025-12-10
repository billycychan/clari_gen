"""Test suite for the ambiguity detection system."""

import pytest
from unittest.mock import Mock, patch

from clari_gen.models import Query, QueryStatus, AmbiguityType
from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.clients import SmallModelClient, LargeModelClient


class TestAmbiguityPipeline:
    """Test cases for the AmbiguityPipeline orchestration."""

    def test_non_ambiguous_query(self):
        """Test that a clear query passes through without clarification."""
        # Mock clients
        small_model = Mock(spec=SmallModelClient)
        small_model.detect_ambiguity.return_value = "NOT_AMBIGUOUS"

        large_model = Mock(spec=LargeModelClient)

        pipeline = AmbiguityPipeline(
            small_model_client=small_model,
            large_model_client=large_model,
        )

        result = pipeline.process_query("What is 2 + 2?")

        assert result.status == QueryStatus.COMPLETED
        assert result.is_ambiguous == False
        assert result.get_final_output() == "What is 2 + 2?"

        # Large model should not be called for non-ambiguous queries
        large_model.classify_ambiguity.assert_not_called()

    def test_ambiguous_query_without_callback(self):
        """Test that an ambiguous query stops at AWAITING_CLARIFICATION without callback."""
        # Mock clients
        small_model = Mock(spec=SmallModelClient)
        small_model.detect_ambiguity.return_value = "AMBIGUOUS"

        large_model = Mock(spec=LargeModelClient)
        large_model.classify_ambiguity.return_value = (
            "SEMANTIC\nThis query lacks context."
        )
        large_model.generate_clarifying_question.return_value = """
        {
            "original_query": "When did he land on the moon?",
            "ambiguity_type": "SEMANTIC",
            "clarifying_question": "Who are you referring to?"
        }
        """

        pipeline = AmbiguityPipeline(
            small_model_client=small_model,
            large_model_client=large_model,
        )

        result = pipeline.process_query("When did he land on the moon?")

        assert result.status == QueryStatus.AWAITING_CLARIFICATION
        assert result.is_ambiguous == True
        assert len(result.ambiguity_types) > 0
        assert AmbiguityType.SEMANTIC in result.ambiguity_types
        assert result.clarifying_question is not None
        assert result.reformulated_query is None

    def test_ambiguous_query_with_valid_clarification(self):
        """Test full pipeline with valid clarification."""
        # Mock clients
        small_model = Mock(spec=SmallModelClient)
        small_model.detect_ambiguity.return_value = "AMBIGUOUS"

        large_model = Mock(spec=LargeModelClient)
        large_model.classify_ambiguity.return_value = (
            "WHO\nMissing information about who the mother is."
        )
        large_model.generate_clarifying_question.return_value = """
        {
            "original_query": "Suggest me some gifts for my mother.",
            "ambiguity_type": "WHO",
            "clarifying_question": "What are your mother's interests or hobbies?"
        }
        """
        large_model.validate_clarification.return_value = (
            "VALID\nThe clarification provides specific information."
        )
        large_model.reformulate_query.return_value = "Suggest me some gifts for my mother who enjoys gardening and reading mystery novels."

        pipeline = AmbiguityPipeline(
            small_model_client=small_model,
            large_model_client=large_model,
        )

        # Mock clarification callback
        def mock_clarification(question):
            return "She enjoys gardening and reading mystery novels."

        result = pipeline.process_query(
            "Suggest me some gifts for my mother.",
            clarification_callback=mock_clarification,
        )

        assert result.status == QueryStatus.COMPLETED
        assert result.is_ambiguous == True
        assert AmbiguityType.WHO in result.ambiguity_types
        assert result.clarification_is_valid == True
        assert result.reformulated_query is not None
        assert "gardening" in result.get_final_output().lower()

    def test_invalid_clarification_retry(self):
        """Test that invalid clarifications trigger retry."""
        small_model = Mock(spec=SmallModelClient)
        small_model.detect_ambiguity.return_value = "AMBIGUOUS"

        large_model = Mock(spec=LargeModelClient)
        large_model.classify_ambiguity.return_value = (
            "WHEN\nMissing temporal information."
        )
        large_model.generate_clarifying_question.return_value = """
        {
            "original_query": "How many goals did Argentina score?",
            "ambiguity_type": "WHEN",
            "clarifying_question": "Which World Cup are you asking about?"
        }
        """
        # First validation: INVALID, Second: VALID
        large_model.validate_clarification.side_effect = [
            "INVALID\nToo vague.",
            "VALID\nSpecific information provided.",
        ]
        large_model.reformulate_query.return_value = (
            "How many goals did Argentina score in the 2022 FIFA World Cup?"
        )

        pipeline = AmbiguityPipeline(
            small_model_client=small_model,
            large_model_client=large_model,
            max_clarification_attempts=3,
        )

        # Mock clarification callback with different responses
        responses = ["Some World Cup", "The 2022 World Cup in Qatar"]
        response_iter = iter(responses)

        def mock_clarification(question):
            return next(response_iter)

        result = pipeline.process_query(
            "How many goals did Argentina score?",
            clarification_callback=mock_clarification,
        )

        assert result.status == QueryStatus.COMPLETED
        assert large_model.validate_clarification.call_count == 2


class TestPromptParsing:
    """Test cases for prompt response parsing."""

    def test_ambiguity_detection_parsing(self):
        """Test parsing of ambiguity detection responses."""
        from clari_gen.prompts import AmbiguityDetectionPrompt

        assert AmbiguityDetectionPrompt.parse_response("AMBIGUOUS") == True
        assert AmbiguityDetectionPrompt.parse_response("NOT_AMBIGUOUS") == False
        assert AmbiguityDetectionPrompt.parse_response("not ambiguous") == False

        with pytest.raises(ValueError):
            AmbiguityDetectionPrompt.parse_response("UNCLEAR")

    def test_classification_parsing(self):
        """Test parsing of classification responses."""
        from clari_gen.prompts import AmbiguityClassificationPrompt

        response = "LEXICAL\nThe term 'source' has multiple meanings."
        ambiguity_types, reasoning = AmbiguityClassificationPrompt.parse_response(
            response
        )

        assert ambiguity_types == ["LEXICAL"]
        assert "multiple meanings" in reasoning

        # Test multiple types
        response_multi = "LEXICAL, SEMANTIC\nMultiple issues present."
        ambiguity_types, reasoning = AmbiguityClassificationPrompt.parse_response(
            response_multi
        )

        assert ambiguity_types == ["LEXICAL", "SEMANTIC"]
        assert "Multiple issues" in reasoning

    def test_clarification_json_parsing(self):
        """Test parsing of JSON clarification responses."""
        from clari_gen.prompts import ClarificationGenerationPrompt

        response = """
        {
            "original_query": "Test query",
            "ambiguity_type": "WHO",
            "clarifying_question": "Who are you asking about?"
        }
        """

        data = ClarificationGenerationPrompt.parse_response(response)

        assert data["original_query"] == "Test query"
        assert data["ambiguity_type"] == "WHO"
        assert "Who" in data["clarifying_question"]

    def test_validation_parsing(self):
        """Test parsing of validation responses."""
        from clari_gen.prompts import ClarificationValidationPrompt

        response = "VALID\nProvides sufficient information."
        is_valid, explanation = ClarificationValidationPrompt.parse_response(response)

        assert is_valid == True
        assert "sufficient" in explanation

        response = "INVALID\nToo vague."
        is_valid, explanation = ClarificationValidationPrompt.parse_response(response)

        assert is_valid == False
        assert "vague" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
