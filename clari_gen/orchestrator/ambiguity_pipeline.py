"""Main orchestration pipeline for ambiguity detection and clarification."""

import logging
from typing import Optional, Callable

from ..models import Query, QueryStatus, AmbiguityType
from ..clients import SmallModelClient, LargeModelClient
from ..prompts import (
    AmbiguityDetectionPrompt,
    AmbiguityClassificationPrompt,
    ClarificationGenerationPrompt,
    ClarificationValidationPrompt,
    QueryReformulationPrompt,
)

logger = logging.getLogger(__name__)


class AmbiguityPipeline:
    """Orchestrates the full pipeline for handling ambiguous queries."""

    def __init__(
        self,
        small_model_client: Optional[SmallModelClient] = None,
        large_model_client: Optional[LargeModelClient] = None,
        max_clarification_attempts: int = 3,
    ):
        """Initialize the ambiguity pipeline.

        Args:
            small_model_client: Client for the 7B model (ambiguity detection)
            large_model_client: Client for the 70B model (classification, clarification, etc.)
            max_clarification_attempts: Maximum number of times to ask for clarification
                                 most NOT_AMBIGUOUS predictions are verified by the large model.
        """
        # self.small_model = small_model_client or SmallModelClient()
        self.small_model = small_model_client or SmallModelClient()
        self.large_model = large_model_client or LargeModelClient()
        self.max_clarification_attempts = max_clarification_attempts

        logger.info(f"Initialized AmbiguityPipeline")

    def process_query(
        self,
        query_text: str,
        clarification_callback: Optional[Callable[[str], str]] = None,
    ) -> Query:
        """Process a query through the full pipeline.

        Args:
            query_text: The user's query to process
            clarification_callback: Optional function to call when clarification is needed.
                                   Should take clarifying_question as input and return user's response.
                                   If None, pipeline will stop at AWAITING_CLARIFICATION state.

        Returns:
            Query object with full processing state and final output
        """
        query = Query(original_query=query_text)
        logger.info(f"Processing query: {query_text[:50]}...")

        try:
            # Step 1: Detect if query is ambiguous
            query = self._detect_ambiguity(query)

            if not query.is_ambiguous:
                # Query is clear - return as-is
                query.status = QueryStatus.COMPLETED
                logger.info("Query is not ambiguous - returning original")
                return query

            # Query is ambiguous - proceed with classification
            query = self._classify_ambiguity(query)

            # Generate clarifying question
            query = self._generate_clarifying_question(query)

            # If no callback provided, return query in awaiting state
            if clarification_callback is None:
                query.status = QueryStatus.AWAITING_CLARIFICATION
                logger.info(
                    "No clarification callback - returning query in awaiting state"
                )
                return query

            # Attempt to get and validate clarification
            attempts = 0
            while attempts < self.max_clarification_attempts:
                attempts += 1
                logger.info(
                    f"Clarification attempt {attempts}/{self.max_clarification_attempts}"
                )

                # Get user's clarification
                user_clarification = clarification_callback(query.clarifying_question)
                query.user_clarification = user_clarification
                query.status = QueryStatus.CLARIFICATION_RECEIVED

                # Validate the clarification
                query = self._validate_clarification(query)

                if query.clarification_is_valid:
                    # Clarification is valid - reformulate query
                    query = self._reformulate_query(query)
                    query.status = QueryStatus.COMPLETED
                    logger.info("Query processing completed successfully")
                    return query
                else:
                    # Invalid clarification - ask again if attempts remain
                    if attempts < self.max_clarification_attempts:
                        logger.warning(
                            f"Clarification invalid: {query.clarification_validation_feedback}"
                        )
                        # Update clarifying question with feedback
                        query.clarifying_question = (
                            f"{query.clarifying_question}\n\n"
                            f"(Your previous answer was unclear: {query.clarification_validation_feedback}. "
                            f"Please try again.)"
                        )
                        query.status = QueryStatus.AWAITING_CLARIFICATION

            # Max attempts reached with invalid clarification
            query.status = QueryStatus.ERROR
            query.error_message = (
                "Maximum clarification attempts reached with invalid responses"
            )
            logger.error(query.error_message)
            return query

        except Exception as e:
            query.status = QueryStatus.ERROR
            query.error_message = str(e)
            logger.error(f"Error processing query: {e}", exc_info=True)
            return query

    def _detect_ambiguity(self, query: Query) -> Query:
        """Detect if the query is ambiguous using the small model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with ambiguity detection results
        """
        query.status = QueryStatus.CHECKING_AMBIGUITY
        logger.info("Step 1: Detecting ambiguity")

        messages = AmbiguityDetectionPrompt.create_messages(query.original_query)
        response = self.small_model.detect_ambiguity(
            messages,
            response_format=AmbiguityDetectionPrompt.get_response_schema(),
        )

        is_ambiguous = AmbiguityDetectionPrompt.parse_response(response)
        query.is_ambiguous = is_ambiguous

        if query.is_ambiguous:
            query.status = QueryStatus.AMBIGUOUS
            logger.info("Query detected as AMBIGUOUS")
        else:
            query.status = QueryStatus.NOT_AMBIGUOUS
            logger.info("Query detected as NOT_AMBIGUOUS")

        return query

    def _classify_ambiguity(self, query: Query) -> Query:
        """Classify the type of ambiguity using the large model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with ambiguity classification
        """
        logger.info("Step 2: Classifying ambiguity type(s)")

        messages = AmbiguityClassificationPrompt.create_messages(query.original_query)
        response = self.large_model.classify_ambiguity(
            messages,
            response_format=AmbiguityClassificationPrompt.get_response_schema(),
        )

        ambiguity_types_strs, reasoning = AmbiguityClassificationPrompt.parse_response(
            response
        )

        query.ambiguity_types = [AmbiguityType[t] for t in ambiguity_types_strs]
        query.ambiguity_reasoning = reasoning

        types_str = ", ".join(ambiguity_types_strs)
        logger.info(f"Ambiguity classified as: {types_str}")

        return query

    def _generate_clarifying_question(self, query: Query) -> Query:
        """Generate a clarifying question using the large model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with clarifying question
        """
        logger.info("Step 3: Generating clarifying question")

        ambiguity_types_strs = [t.value for t in query.ambiguity_types]
        messages = ClarificationGenerationPrompt.create_messages(
            query.original_query,
            ambiguity_types_strs,
            query.ambiguity_reasoning,
        )
        response = self.large_model.generate_clarifying_question(
            messages,
            response_format=ClarificationGenerationPrompt.get_response_schema(),
        )

        data = ClarificationGenerationPrompt.parse_response(response)
        query.clarifying_question = data["clarifying_question"]

        logger.info(f"Generated question: {query.clarifying_question}")

        return query

    def _validate_clarification(self, query: Query) -> Query:
        """Validate the user's clarification using the large model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with validation results
        """
        query.status = QueryStatus.VALIDATING_CLARIFICATION
        logger.info("Step 4: Validating user clarification")

        ambiguity_types_strs = [t.value for t in query.ambiguity_types]
        messages = ClarificationValidationPrompt.create_messages(
            query.original_query,
            ambiguity_types_strs,
            query.clarifying_question,
            query.user_clarification,
        )
        response = self.large_model.validate_clarification(
            messages,
            response_format=ClarificationValidationPrompt.get_response_schema(),
        )

        is_valid, explanation = ClarificationValidationPrompt.parse_response(response)

        query.clarification_is_valid = is_valid
        query.clarification_validation_feedback = explanation

        if is_valid:
            logger.info(f"Clarification VALID: {explanation}")
        else:
            query.status = QueryStatus.CLARIFICATION_INVALID
            logger.warning(f"Clarification INVALID: {explanation}")

        return query

    def _reformulate_query(self, query: Query) -> Query:
        """Reformulate the query based on the user's clarification.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with reformulated query
        """
        query.status = QueryStatus.REFORMULATING
        logger.info("Step 5: Reformulating query")

        ambiguity_types_strs = [t.value for t in query.ambiguity_types]
        messages = QueryReformulationPrompt.create_messages(
            query.original_query,
            ambiguity_types_strs,
            query.clarifying_question,
            query.user_clarification,
        )
        response = self.large_model.reformulate_query(messages)

        query.reformulated_query = QueryReformulationPrompt.parse_response(response)

        logger.info(f"Reformulated query: {query.reformulated_query}")

        return query

    def test_connections(self) -> dict:
        """Test connections to both model servers.

        Returns:
            Dictionary with connection test results
        """
        logger.info("Testing model server connections")

        results = {
            "small_model": self.small_model.test_connection(),
            "large_model": self.large_model.test_connection(),
        }

        if all(results.values()):
            logger.info("All model servers are accessible")
        else:
            logger.error(f"Connection test failed: {results}")

        return results
