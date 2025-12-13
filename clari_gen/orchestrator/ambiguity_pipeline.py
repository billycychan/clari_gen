"""Main orchestration pipeline for ambiguity detection and clarification."""

import logging
from typing import Optional, Callable

from ..models import Query, QueryStatus, AmbiguityType
from ..clients import SmallModelClient, LargeModelClient
from ..prompts import (
    BinaryDetectionPrompt,
    ClarificationATStandardPrompt,
    ClarificationATCoTPrompt,
    QueryReformulationPrompt,
)
from ..prompts.clarification_generation.vanilla import ClarificationVanillaPrompt

logger = logging.getLogger(__name__)


class AmbiguityPipeline:
    """Orchestrates the full pipeline for handling ambiguous queries."""

    def __init__(
        self,
        small_model_client: Optional[SmallModelClient] = None,
        large_model_client: Optional[LargeModelClient] = None,
        max_clarification_attempts: int = 3,
        clarification_strategy: str = "at_standard",
    ):
        """Initialize the ambiguity pipeline.

        Args:
            small_model_client: Client for the 8B model (binary ambiguity detection)
            large_model_client: Client for the 70B model (clarification with classification, validation, reformulation)
            max_clarification_attempts: Maximum number of times to ask for clarification
            clarification_strategy: Strategy for clarification generation ("at_standard" or "at_cot")
        """
        self.small_model = small_model_client or SmallModelClient()
        self.large_model = large_model_client or LargeModelClient()
        self.max_clarification_attempts = max_clarification_attempts
        self.clarification_strategy = clarification_strategy

        # Select the appropriate prompt class based on strategy
        if clarification_strategy == "at_cot":
            self.clarification_prompt_class = ClarificationATCoTPrompt
        elif clarification_strategy == "at_standard":
            self.clarification_prompt_class = ClarificationATStandardPrompt
        elif clarification_strategy == "vanilla":
            self.clarification_prompt_class = ClarificationVanillaPrompt
        else:
            raise ValueError(
                f"Invalid clarification strategy: {clarification_strategy}. Must be 'at_standard', 'at_cot', or 'vanilla'"
            )

        logger.info(
            f"Initialized AmbiguityPipeline with clarification strategy: {clarification_strategy}"
        )

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
            # Step 1: Binary ambiguity detection
            query = self._detect_binary_ambiguity(query)

            # Check if query is not ambiguous
            if not query.is_ambiguous:
                # Query is clear - return as-is
                query.status = QueryStatus.COMPLETED
                logger.info("Query is not ambiguous - returning original")
                return query

            # Query is ambiguous - proceed to clarification generation
            query.status = QueryStatus.AMBIGUOUS

            # Step 2: Generate clarifying question with embedded classification
            query = self._generate_clarifying_question(query)

            # If no callback provided, return query in awaiting state
            if clarification_callback is None:
                query.status = QueryStatus.AWAITING_CLARIFICATION
                logger.info(
                    "No clarification callback - returning query in awaiting state"
                )
                return query

            # Attempt to get clarification
            # Get user's clarification
            user_clarification = clarification_callback(query.clarifying_question)
            query.user_clarification = user_clarification
            query.status = QueryStatus.CLARIFICATION_RECEIVED

            # Reformulate query
            query = self._reformulate_query(query)
            query.status = QueryStatus.COMPLETED
            logger.info("Query processing completed successfully")
            return query

        except Exception as e:
            query.status = QueryStatus.ERROR
            query.error_message = str(e)
            logger.error(f"Error processing query: {e}", exc_info=True)
            return query

    def continue_with_clarification(self, query_dict: dict, user_clarification: str) -> Query:
        """Continue processing a query with user's clarification (stateless mode).

        Args:
            query_dict: Dictionary representation of the Query object (context)
            user_clarification: Use's answer to the clarifying question

        Returns:
            Query object with full processing state and final output
        """
        # Reconstruct query object from dictionary
        query = Query(**query_dict)
        
        # Ensure it was waiting for clarification
        # Note: We rely on the caller to handle state management, but good to check
        # if query.status != QueryStatus.AWAITING_CLARIFICATION:
        #     logger.warning(f"Resuming query from status {query.status}, expected AWAITING_CLARIFICATION")

        query.user_clarification = user_clarification
        query.status = QueryStatus.CLARIFICATION_RECEIVED
        
        logger.info(f"Resuming query: {query.original_query[:50]}... with clarification")

        try:
            # Reformulate query
            query = self._reformulate_query(query)
            query.status = QueryStatus.COMPLETED
            logger.info("Query processing completed successfully")
            return query

        except Exception as e:
            query.status = QueryStatus.ERROR
            query.error_message = str(e)
            logger.error(f"Error resuming query: {e}", exc_info=True)
            return query


    def _detect_binary_ambiguity(self, query: Query) -> Query:
        """Detect if query is ambiguous using binary classification with the small model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with is_ambiguous field set
        """
        query.status = QueryStatus.CHECKING_AMBIGUITY
        logger.info("Step 1: Binary ambiguity detection")

        messages = BinaryDetectionPrompt.create_messages(query.original_query)
        response = self.small_model.detect_binary_ambiguity(
            messages,
            response_format=BinaryDetectionPrompt.get_response_schema(),
        )

        data = BinaryDetectionPrompt.parse_response(response)
        query.is_ambiguous = data["is_ambiguous"]

        logger.info(f"Binary ambiguity detection result: {query.is_ambiguous}")

        return query

    def _generate_clarifying_question(self, query: Query) -> Query:
        """Generate a clarifying question with embedded classification using the large model.

        This method now both identifies ambiguity types and generates the clarifying question
        in a single call to the large model.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with ambiguity_types, reasoning, and clarifying_question
        """
        logger.info(
            f"Step 2: Generating clarifying question with embedded classification (strategy: {self.clarification_strategy})"
        )

        messages = self.clarification_prompt_class.create_messages(query.original_query)
        response = self.large_model.generate_clarification(
            messages,
            response_format=self.clarification_prompt_class.get_response_schema(),
        )

        data = self.clarification_prompt_class.parse_response(response)

        # Populate ambiguity types from the generation response
        query.ambiguity_types = data.get("ambiguity_types", [])
        query.ambiguity_reasoning = data.get("reasoning", "")
        query.clarifying_question = data["clarifying_question"]
        
        types_str = ", ".join(query.ambiguity_types)
        logger.info(f"Identified ambiguity types: {types_str}")
        logger.info(f"Generated question: {query.clarifying_question}")

        return query



    def _reformulate_query(self, query: Query) -> Query:
        """Reformulate the query based on the user's clarification.

        Args:
            query: Query object to process

        Returns:
            Updated Query object with reformulated query
        """
        query.status = QueryStatus.REFORMULATING
        logger.info("Step 4: Reformulating query")

        ambiguity_types_strs = query.ambiguity_types
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
