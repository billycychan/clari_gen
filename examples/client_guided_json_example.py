"""
Example: Using Client Classes with guided_json for Structured Output

This example demonstrates how to use the client classes (SmallModelClient and 
LargeModelClient) with vLLM's guided_json parameter to generate structured JSON 
outputs that conform to Pydantic schemas.

The clients use vLLM's native guided_json for efficient, schema-compliant generation.

Usage:
    python examples/client_guided_json_example.py
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List

# Import the client classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clari_gen.clients.small_model_client import SmallModelClient
from clari_gen.clients.large_model_client import LargeModelClient


# ============================================================================
# Define Pydantic Schemas
# ============================================================================

class AmbiguityType(str, Enum):
    """Types of query ambiguity."""
    NONE = "NONE"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    INCOMPLETENESS = "INCOMPLETENESS"
    MULTIPLE_INTERPRETATIONS = "MULTIPLE_INTERPRETATIONS"


class AmbiguityClassification(BaseModel):
    """Schema for ambiguity classification with reasoning."""
    query: str = Field(description="The original query being classified")
    ambiguity_types: List[AmbiguityType] = Field(
        description="List of detected ambiguity types (or [NONE] if clear)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(description="Brief explanation for the classification")


class ClarifyingQuestion(BaseModel):
    """Schema for clarifying question generation."""
    original_query: str = Field(description="The original ambiguous query")
    clarifying_question: str = Field(
        description="A natural question to resolve the ambiguity"
    )
    purpose: str = Field(
        description="What aspect of the query this question clarifies"
    )


class ValidationResult(BaseModel):
    """Schema for validation of user clarification."""
    is_valid: bool = Field(
        description="Whether the clarification successfully resolves the ambiguity"
    )
    explanation: str = Field(
        description="Explanation of why the clarification is valid or invalid"
    )
    remaining_ambiguities: List[str] = Field(
        default=[],
        description="Any remaining ambiguities after the clarification"
    )


# ============================================================================
# Example 1: Small Model - Ambiguity Classification with guided_json
# ============================================================================

def example_small_model_classification():
    """Use SmallModelClient to classify query ambiguity with guided_json."""
    print("=" * 80)
    print("Example 1: Small Model (8B) - Ambiguity Classification")
    print("=" * 80)
    
    # Initialize the small model client
    client = SmallModelClient(base_url="http://localhost:8368/v1")
    
    # Test query
    test_query = "What is the capital?"
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert at identifying ambiguous queries. "
                      "Classify whether queries are clear or contain ambiguities."
        },
        {
            "role": "user",
            "content": f"Classify this query for ambiguity: '{test_query}'"
        }
    ]
    
    print(f"\nQuery: {test_query}")
    print(f"Schema: {AmbiguityClassification.__name__}\n")
    
    # Method 1: Using classify_ambiguity() with response_format
    print("Using classify_ambiguity() with guided_json:")
    response_json = client.classify_ambiguity(
        messages=messages,
        response_format=AmbiguityClassification
    )
    print(f"Response JSON:\n{response_json}\n")
    
    # Parse the response
    classification = AmbiguityClassification.model_validate_json(response_json)
    print(f"Parsed Classification:")
    print(f"  Query: {classification.query}")
    print(f"  Ambiguity Types: {[t.value for t in classification.ambiguity_types]}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print(f"  Reasoning: {classification.reasoning}")
    print()
    
    # Method 2: Using generate_structured() for automatic parsing
    print("Using generate_structured() for automatic parsing:")
    classification = client.generate_structured(
        messages=messages,
        response_format=AmbiguityClassification
    )
    print(f"Parsed Classification:")
    print(f"  Query: {classification.query}")
    print(f"  Ambiguity Types: {[t.value for t in classification.ambiguity_types]}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print(f"  Reasoning: {classification.reasoning}")
    print()


# ============================================================================
# Example 2: Large Model - Clarifying Question Generation
# ============================================================================

def example_large_model_clarification():
    """Use LargeModelClient to generate clarifying questions with guided_json."""
    print("=" * 80)
    print("Example 2: Large Model (70B) - Clarifying Question Generation")
    print("=" * 80)
    
    # Initialize the large model client
    client = LargeModelClient(base_url="http://localhost:8369/v1")
    
    # Test query
    test_query = "What is the capital?"
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert at generating clarifying questions for ambiguous queries."
        },
        {
            "role": "user",
            "content": f"Generate a clarifying question for this ambiguous query: '{test_query}'"
        }
    ]
    
    print(f"\nQuery: {test_query}")
    print(f"Schema: {ClarifyingQuestion.__name__}\n")
    
    # Generate clarifying question with guided_json
    print("Using generate_clarifying_question() with guided_json:")
    response_json = client.generate_clarifying_question(
        messages=messages,
        response_format=ClarifyingQuestion
    )
    print(f"Response JSON:\n{response_json}\n")
    
    # Parse the response
    clarification = ClarifyingQuestion.model_validate_json(response_json)
    print(f"Parsed Clarifying Question:")
    print(f"  Original Query: {clarification.original_query}")
    print(f"  Clarifying Question: {clarification.clarifying_question}")
    print(f"  Purpose: {clarification.purpose}")
    print()


# ============================================================================
# Example 3: Large Model - Validation with guided_json
# ============================================================================

def example_large_model_validation():
    """Use LargeModelClient to validate clarifications with guided_json."""
    print("=" * 80)
    print("Example 3: Large Model (70B) - Clarification Validation")
    print("=" * 80)
    
    # Initialize the large model client
    client = LargeModelClient(base_url="http://localhost:8369/v1")
    
    # Test data
    original_query = "What is the capital?"
    user_clarification = "I mean the capital of France."
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert at validating whether user clarifications "
                      "successfully resolve query ambiguities."
        },
        {
            "role": "user",
            "content": f"Original query: '{original_query}'\n"
                      f"User clarification: '{user_clarification}'\n"
                      f"Does this clarification resolve the ambiguity?"
        }
    ]
    
    print(f"\nOriginal Query: {original_query}")
    print(f"User Clarification: {user_clarification}")
    print(f"Schema: {ValidationResult.__name__}\n")
    
    # Validate with guided_json
    print("Using validate_clarification() with guided_json:")
    response_json = client.validate_clarification(
        messages=messages,
        response_format=ValidationResult
    )
    print(f"Response JSON:\n{response_json}\n")
    
    # Parse the response
    validation = ValidationResult.model_validate_json(response_json)
    print(f"Parsed Validation:")
    print(f"  Is Valid: {validation.is_valid}")
    print(f"  Explanation: {validation.explanation}")
    print(f"  Remaining Ambiguities: {validation.remaining_ambiguities}")
    print()


# ============================================================================
# Example 4: Multiple Queries Batch Processing
# ============================================================================

def example_batch_processing():
    """Process multiple queries with guided_json."""
    print("=" * 80)
    print("Example 4: Batch Processing Multiple Queries")
    print("=" * 80)
    
    client = SmallModelClient(base_url="http://localhost:8368/v1")
    
    test_queries = [
        "What is the capital?",
        "Show me restaurants",
        "When was the Declaration of Independence signed?",
        "How do I get there?"
    ]
    
    print(f"\nProcessing {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        messages = [
            {
                "role": "system",
                "content": "Classify queries for ambiguity."
            },
            {
                "role": "user",
                "content": f"Classify: '{query}'"
            }
        ]
        
        classification = client.generate_structured(
            messages=messages,
            response_format=AmbiguityClassification
        )
        
        print(f"{i}. Query: {query}")
        print(f"   Types: {[t.value for t in classification.ambiguity_types]}")
        print(f"   Confidence: {classification.confidence:.2f}")
        print(f"   Reasoning: {classification.reasoning}")
        print()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CLIENT GUIDED JSON EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        # Run Example 1: Small Model Classification
        example_small_model_classification()
        
        # Run Example 2: Large Model Clarification
        example_large_model_clarification()
        
        # Run Example 3: Large Model Validation
        example_large_model_validation()
        
        # Run Example 4: Batch Processing
        example_batch_processing()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your vLLM servers are running:")
        print("  - Small model (8B): http://localhost:8368/v1")
        print("  - Large model (70B): http://localhost:8369/v1")


if __name__ == "__main__":
    main()
