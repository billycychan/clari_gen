"""
Example: Using guided_json for Structured Output Generation

This example demonstrates how to use the `guided_json` parameter with vLLM
to generate structured JSON outputs that conform to a Pydantic schema.

The guided_json feature ensures that the model's output strictly follows
the defined JSON schema, making it ideal for applications that require
reliable, parseable structured data.

Requirements:
    - vLLM server running with a compatible model
    - OpenAI Python client library
    - Pydantic for schema definition

Usage:
    python examples/guided_json_example.py
"""

from pydantic import BaseModel, Field
from enum import Enum
from openai import OpenAI
from typing import Optional


# ============================================================================
# Example 1: Car Description (from your snippet)
# ============================================================================

class CarType(str, Enum):
    """Enumeration of car types."""
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    """Schema for car description."""
    brand: str
    model: str
    car_type: CarType


def example_car_description():
    """Generate a structured car description using guided_json."""
    # Initialize the OpenAI client pointing to your vLLM server
    client = OpenAI(
        base_url="http://localhost:8368/v1",  # Llama 3.1 8B server
        api_key="token-abc123"
    )
    
    # Get the JSON schema from the Pydantic model
    json_schema = CarDescription.model_json_schema()
    
    print("=" * 80)
    print("Example 1: Car Description")
    print("=" * 80)
    print(f"Schema: {json_schema}\n")
    
    # Make the completion request with guided_json
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    
    result = completion.choices[0].message.content
    print(f"Generated JSON:\n{result}\n")
    
    # Parse and validate the result
    car = CarDescription.model_validate_json(result)
    print(f"Validated Car Object:")
    print(f"  Brand: {car.brand}")
    print(f"  Model: {car.model}")
    print(f"  Type: {car.car_type}")
    print()


# ============================================================================
# Example 2: Ambiguity Classification (relevant to your project)
# ============================================================================

class AmbiguityType(str, Enum):
    """Types of query ambiguity."""
    clear = "clear"
    ambiguous = "ambiguous"


class AmbiguityClassification(BaseModel):
    """Schema for ambiguity classification with reasoning."""
    query: str = Field(description="The original query being classified")
    classification: AmbiguityType = Field(description="Whether the query is clear or ambiguous")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the classification")


def example_ambiguity_classification():
    """Classify query ambiguity using guided_json."""
    client = OpenAI(
        base_url="http://localhost:8368/v1",
        api_key="token-abc123"
    )
    
    json_schema = AmbiguityClassification.model_json_schema()
    
    print("=" * 80)
    print("Example 2: Ambiguity Classification")
    print("=" * 80)
    print(f"Schema: {json_schema}\n")
    
    test_query = "What is the capital?"
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at identifying ambiguous queries. Classify whether queries are clear or ambiguous."
            },
            {
                "role": "user",
                "content": f"Classify this query for ambiguity: '{test_query}'"
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    
    result = completion.choices[0].message.content
    print(f"Generated JSON:\n{result}\n")
    
    # Parse and validate
    classification = AmbiguityClassification.model_validate_json(result)
    print(f"Validated Classification:")
    print(f"  Query: {classification.query}")
    print(f"  Classification: {classification.classification}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print(f"  Reasoning: {classification.reasoning}")
    print()


# ============================================================================
# Example 3: Complex Nested Structure
# ============================================================================

class Address(BaseModel):
    """Nested address schema."""
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None


class Person(BaseModel):
    """Complex schema with nested objects and optional fields."""
    name: str
    age: int = Field(ge=0, le=150)
    email: str
    address: Address
    occupation: Optional[str] = None


def example_nested_structure():
    """Generate complex nested JSON structures."""
    client = OpenAI(
        base_url="http://localhost:8368/v1",
        api_key="token-abc123"
    )
    
    json_schema = Person.model_json_schema()
    
    print("=" * 80)
    print("Example 3: Complex Nested Structure")
    print("=" * 80)
    print(f"Schema: {json_schema}\n")
    
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON for a fictional software engineer living in San Francisco"
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    
    result = completion.choices[0].message.content
    print(f"Generated JSON:\n{result}\n")
    
    # Parse and validate
    person = Person.model_validate_json(result)
    print(f"Validated Person Object:")
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Email: {person.email}")
    print(f"  Address: {person.address.street}, {person.address.city}, {person.address.country}")
    print(f"  Occupation: {person.occupation}")
    print()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("GUIDED JSON EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        # Run Example 1: Car Description
        example_car_description()
        
        # Run Example 2: Ambiguity Classification
        example_ambiguity_classification()
        
        # Run Example 3: Complex Nested Structure
        example_nested_structure()
        
        print("=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your vLLM server is running and the model name is correct.")
        print("Update the base_url and model name in the script if needed.")


if __name__ == "__main__":
    main()
