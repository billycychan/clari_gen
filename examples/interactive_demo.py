#!/usr/bin/env python3
"""
Interactive demo showcasing the full clarification flow.
"""

from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.clients import SmallModelClient, LargeModelClient


def run_demo_with_clarification():
    """Demonstrate the full pipeline with a predefined clarification."""

    print("=" * 70)
    print("Full Clarification Flow Demo")
    print("=" * 70)
    print()

    # Initialize pipeline
    print("📊 Initializing pipeline...\n")
    small_client = SmallModelClient()
    large_client = LargeModelClient()

    pipeline = AmbiguityPipeline(
        small_model_client=small_client,
        large_model_client=large_client,
    )

    # Test connections
    results = pipeline.test_connections()
    if not all(results.values()):
        print("❌ Error: Model servers not available")
        return

    print("✓ Connected to models\n")

    # Example query that will need clarification
    query = "How many goals did Argentina score in the World Cup?"

    print("=" * 70)
    print("Scenario: Ambiguous temporal reference")
    print("=" * 70)
    print(f"\nOriginal Query: {query}\n")

    # Simulated user responses
    clarifications = {
        "attempt_1": "Some tournament",  # Invalid - too vague
        "attempt_2": "The 2022 FIFA World Cup in Qatar",  # Valid
    }

    attempt = 0

    def mock_clarification(question):
        nonlocal attempt
        attempt += 1

        print(f"\n📝 Clarifying Question (Attempt {attempt}):")
        print(f"   {question}\n")

        if attempt == 1:
            response = clarifications["attempt_1"]
            print(f"🤖 Simulated Response: {response}")
            print("   (This is intentionally vague to trigger validation failure)\n")
        else:
            response = clarifications["attempt_2"]
            print(f"🤖 Simulated Response: {response}")
            print("   (This provides specific information)\n")

        return response

    # Process the query
    print("🔄 Processing query through pipeline...\n")
    result = pipeline.process_query(query, clarification_callback=mock_clarification)

    # Show results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"✓ Original Query: {result.original_query}")
    types_str = ", ".join([t.value for t in result.ambiguity_types])
    print(f"✓ Ambiguity Type(s): {types_str}")
    print(f"✓ Reasoning: {result.ambiguity_reasoning}")
    print(f"✓ Total Clarification Attempts: {attempt}")
    print()

    if result.status.value == "COMPLETED":
        print(f"🎯 Reformulated Query:")
        print(f"   {result.get_final_output()}")
        print()
        print("✓ Query successfully clarified and reformulated!")
    else:
        print(f"❌ Status: {result.status.value}")
        if result.error_message:
            print(f"   Error: {result.error_message}")

    print("=" * 70)


def main():
    """Main entry point."""
    try:
        run_demo_with_clarification()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the vLLM servers are running:")
        print("  cd server && ./serve_models.sh")


if __name__ == "__main__":
    main()
