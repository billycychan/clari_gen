#!/usr/bin/env python3
"""
Simple interactive demo of the ambiguity detection system.
"""

from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.clients import SmallModelClient, LargeModelClient


def main():
    """Run a simple demonstration with predefined queries."""

    print("=" * 70)
    print("Ambiguity Detection System - Simple Demo")
    print("=" * 70)
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    small_client = SmallModelClient()
    large_client = LargeModelClient()

    pipeline = AmbiguityPipeline(
        small_model_client=small_client,
        large_model_client=large_client,
    )

    # Test connections
    print("Testing model server connections...")
    results = pipeline.test_connections()
    if not all(results.values()):
        print("\n❌ Error: Could not connect to model servers.")
        print("Please ensure the vLLM servers are running:")
        print("  cd server && ./serve_models.sh")
        return

    print("✓ Connected to both models\n")

    # Demo queries
    demo_queries = [
        ("What is the capital of France?", "Clear query"),
        ("Tell me about the source of Nile.", "Lexical ambiguity"),
        ("When did he land on the moon?", "Semantic ambiguity - missing context"),
    ]

    for i, (query, description) in enumerate(demo_queries, 1):
        print("=" * 70)
        print(f"Demo {i}/3: {description}")
        print("=" * 70)
        print(f"\nQuery: {query}\n")

        result = pipeline.process_query(query)

        if result.is_ambiguous:
            print(f"✓ Detected as AMBIGUOUS")
            types_str = ", ".join([t.value for t in result.ambiguity_types])
            print(f"✓ Type(s): {types_str}")
            print(f"✓ Reasoning: {result.ambiguity_reasoning}")
            print(f"\n📝 Clarifying Question:")
            print(f"   {result.clarifying_question}")
            print("\n(In interactive mode, you would answer this question)")
        else:
            print(f"✓ Detected as NOT AMBIGUOUS")
            print(f"✓ Final output: {result.get_final_output()}")

        print()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("To run the full interactive system:")
    print("  python -m clari_gen.main")
    print()
    print("To process a single query:")
    print('  python -m clari_gen.main --query "Your query here"')
    print("=" * 70)


if __name__ == "__main__":
    main()
