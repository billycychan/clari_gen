"""
Main CLI interface for the ambiguity detection and clarification system.
"""

import sys
import argparse
import json
from typing import Optional

from clari_gen.config import Config
from clari_gen.clients import SmallModelClient, LargeModelClient
from clari_gen.orchestrator import AmbiguityPipeline
from clari_gen.utils import setup_logger


def interactive_mode(pipeline: AmbiguityPipeline):
    """Run the pipeline in interactive mode with user input.

    Args:
        pipeline: Configured AmbiguityPipeline instance
    """
    print("\n" + "=" * 70)
    print("Ambiguity Detection and Clarification System")
    print("=" * 70)
    print("\nThis system analyzes your queries for ambiguity and helps clarify them.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Get query from user
        print("-" * 70)
        query_text = input("\nEnter your query: ").strip()

        if query_text.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not query_text:
            print("Please enter a valid query.")
            continue

        # Define clarification callback for interactive input
        def get_clarification(clarifying_question: str) -> str:
            print("\n" + "=" * 70)
            print("CLARIFICATION NEEDED")
            print("=" * 70)
            print(f"\n{clarifying_question}\n")
            response = input("Your answer: ").strip()
            return response

        # Process the query
        print("\n📊 Processing query...")
        result = pipeline.process_query(
            query_text, clarification_callback=get_clarification
        )

        # Display results
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        if result.status.value == "ERROR":
            print(f"\n❌ Error: {result.error_message}")
        elif result.is_ambiguous:
            print(f"\n✓ Original query: {result.original_query}")
            types_str = ", ".join([t.value for t in result.ambiguity_types])
            print(f"✓ Ambiguity type(s): {types_str}")
            print(f"✓ Reasoning: {result.ambiguity_reasoning}")
            if result.user_clarification:
                print(f"✓ Your clarification: {result.user_clarification}")
            print(f"\n🎯 Reformulated query: {result.get_final_output()}")
        else:
            print(f"\n✓ Query is clear and unambiguous")
            print(f"\n🎯 Final query: {result.get_final_output()}")

        print()


def batch_mode(pipeline: AmbiguityPipeline, query_text: str, output_json: bool = False):
    """Process a single query in batch mode (non-interactive).

    Args:
        pipeline: Configured AmbiguityPipeline instance
        query_text: The query to process
        output_json: Whether to output JSON format
    """
    # Process without clarification callback - will stop at AWAITING_CLARIFICATION if needed
    result = pipeline.process_query(query_text, clarification_callback=None)

    if output_json:
        # Output JSON format
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Human-readable output
        if result.status.value == "ERROR":
            print(f"Error: {result.error_message}", file=sys.stderr)
            sys.exit(1)
        elif result.status.value == "AWAITING_CLARIFICATION":
            print("Query is ambiguous and requires clarification:", file=sys.stderr)
            types_str = ", ".join([t.value for t in result.ambiguity_types])
            print(f"Type(s): {types_str}", file=sys.stderr)
            print(f"Question: {result.clarifying_question}", file=sys.stderr)
            sys.exit(2)
        else:
            # Output the final query
            print(result.get_final_output())


def test_connections(pipeline: AmbiguityPipeline):
    """Test connections to model servers.

    Args:
        pipeline: Configured AmbiguityPipeline instance
    """
    print("\n🔍 Testing connections to model servers...\n")

    results = pipeline.test_connections()

    for model_name, status in results.items():
        status_icon = "✓" if status else "✗"
        status_text = "Connected" if status else "Failed"
        print(f"{status_icon} {model_name}: {status_text}")

    if all(results.values()):
        print("\n✓ All connections successful!")
        sys.exit(0)
    else:
        print("\n✗ Some connections failed. Please check your model servers.")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ambiguity Detection and Clarification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m clari_gen.main
  
  # Batch mode with single query
  python -m clari_gen.main --query "Tell me about the source of Nile."
  
  # Output as JSON
  python -m clari_gen.main --query "Who won the championship?" --json
  
  # Test server connections
  python -m clari_gen.main --test
  
  # Set log level
  python -m clari_gen.main --log-level DEBUG
        """,
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Query to process (batch mode). If omitted, runs in interactive mode.",
    )

    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON (batch mode only)",
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test connections to model servers and exit",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum clarification attempts (default: 3)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(name="clari_gen", level=args.log_level)
    logger = setup_logger(name=__name__, level=args.log_level)

    # Load configuration
    config = Config.default()
    config.pipeline.max_clarification_attempts = args.max_attempts

    # Initialize clients
    small_client = SmallModelClient(
        base_url=config.model.small_model_base_url,
        api_key=config.model.api_key,
    )

    large_client = LargeModelClient(
        base_url=config.model.large_model_base_url,
        api_key=config.model.api_key,
    )

    # Initialize pipeline
    pipeline = AmbiguityPipeline(
        small_model_client=small_client,
        large_model_client=large_client,
        max_clarification_attempts=config.pipeline.max_clarification_attempts,
    )

    # Route to appropriate mode
    if args.test:
        test_connections(pipeline)
    elif args.query:
        batch_mode(pipeline, args.query, output_json=args.json)
    else:
        try:
            interactive_mode(pipeline)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
