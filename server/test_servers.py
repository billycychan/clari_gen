#!/usr/bin/env python3
"""
Example client to test the vLLM OpenAI-compatible servers.
"""

from openai import OpenAI

# Configure clients for each model
llama_3_1_client = OpenAI(
    api_key="token-abc123",
    base_url="http://localhost:8368/v1",
)

llama_3_3_client = OpenAI(
    api_key="token-abc123",
    base_url="http://localhost:8369/v1",
)


def test_model(client, model_name, port):
    """Test a model with a simple chat completion."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} on port {port}")
    print(f"{'='*70}")

    try:
        # Test chat completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print(f"\nResponse from {model_name}:")
        print(f"{response.choices[0].message.content}")
        print(f"\nTokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error testing {model_name}: {e}")


def main():
    print("\nvLLM Multi-Model Server Test")
    print("=" * 70)

    # Test Llama 3.1 8B Instruct
    test_model(llama_3_1_client, "meta-llama/Llama-3.1-8B-Instruct", 8368)

    # Test Llama 3.3 70B FP8
    test_model(llama_3_3_client, "nvidia/Llama-3.3-70B-Instruct-FP8", 8369)

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
