#!/usr/bin/env python3
"""Check what models are actually available on the vLLM servers."""

import requests

servers = [
    ("http://localhost:8368/v1", "Llama-3.1-8B"),
    ("http://localhost:8369/v1", "Llama-3.3-70B"),
]

headers = {"Authorization": "Bearer token-abc123"}

for base_url, name in servers:
    print(f"\n{'='*70}")
    print(f"Checking {name} at {base_url}")
    print(f"{'='*70}")

    try:
        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        data = response.json()

        print(f"Available models:")
        for model in data.get("data", []):
            print(f"  - {model.get('id')}")

    except Exception as e:
        print(f"Error: {e}")
