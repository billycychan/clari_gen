from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pytest

from clari_gen.api.main import app
from clari_gen.models import Query, QueryStatus


# Mock the pipeline to avoid loading large models
@patch("clari_gen.api.main.pipeline")
def test_pipeline_flow(mock_pipeline):
    client = TestClient(app)

    # --- Test 1: Unambiguous Query ---
    # Setup mock return
    mock_q1 = Query(original_query="What is 2+2?")
    mock_q1.status = QueryStatus.NOT_AMBIGUOUS
    mock_q1.is_ambiguous = False

    # We need to mock process_query on the pipeline instance
    # Since 'pipeline' is global in main.py, we need to inspect how it's used.
    # The lifespan sets the global variable.
    # For testing, we can override the dependency or mock the global.

    # However, since we are patching the global 'pipeline' variable imported in the test,
    # we need to make sure the app uses the mocked one.
    # Actually, patching 'clari_gen.api.main.pipeline' should work if we construct the app after,
    # but the app is already constructed.
    # A better way is to override the dependency if we had one, but here we use a global.

    # For simplicity in this script, let's just assume the environment is mocked
    # But since I can't easily mock the global variable inside the running app without complex setup,
    # I will rely on the real pipeline being initialized or mocked at a system level.

    # BETTER APPROACH: Create a test that doesn't rely on the global variable being set by lifespan
    # OR trigger the lifespan.
    pass


# Let's write a simpler test script that doesn't use pytest but runs a quick check if possible.
# Or better, since I can't easily mock the models without loading them (which is slow/impossible without GPU),
# I will create a dummy pipeline class and swap it.

import clari_gen.api.main as main_module


class DummyPipeline:
    def process_query(self, text, clarification_callback=None):
        if "bank" in text:
            q = Query(original_query=text)
            q.status = QueryStatus.AWAITING_CLARIFICATION
            q.is_ambiguous = True
            q.clarifying_question = "Do you mean river bank or money bank?"
            q.ambiguity_types = ["polysemy"]
            return q
        else:
            q = Query(original_query=text)
            q.status = QueryStatus.COMPLETED
            q.is_ambiguous = False
            return q

    def continue_with_clarification(self, query_dict, user_clarification):
        q = Query(**query_dict)
        q.user_clarification = user_clarification
        q.reformulated_query = f"{q.original_query} ({user_clarification})"
        q.status = QueryStatus.AWAITING_CONFIRMATION
        return q

    def confirm_reformulation(self, query_dict, confirmation, alternative_query=None):
        q = Query(**query_dict)
        if confirmation.lower() in ["yes", "y"]:
            q.confirmed_query = q.reformulated_query
        elif alternative_query:
            q.confirmed_query = alternative_query
        else:
            q.confirmed_query = q.reformulated_query
        q.status = QueryStatus.COMPLETED
        return q


def run_test():
    # Swap pipeline
    main_module.pipeline = DummyPipeline()
    client = TestClient(app)

    print("--- Test 1: Unambiguous Query 'Hello world' ---")
    response = client.post("/v1/query", json={"text": "Hello world"})
    print(response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    print("\n--- Test 2: Ambiguous Query 'Where is the bank?' ---")
    response = client.post("/v1/query", json={"text": "Where is the bank?"})
    r_json = response.json()
    print(r_json)
    assert response.status_code == 200
    assert r_json["status"] == "clarification_needed"
    assert "context" in r_json

    print("\n--- Test 3: Clarification Flow ---")
    context = r_json["context"]
    c_response = client.post(
        "/v1/clarify", json={"answer": "Money bank", "context": context}
    )
    c_json = c_response.json()
    print(c_json)
    assert c_response.status_code == 200
    assert c_json["status"] == "confirmation_needed"
    assert "Money bank" in c_json["reformulated_query"]

    print("\n--- Test 4: Confirmation Flow (Yes) ---")
    context = c_json["context"]
    confirm_response = client.post(
        "/v1/confirm", json={"confirmation": "yes", "context": context}
    )
    confirm_json = confirm_response.json()
    print(confirm_json)
    assert confirm_response.status_code == 200
    assert confirm_json["status"] == "completed"
    assert "Money bank" in confirm_json["confirmed_query"]

    print("\n--- Test 5: Confirmation Flow (No with Alternative) ---")
    # Reset to clarification stage
    response = client.post("/v1/query", json={"text": "Where is the bank?"})
    context = response.json()["context"]
    c_response = client.post(
        "/v1/clarify", json={"answer": "River bank", "context": context}
    )
    context = c_response.json()["context"]

    alt_response = client.post(
        "/v1/confirm",
        json={
            "confirmation": "no",
            "alternative_query": "Where is the nearest river bank for fishing?",
            "context": context,
        },
    )
    alt_json = alt_response.json()
    print(alt_json)
    assert alt_response.status_code == 200
    assert alt_json["status"] == "completed"
    assert alt_json["confirmed_query"] == "Where is the nearest river bank for fishing?"

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_test()
