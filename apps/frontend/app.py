import streamlit as st
import requests
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8370/v1")

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QUERY_FILE = os.path.join(PROJECT_ROOT, "evaluation", "results", "real-queries.tsv")
QUERY_FILE = os.getenv("QUERY_FILE", DEFAULT_QUERY_FILE)

st.set_page_config(page_title="Clarification Module Demo", layout="wide")

st.title("Clarification Module Demo")
st.markdown("Test the ambiguity detection and clarification system.")

# --- Sidebar: Query Selection ---
st.sidebar.header("Data Source")


@st.cache_data
def load_queries():
    try:
        # Assuming TSV has no header and first column is ID, second is query
        df = pd.read_csv(QUERY_FILE, sep="\t", header=None, names=["id", "query"])
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading queries: {e}")
        return pd.DataFrame()


df_queries = load_queries()

if not df_queries.empty:
    selected_query_index = st.sidebar.selectbox(
        "Select an example query:",
        options=df_queries.index,
        format_func=lambda x: f"{df_queries.iloc[x]['id']}: {df_queries.iloc[x]['query'][:60]}...",
        index=0,
    )
    default_query_text = df_queries.iloc[selected_query_index]["query"]
else:
    default_query_text = ""

# --- Session State ---
if "current_context" not in st.session_state:
    st.session_state.current_context = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "api_calls" not in st.session_state:
    st.session_state.api_calls = []

# --- Main Interface ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Query Input")

    # Input area - updates when selection changes if not modified by user
    # We use a session state key for the text area to allow programmatic updates
    if "input_text" not in st.session_state:
        st.session_state.input_text = default_query_text

    # Update input text if selection changed
    if st.sidebar.button("Load Selected Query"):
        st.session_state.input_text = default_query_text
        st.session_state.current_context = None
        st.session_state.chat_history = []
        st.session_state.api_calls = []

    query_text = st.text_area("Enter your query:", key="input_text", height=150)

    if st.button("Analyze Query", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                req_body = {"text": query_text}
                response = requests.post(f"{API_URL}/query", json=req_body)

                # Capture API details
                try:
                    resp_body = response.json()
                except:
                    resp_body = response.text

                st.session_state.api_calls.append(
                    {
                        "title": "Analyze Analysis (Query)",
                        "request": {
                            "url": f"{API_URL}/query",
                            "method": "POST",
                            "headers": dict(response.request.headers),
                            "body": req_body,
                        },
                        "response": {
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "body": resp_body,
                        },
                    }
                )

                response.raise_for_status()
                data = resp_body

                # Reset history for new query
                st.session_state.chat_history = [
                    {"role": "user", "content": query_text}
                ]
                # If we are starting a completely new analysis, we might want to keep the API calls relevant to this analysis
                # But since the button "Analyze Query" effectively restarts the flow, maybe we should clear previous API calls?
                # The user might want to see history of previous attempts, but "Load Selected Query" clears everything.
                # Let's clear it here too to keep it clean for the new "Analyze" action which implies a fresh start for the current text.
                # Actually, capturing the flow "Analyze -> Clarify" is the goal. If I hit Analyze again, it's a new flow.
                # So we should probably keep the list, but maybe the user wants to see the *sequence*.
                # Let's clear api_calls if we are starting a fresh analysis to avoid confusion.
                st.session_state.api_calls = [
                    st.session_state.api_calls[-1]
                ]  # Keep only the one we just added

                if data["status"] == "clarification_needed":
                    st.session_state.current_context = data["context"]
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"**Ambiguity Detected:** {', '.join(data['context'].get('ambiguity_types', []))}\n\n**Question:** {data['clarifying_question']}",
                        }
                    )
                elif data["status"] == "completed":
                    st.session_state.current_context = None
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"**Query is Clear.**\n\n**Result:** {data['reformulated_query']}",
                        }
                    )

            except Exception as e:
                st.error(f"Error calling API: {e}")

    # Clarification/Confirmation Input (Only enabled if context exists)
    if st.session_state.current_context:
        context_status = st.session_state.current_context.get("status", "")

        if context_status == "AWAITING_CLARIFICATION":
            st.info("The system needs clarification.")
            clarification_answer = st.text_input("Your Answer:")

            if st.button("Submit Clarification"):
                if not clarification_answer:
                    st.warning("Please enter an answer.")
                else:
                    with st.spinner("Reformulating..."):
                        try:
                            # Append user answer to history locally so it shows up immediately
                            st.session_state.chat_history.append(
                                {"role": "user", "content": clarification_answer}
                            )

                            payload = {
                                "answer": clarification_answer,
                                "context": st.session_state.current_context,
                            }
                            response = requests.post(f"{API_URL}/clarify", json=payload)

                            # Capture API details
                            try:
                                resp_body = response.json()
                            except:
                                resp_body = response.text

                            st.session_state.api_calls.append(
                                {
                                    "title": "Submit Clarification",
                                    "request": {
                                        "url": f"{API_URL}/clarify",
                                        "method": "POST",
                                        "headers": dict(response.request.headers),
                                        "body": payload,
                                    },
                                    "response": {
                                        "status_code": response.status_code,
                                        "headers": dict(response.headers),
                                        "body": resp_body,
                                    },
                                }
                            )

                            response.raise_for_status()
                            data = resp_body

                            if data["status"] == "confirmation_needed":
                                # Update context for confirmation step
                                st.session_state.current_context = data["context"]
                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": f"**Reformulated Query:** {data['reformulated_query']}",
                                    }
                                )
                                st.rerun()
                            elif data["status"] == "completed":
                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": f"**Reformulated Query:** {data['reformulated_query']}",
                                    }
                                )
                                # Clear context to disable input
                                st.session_state.current_context = None

                        except Exception as e:
                            st.error(f"Error submitting clarification: {e}")

        elif context_status == "AWAITING_CONFIRMATION":
            st.info("Please confirm the reformulated query.")
            reformulated = st.session_state.current_context.get(
                "reformulated_query", ""
            )

            st.markdown(f"**Reformulated Query:** {reformulated}")
            st.markdown("===")
            st.markdown(
                "Is this the query you want? Please answer yes or no. If no, provide the alternative query."
            )

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("✓ Yes", use_container_width=True):
                    with st.spinner("Confirming..."):
                        try:
                            payload = {
                                "confirmation": "yes",
                                "context": st.session_state.current_context,
                            }
                            response = requests.post(f"{API_URL}/confirm", json=payload)

                            # Capture API details
                            try:
                                resp_body = response.json()
                            except:
                                resp_body = response.text

                            st.session_state.api_calls.append(
                                {
                                    "title": "Confirm Query (Yes)",
                                    "request": {
                                        "url": f"{API_URL}/confirm",
                                        "method": "POST",
                                        "headers": dict(response.request.headers),
                                        "body": payload,
                                    },
                                    "response": {
                                        "status_code": response.status_code,
                                        "headers": dict(response.headers),
                                        "body": resp_body,
                                    },
                                }
                            )

                            response.raise_for_status()
                            data = resp_body

                            if data["status"] == "completed":
                                st.session_state.chat_history.append(
                                    {"role": "user", "content": "Yes"}
                                )
                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": f"**Final Query:** {data['confirmed_query']}",
                                    }
                                )
                                st.session_state.current_context = None
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error confirming query: {e}")

            with col_b:
                if st.button("✗ No", use_container_width=True):
                    st.session_state["show_alternative_input"] = True

            if st.session_state.get("show_alternative_input", False):
                alternative_query = st.text_area(
                    "Enter your alternative query:", key="alt_query_input"
                )
                if st.button("Submit Alternative"):
                    if not alternative_query:
                        st.warning("Please enter an alternative query.")
                    else:
                        with st.spinner("Processing..."):
                            try:
                                payload = {
                                    "confirmation": "no",
                                    "alternative_query": alternative_query,
                                    "context": st.session_state.current_context,
                                }
                                response = requests.post(
                                    f"{API_URL}/confirm", json=payload
                                )

                                # Capture API details
                                try:
                                    resp_body = response.json()
                                except:
                                    resp_body = response.text

                                st.session_state.api_calls.append(
                                    {
                                        "title": "Confirm Query (No - Alternative)",
                                        "request": {
                                            "url": f"{API_URL}/confirm",
                                            "method": "POST",
                                            "headers": dict(response.request.headers),
                                            "body": payload,
                                        },
                                        "response": {
                                            "status_code": response.status_code,
                                            "headers": dict(response.headers),
                                            "body": resp_body,
                                        },
                                    }
                                )

                                response.raise_for_status()
                                data = resp_body

                                if data["status"] == "completed":
                                    st.session_state.chat_history.append(
                                        {
                                            "role": "user",
                                            "content": f"No, use this instead: {alternative_query}",
                                        }
                                    )
                                    st.session_state.chat_history.append(
                                        {
                                            "role": "assistant",
                                            "content": f"**Final Query:** {data['confirmed_query']}",
                                        }
                                    )
                                    st.session_state.current_context = None
                                    st.session_state["show_alternative_input"] = False
                                    st.rerun()

                            except Exception as e:
                                st.error(f"Error submitting alternative: {e}")

with col2:
    st.subheader("Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Debug info
    if st.session_state.current_context:
        with st.expander("Debug: Context Blob"):
            st.json(st.session_state.current_context)

    # API Inspector
    # API Inspector
    if "api_calls" in st.session_state and st.session_state.api_calls:
        st.divider()
        st.subheader("API Inspector")

        # Reverse order to show newest on top usually, or keep chronological?
        # Chronological (iteration) makes more sense for a "flow".
        for i, call in enumerate(st.session_state.api_calls):
            with st.expander(f"{i+1}. {call.get('title', 'API Call')}", expanded=False):
                tab1, tab2 = st.tabs(["Request", "Response"])

                with tab1:
                    st.markdown(f"**URL:** `{call['request']['url']}`")
                    st.markdown(f"**Method:** `{call['request']['method']}`")
                    st.subheader("Headers")
                    st.json(call["request"]["headers"])
                    st.subheader("Body")
                    st.json(call["request"]["body"])

                with tab2:
                    st.markdown(f"**Status Code:** `{call['response']['status_code']}`")
                    st.subheader("Headers")
                    st.json(call["response"]["headers"])
                    st.subheader("Body")
                    st.json(call["response"]["body"])
