"""
Streamlit Chat Frontend for NL2SQL System.

A lightweight, production-ready chat interface that consumes the existing
NL2SQL backend API. Designed for clarity, speed, and extensibility.

Author: NL2SQL Team
Version: 2.0
"""

import streamlit as st
import requests
import json
import os
import pandas as pd
import random
from typing import Dict, Any, Optional, Generator, List
from dotenv import load_dotenv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# API Configuration - reads from environment, falls back to defaults
API_HOST = os.getenv("APP_HOST", "localhost")
API_PORT = os.getenv("APP_PORT", "8000")

# Translate 0.0.0.0 (server binding) to localhost (client connection)
if API_HOST == "0.0.0.0":
    API_HOST = "localhost"

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# History persistence (simple local JSON file)
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


def _new_chat(title: str = "New chat") -> Dict[str, Any]:
    chat_id = f"chat_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
    return {
        "id": chat_id,
        "title": title,
        "messages": [],
        "created_at": datetime.utcnow().isoformat()
    }


def load_history() -> Dict[str, Any]:
    """Load persisted chat history from disk."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Backward compatibility: old format was a list of messages
                if isinstance(data, list):
                    chat = _new_chat("Chat 1")
                    chat["messages"] = data
                    return {"chats": [chat]}
                if isinstance(data, dict) and "chats" in data:
                    return data
    except Exception:
        pass
    return {"chats": [_new_chat("Chat 1")]}


def save_history(chats: List[Dict[str, Any]]) -> None:
    """Persist chat history to disk (best-effort)."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"chats": chats}, f)
    except Exception:
        pass


def get_active_chat() -> Optional[Dict[str, Any]]:
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id:
        return None
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            return chat
    return None


def get_active_messages() -> List[Dict[str, Any]]:
    chat = get_active_chat()
    return chat["messages"] if chat else []


def get_jwt_token(role: str = "viewer") -> str:
    """
    Generate a JWT token for authentication.

    Args:
        role: User role ('admin' or 'viewer')

    Returns:
        Signed JWT token string
    """
    try:
        # Import from the backend auth module
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from app.auth import create_jwt_token
        return create_jwt_token(f"streamlit_user", role)
    except Exception as e:
        # If import fails, return None and handle gracefully
        print(f"Warning: Could not generate JWT token: {e}")
        return None


def generate_insights(schema: Dict[str, Any], role: str, last_question: Optional[str] = None) -> List[str]:
    """
    Generate contextual insight questions based on schema structure and user role.

    Args:
        schema: Database schema with tables and columns
        role: User role ('admin' or 'viewer')

    Returns:
        List of 3-5 high-confidence insight questions
    """
    if not schema or "tables" not in schema:
        return ["How many records are in the database?"]

    insights = []
    tables = [table["name"] for table in schema["tables"]]
    table_set = set(tables)

    # Base insights (safe for all users)
    if "orders" in table_set:
        insights.append("What is the total revenue this month?")
        insights.append("Show recent orders and their status")

    if "customers" in table_set:
        insights.append("How many customers are currently active?")

    if "products" in table_set:
        insights.append("Which products are selling the most?")

    # Admin-only insights (deeper, operational)
    if role == "admin":
        if "products" in table_set:
            products_table = next(
                (t for t in schema["tables"] if t["name"] == "products"), None)
            if products_table:
                product_cols = [c["name"] for c in products_table["columns"]]
                if "price" in product_cols and "cost" in product_cols:
                    insights.append("Which products have the lowest margin?")
                if "stock_quantity" in product_cols and "reorder_level" in product_cols:
                    insights.append(
                        "Which products are close to reorder level?")

        if "suppliers" in table_set and "orders" in table_set:
            insights.append("Are there suppliers with delayed lead times?")

    # Story Mode: Context-aware follow-ups
    if last_question:
        q_lower = last_question.lower()
        story_options = []

        if any(w in q_lower for w in ["revenue", "sales", "income", "profit"]):
            story_options.extend([
                "How does this compare to last month?",
                "Show the revenue trend over the last 6 months",
                "Break this down by product category"
            ])

        if any(w in q_lower for w in ["customer", "user", "client"]):
            story_options.extend([
                "Show the top 5 most active customers",
                "What is the average lifetime value?",
                "Show distribution of customers by region"
            ])

        if any(w in q_lower for w in ["product", "item", "stock", "inventory"]):
            story_options.extend([
                "Which products have high stock but low sales?",
                "Show sales performance for this category",
                "What is the return rate for these items?"
            ])

        if any(w in q_lower for w in ["order", "transaction"]):
            story_options.extend([
                "What is the average order processing time?",
                "Show orders with delayed status"
            ])

        insights.extend(story_options)

    # De-duplicate and Shuffle
    insights = list(set(insights))
    random.shuffle(insights)

    # Return limited set, prioritize story items if present (already shuffled though)
    return insights[:4] if insights else ["How many records are in the database?"]


def normalize_query_data(data: Any) -> Optional[List[Dict]]:
    """
    Normalize query result data to a list of dicts format.

    Backend may return:
    - {"columns": [...], "rows": [[...], ...]} format
    - List of dicts format

    Args:
        data: Raw data from backend

    Returns:
        List of dicts suitable for DataFrame, or None if invalid
    """
    if not data:
        return None

    # If already a list of dicts, return as-is
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            return data
        # If list of lists, it might be raw rows - can't process without columns
        return None

    # Handle {"columns": [...], "rows": [...]} format
    if isinstance(data, dict):
        columns = data.get("columns")
        rows = data.get("rows")
        if columns and rows and isinstance(columns, list) and isinstance(rows, list):
            # Convert to list of dicts
            return [dict(zip(columns, row)) for row in rows]

    return None


# Page Configuration
st.set_page_config(
    page_title="NL2SQL Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "chats" not in st.session_state:
    st.session_state.chats = load_history().get("chats", [])
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = st.session_state.chats[0]["id"] if st.session_state.chats else None

if "schema" not in st.session_state:
    st.session_state.schema = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "debug_events" not in st.session_state:
    st.session_state.debug_events = []

if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "insights" not in st.session_state:
    st.session_state.insights = None

# ============================================================================
# MINIMAL STYLING
# ============================================================================

st.markdown("""
<style>
    /* Clean header */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    /* Status badges */
    .status-ok {
        background: #ecfdf5;
        color: #047857;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-error {
        background: #fef2f2;
        color: #dc2626;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# API CLIENT FUNCTIONS
# ============================================================================


def check_api_health() -> Dict[str, Any]:
    """
    Check if the backend API is healthy.

    Returns:
        Dict with 'status', 'database_connected', 'agent_ready' keys
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "error": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_database_schema(token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch database schema from the API.

    Args:
        token: Optional JWT token for authentication

    Returns:
        Schema dict or None if fetch fails
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"{API_BASE_URL}/schema",
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def query_backend_stream(question: str, token: Optional[str] = None) -> Generator[Dict, None, None]:
    """
    Stream query results from the backend via SSE.

    Args:
        question: Natural language question
        token: Optional JWT token for authentication

    Yields:
        Event dicts with 'type' key indicating event type
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"question": question},
            headers=headers,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    event_data = line_str[6:]  # Remove 'data: ' prefix
                    try:
                        yield json.loads(event_data)
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield {"type": "error", "error": str(e)}


def query_backend_once(question: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Non-streaming query fallback to the backend /query endpoint.

    Returns:
        Dict with keys: answer, data, execution_time, reasoning_steps, error (optional)
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        headers=headers,
        timeout=300
    )
    if response.status_code != 200:
        return {"answer": "", "data": None, "execution_time": 0, "reasoning_steps": 0, "error": response.text}
    return response.json()


# ============================================================================
# DATA TABLE RENDERING FUNCTIONS
# ============================================================================


def render_data_table(df: pd.DataFrame) -> None:
    """Render only the raw data table (no chart)."""
    if df.empty:
        st.info("No data to display.")
        return

    try:
        with st.expander("📋 View Raw Data", expanded=True):
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} rows")

    except Exception as e:
        st.warning(f"Could not render table: {e}")
        st.dataframe(df, use_container_width=True)


# ============================================================================
# RESPONSE RENDERING
# ============================================================================


def render_response(answer: str, data: Any, metadata: Dict, pii: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the assistant's response with optional table/chart.

    Args:
        answer: Natural language answer text
        data: Query result data (may be columns/rows format or list of dicts)
        metadata: Execution metadata (time, steps)
    """
    # Display the answer
    st.markdown(answer)

    # Display detected PII summary (if available)
    if pii is not None:
        pii_count = pii.get("count", 0)
        st.markdown(f"**PII Detected:** {pii_count}")
        entities = pii.get("entities", [])
        if entities:
            st.dataframe(entities, use_container_width=True)
        else:
            st.caption("No PII entities detected in the question.")

    # Normalize and render table if present
    normalized_data = normalize_query_data(data)
    if normalized_data and len(normalized_data) > 0:
        try:
            df = pd.DataFrame(normalized_data)

            # Smart Type Coercion for Charting
            # Backend might return numbers as strings (e.g., "100", "₹1,000")
            for col in df.columns:
                try:
                    # Clean potential currency/formatting if string type
                    if df[col].dtype == 'object':
                        # Remove currency symbols (₹, $), commas, and whitespace
                        cleaned_col = df[col].astype(str).str.replace(
                            r'[₹$,\s]', '', regex=True)

                        # Attempt to convert to numeric with coercion (turning errors to NaN)
                        coerced = pd.to_numeric(cleaned_col, errors='coerce')

                        # LOGIC: If a significant portion (>40%) of the column converts to numbers,
                        # assume it's a numeric column and use the coerced values.
                        # This handles mixed columns (e.g. ["100", "N/A"]) while preserving
                        # text columns (e.g. ["Widget A"]) which would become all NaNs.
                        if coerced.notna().mean() > 0.4:
                            df[col] = coerced
                        # Else: Keep original object type
                    else:
                        # Already numeric or other type
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    # Keep as original if conversion fails
                    pass

            render_data_table(df)

        except Exception as e:
            st.warning(f"Could not process visualization: {e}")
            # Fallback to simple table
            st.dataframe(pd.DataFrame(normalized_data),
                         use_container_width=True)

    # Display metadata
    if metadata:
        col1, col2 = st.columns(2)
        with col1:
            exec_time = metadata.get('execution_time', 0)
            if isinstance(exec_time, (int, float)):
                st.caption(f"⏱️ {exec_time:.2f}s")
            else:
                st.caption(f"⏱️ {exec_time}s")
        with col2:
            st.caption(f"🔄 {metadata.get('reasoning_steps', 0)} steps")


# ============================================================================
# SIDEBAR
# ============================================================================


def render_sidebar() -> None:
    """Render the sidebar with auth, status, schema, and controls."""

    # Custom CSS to reduce sidebar top padding and style status
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] div.block-container {
            padding-top: 3rem;
        }
        .main-header {
            margin-top: -1rem; /* Pull header up */
        }
        .status-item {
            display: flex;
            align_items: center;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # App Header in Sidebar
        st.markdown('<div class="main-header">💬 NL2SQL Chat</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Ask questions about your database in natural language</div>',
            unsafe_allow_html=True
        )

        # Auto-login as admin (silent)
        if not st.session_state.jwt_token:
            token = get_jwt_token("admin")
            if token:
                st.session_state.jwt_token = token
                st.session_state.user_role = "admin"
                with st.spinner("Loading schema..."):
                    schema = get_database_schema(token)
                    if schema:
                        st.session_state.schema = schema
                        st.session_state.insights = generate_insights(schema, "admin")
                st.rerun()

        st.divider()

        # History
        st.markdown("**History**")
        if st.button("➕ New Chat", use_container_width=True):
            new_chat = _new_chat(f"Chat {len(st.session_state.chats) + 1}")
            st.session_state.chats.append(new_chat)
            st.session_state.active_chat_id = new_chat["id"]
            save_history(st.session_state.chats)
            st.rerun()

        if st.session_state.chats:
            for chat in reversed(st.session_state.chats[-20:]):
                is_active = chat["id"] == st.session_state.active_chat_id
                label = f"{'➡️ ' if is_active else ''}{chat.get('title','Chat')}"
                if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                    st.session_state.active_chat_id = chat["id"]
                    st.rerun()
        else:
            st.caption("No chats yet.")

        st.divider()

        # API Status - Granular
        st.markdown("**API Status**")
        health = check_api_health()

        # Database Status
        if health.get("database_connected"):
            st.markdown(
                '<div class="status-item">🗄️ Database: <span style="color:green; margin-left:5px;"><b>Connected</b></span></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="status-item">🗄️ Database: <span style="color:red; margin-left:5px;"><b>Disconnected</b></span></div>', unsafe_allow_html=True)

        # Agent Status
        if health.get("agent_ready"):
            st.markdown(
                '<div class="status-item">🤖 Agent: <span style="color:green; margin-left:5px;"><b>Ready</b></span></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="status-item">🤖 Agent: <span style="color:red; margin-left:5px;"><b>Not Ready</b></span></div>', unsafe_allow_html=True)

        # Redis Status
        if health.get("redis_connected"):
            st.markdown(
                '<div class="status-item">📦 Redis: <span style="color:green; margin-left:5px;"><b>Connected</b></span></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="status-item">📦 Redis: <span style="color:orange; margin-left:5px;"><b>Disconnected</b> (Stateless)</span></div>', unsafe_allow_html=True)

        if health.get("status") != "healthy" and "error" in health:
            st.caption(f"Error: {health.get('error')}")

        st.divider()

        # Schema (lazy load) - requires auth
        st.markdown("**Database Schema**")
        if st.session_state.jwt_token:
            if st.button("📊 Load Schema", use_container_width=True):
                with st.spinner("Loading..."):
                    schema = get_database_schema(st.session_state.jwt_token)
                    if schema:
                        st.session_state.schema = schema
                        # Generate role-based insights
                        st.session_state.insights = generate_insights(
                            schema, st.session_state.user_role or "viewer"
                        )
                        st.success("Schema loaded!")
                    else:
                        st.error("Failed to load schema")
        else:
            st.caption("Login to load schema")

        if st.session_state.schema:
            with st.expander("View Schema"):
                st.json(st.session_state.schema, expanded=False)

        st.divider()

        # Delete current chat
        if st.button("🗑️ Delete Current Chat", use_container_width=True):
            active_id = st.session_state.active_chat_id
            st.session_state.chats = [c for c in st.session_state.chats if c["id"] != active_id]
            if st.session_state.chats:
                st.session_state.active_chat_id = st.session_state.chats[-1]["id"]
            else:
                new_chat = _new_chat("Chat 1")
                st.session_state.chats = [new_chat]
                st.session_state.active_chat_id = new_chat["id"]
            st.session_state.debug_events = []
            save_history(st.session_state.chats)
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point."""

    # Render sidebar (now contains the app logo/header)
    render_sidebar()

    # Ensure there is an active chat
    if not st.session_state.active_chat_id and st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
    if not st.session_state.chats:
        new_chat = _new_chat("Chat 1")
        st.session_state.chats.append(new_chat)
        st.session_state.active_chat_id = new_chat["id"]

    active_chat = get_active_chat()
    active_messages = active_chat["messages"] if active_chat else []

    # Conditional Header in Main Area: Only show if chat is empty
    if not active_messages:
        st.markdown('<div class="main-header">💬 NL2SQL Chat</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Ask questions about your database in natural language</div>',
            unsafe_allow_html=True
        )

    # Display chat history
    for message in active_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_response(
                    message.get("content", ""),
                    message.get("data"),
                    message.get("metadata", {}),
                    message.get("pii")
                )
            else:
                st.markdown(message["content"])

    # Handle pending user message (needs processing)
    if (
        active_messages
        and active_messages[-1]["role"] == "user"
        and not active_messages[-1].get("handled")
        and not st.session_state.is_processing
    ):
        user_question = active_messages[-1]["content"]

        # Check if authenticated
        if not st.session_state.jwt_token:
            with st.chat_message("assistant"):
                st.warning(
                    "⚠️ Please login first using the sidebar to ask questions.")
            active_messages.append({
                "role": "assistant",
                "content": "Please login first using the sidebar to ask questions.",
                "data": None,
                "metadata": {}
            })
            active_messages[-2]["handled"] = True
            st.rerun()

        # Set processing flag
        st.session_state.is_processing = True

        with st.chat_message("assistant"):
            # Thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("*Thinking...*")

            # Stream buffer - accumulates events cleanly
            stream_buffer = {
                "answer": "",
                "data": None,
                "metadata": {},
                "errors": [],
                "pii": None
            }
            received_any_event = False

            # Process SSE stream with JWT token
            for event in query_backend_stream(user_question, st.session_state.jwt_token):
                received_any_event = True
                event_type = event.get("type")

                # Log for debugging (invisible to user)
                st.session_state.debug_events.append(event)

                if event_type == "step_start":
                    step_name = event.get("step_name", "Processing")
                    thinking_placeholder.markdown(f"*{step_name}...*")

                elif event_type == "step_complete":
                    # Capture SQL results from execute_sql_tool
                    tool_name = event.get("tool_name")
                    status = event.get("status")
                    if tool_name == "execute_sql_tool" and status == "success":
                        try:
                            tool_result_str = event.get("tool_result", "{}")
                            sql_data = json.loads(tool_result_str)
                            if sql_data and ("columns" in sql_data or isinstance(sql_data, list)):
                                stream_buffer["data"] = sql_data
                        except (json.JSONDecodeError, TypeError):
                            pass  # Ignore parse errors

                elif event_type == "answer_chunk":
                    stream_buffer["answer"] += event.get("content", "")
                    # Progressive display
                    thinking_placeholder.markdown(stream_buffer["answer"])

                elif event_type == "done":
                    stream_buffer["answer"] = event.get(
                        "answer", stream_buffer["answer"])
                    # Use data from done event if available, otherwise keep the one from step_complete
                    done_data = event.get("data")
                    if done_data:
                        stream_buffer["data"] = done_data
                    stream_buffer["metadata"] = {
                        "execution_time": event.get("execution_time", 0),
                        "reasoning_steps": event.get("reasoning_steps", 0)
                    }
                    if event.get("pii") is not None:
                        stream_buffer["pii"] = event.get("pii")
                    break

                elif event_type == "error":
                    stream_buffer["errors"].append(
                        event.get("error", "Unknown error"))
                    break

            # If the stream returned nothing, fall back to non-streaming endpoint
            if not received_any_event and not stream_buffer["errors"]:
                fallback = query_backend_once(user_question, st.session_state.jwt_token)
                if fallback.get("error"):
                    stream_buffer["errors"].append(fallback.get("error"))
                else:
                    stream_buffer["answer"] = fallback.get("answer", "")
                    stream_buffer["data"] = fallback.get("data")
                    stream_buffer["metadata"] = {
                        "execution_time": fallback.get("execution_time", 0),
                        "reasoning_steps": fallback.get("reasoning_steps", 0)
                    }

            # Clear thinking indicator
            thinking_placeholder.empty()

            # Handle errors
            if stream_buffer["errors"]:
                error_msg = "I encountered an error processing your request."
                st.error(error_msg)
                stream_buffer["answer"] = error_msg

            # Render final response
            render_response(
                stream_buffer["answer"] or "No response received.",
                stream_buffer["data"],
                stream_buffer["metadata"],
                stream_buffer["pii"]
            )

        # Save assistant message
        active_messages.append({
            "role": "assistant",
            "content": stream_buffer["answer"] or "No response received.",
            "data": stream_buffer["data"],
            "metadata": stream_buffer["metadata"],
            "pii": stream_buffer["pii"]
        })

        # Mark user message as handled
        active_messages[-2]["handled"] = True

        # Set chat title from first user question
        if active_chat and active_chat.get("title", "").startswith("Chat"):
            active_chat["title"] = user_question[:60]

        # Regenerate insights based on the new context
        if st.session_state.schema:
            st.session_state.insights = generate_insights(
                st.session_state.schema,
                st.session_state.user_role or "viewer",
                user_question
            )

        # Reset processing flag
        st.session_state.is_processing = False

        # Persist history after assistant response
        save_history(st.session_state.chats)

        # Rerun to clean up UI
        st.rerun()

    # Chat input (disabled while processing)
    # Display suggested questions if available
    if st.session_state.insights and not st.session_state.is_processing:
        # Create a container for suggestions
        with st.container():
            st.markdown("###### 💡 Suggestions")
            # Use columns to display buttons horizontally
            # Limit to first 4 insights to avoid clutter
            suggestions = st.session_state.insights[:4]
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        active_messages.append(
                            {"role": "user", "content": suggestion, "handled": False})
                        st.rerun()

    if prompt := st.chat_input(
        "Ask a question about your database...",
        disabled=st.session_state.is_processing
    ):
        active_messages.append({
            "role": "user",
            "content": prompt,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "handled": False
        })
        save_history(st.session_state.chats)
        st.rerun()


# Entry point
if __name__ == "__main__":
    main()
