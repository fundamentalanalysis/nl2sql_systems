"""Streamlit chat interface for Tafe Analytical Agent."""
import streamlit as st
import requests
import json
import time
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configuration - use existing APP_HOST and APP_PORT from settings
API_HOST = os.getenv("APP_HOST")
API_PORT = os.getenv("APP_PORT")

# 0.0.0.0 is only for server binding, translate to localhost for client connections
if API_HOST == "0.0.0.0":
    API_HOST = "localhost"

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


# def check_api_health() -> Dict[str, Any]:
#     """Check if the API is healthy."""
#     try:
#         response = requests.get(f"{API_BASE_URL}/health", timeout=10)
#         return response.json()
#     except Exception as e:
#         return {"status": "error", "error": str(e)}

def check_api_health() -> Dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "ok"}
        return {"status": "error", "error": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_database_schema() -> Optional[Dict[str, Any]]:
    """Get the database schema from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/schema", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


def query_database_stream(question: str):
    """Send a query to the database and stream the response via SSE."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"question": question},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=600  # 10 minutes for complex analytical queries
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        event_data = line_str[6:]  # Remove 'data: ' prefix
                        yield json.loads(event_data)
        else:
            yield {'type': 'error', 'error': f'API returned status {response.status_code}'}

    except Exception as e:
        yield {'type': 'error', 'error': str(e)}


def format_indian_number(num):
    """
    Format a number using Indian number system with comma separators.
    Examples: 1,000 (thousand), 1,00,000 (lakh), 1,00,00,000 (crore)

    Args:
        num: Number to format (int, float, or string)

    Returns:
        Formatted string with Indian comma separators
    """
    if num is None or num == '':
        return num

    try:
        # Convert to string and handle decimals
        if isinstance(num, (int, float)):
            # Check if it's a whole number
            if isinstance(num, float) and num.is_integer():
                num_str = str(int(num))
                decimal_part = ""
            else:
                num_str = str(num)
                if '.' in num_str:
                    parts = num_str.split('.')
                    num_str = parts[0]
                    decimal_part = '.' + parts[1]
                else:
                    decimal_part = ""
        else:
            num_str = str(num)
            decimal_part = ""
            if '.' in num_str:
                parts = num_str.split('.')
                num_str = parts[0]
                decimal_part = '.' + parts[1]

        # Remove any existing commas
        num_str = num_str.replace(',', '')

        # Handle negative numbers
        negative = num_str.startswith('-')
        if negative:
            num_str = num_str[1:]

        # Only format if it's a number and has more than 3 digits
        if not num_str.isdigit():
            return num

        if len(num_str) <= 3:
            result = num_str
        else:
            # Indian formatting: last 3 digits, then groups of 2
            last_three = num_str[-3:]
            remaining = num_str[:-3]

            # Add commas every 2 digits from right to left in remaining part
            formatted_remaining = ''
            for i, digit in enumerate(reversed(remaining)):
                if i > 0 and i % 2 == 0:
                    formatted_remaining = ',' + formatted_remaining
                formatted_remaining = digit + formatted_remaining

            result = formatted_remaining + ',' + last_three

        # Add back negative sign and decimal part
        if negative:
            result = '-' + result
        result = result + decimal_part

        return result
    except (ValueError, AttributeError):
        # If conversion fails, return original value
        return num


def format_dataframe_indian(df):
    """
    Format numeric columns in a DataFrame using Indian number system.

    Args:
        df: pandas DataFrame

    Returns:
        Formatted DataFrame
    """
    import pandas as pd

    df_formatted = df.copy()

    for col in df_formatted.columns:
        # Check if column contains numeric data
        if pd.api.types.is_numeric_dtype(df_formatted[col]):
            # Apply Indian formatting to numeric columns
            df_formatted[col] = df_formatted[col].apply(format_indian_number)

    return df_formatted


def determine_chart_type(df, question: str) -> str:
    """
    Intelligently determine the best chart type based on dataframe structure and question.

    Args:
        df: pandas DataFrame with query results
        question: Original user question

    Returns:
        Chart type: 'metric', 'pie', 'bar', 'line', 'scatter', or 'table'
    """
    import pandas as pd

    num_rows = len(df)
    num_cols = len(df.columns)

    # Get column types
    numeric_cols = df.select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Check for date/time columns
    date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
            date_cols.append(col)

    question_lower = question.lower()

    # Rule 1: Single value - use metric card
    if num_rows == 1 and num_cols <= 2:
        return 'metric'

    # Rule 2: Time series - use line chart
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        if any(keyword in question_lower for keyword in ['trend', 'over time', 'progression', 'growth', 'change']):
            return 'line'

    # Rule 3: Few categories with proportions - use pie chart
    if len(text_cols) == 1 and len(numeric_cols) == 1 and num_rows <= 10:
        if any(keyword in question_lower for keyword in ['distribution', 'breakdown', 'share', 'proportion', 'percentage']):
            return 'pie'

    # Rule 4: Categorical comparison - use bar chart
    if len(text_cols) >= 1 and len(numeric_cols) >= 1:
        if any(keyword in question_lower for keyword in ['compare', 'top', 'bottom', 'ranking', 'by', 'each']):
            return 'bar'

    # Rule 5: Two numeric columns - scatter plot for correlation
    if len(numeric_cols) >= 2 and len(text_cols) <= 1:
        if any(keyword in question_lower for keyword in ['correlation', 'relationship', 'vs', 'versus']):
            return 'scatter'

    # Rule 6: Default to bar for categorical + numeric
    if len(text_cols) >= 1 and len(numeric_cols) >= 1:
        return 'bar'

    # Fallback to table view
    return 'table'


def format_chart_number_indian(num, add_rupee=False):
    """
    Format numbers for chart labels and tooltips with Indian formatting.

    Args:
        num: Number to format
        add_rupee: Whether to add ₹ symbol

    Returns:
        Formatted string
    """
    if num is None or num == '':
        return num

    try:
        # Handle very large numbers - convert to lakhs/crores
        if isinstance(num, (int, float)):
            if abs(num) >= 10000000:  # 1 crore
                formatted = f"{num/10000000:.2f} Cr"
            elif abs(num) >= 100000:  # 1 lakh
                formatted = f"{num/100000:.2f} L"
            else:
                formatted = format_indian_number(num)

            if add_rupee and num >= 1000:
                return f"₹{formatted}"
            return formatted

        return str(num)
    except:
        return str(num)


def create_visualization(df, chart_type: str, question: str):
    """
    Create a plotly visualization based on chart type.

    Args:
        df: pandas DataFrame with data
        chart_type: Type of chart to create
        question: Original question for context

    Returns:
        plotly figure object or None
    """
    import pandas as pd

    if chart_type == 'table' or df.empty:
        return None

    # Get numeric and text columns
    numeric_cols = df.select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    try:
        if chart_type == 'metric':
            # Metric card - display as a big number
            value_col = numeric_cols[0] if numeric_cols else df.columns[0]
            value = df[value_col].iloc[0]

            # Determine if it's a currency value
            is_currency = any(word in question.lower() for word in [
                              'revenue', 'sales', 'amount', 'price', 'cost', 'value', 'total'])

            # Format the value with Indian number system
            formatted_value = format_indian_number(value)
            if is_currency:
                formatted_value = f"₹{formatted_value}"

            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                number={
                    'font': {'size': 60},
                    'valueformat': '',  # We'll use custom formatting
                },
                title={
                    'text': f"<b>{value_col.replace('_', ' ').title()}</b><br><span style='font-size:48px'>{formatted_value}</span>"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=100, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=16)
            )
            # Hide the default number display since we're showing in title
            fig.update_traces(number_font_size=0)
            return fig

        elif chart_type == 'pie':
            # Pie chart
            label_col = text_cols[0] if text_cols else df.columns[0]
            value_col = numeric_cols[0] if numeric_cols else df.columns[1]

            fig = px.pie(
                df,
                names=label_col,
                values=value_col,
                title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
                hole=0.3  # Donut chart
            )

            # Custom hover template with Indian formatting
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Value: %{customdata}<br>Percentage: %{percent}<extra></extra>',
                customdata=[format_indian_number(val) for val in df[value_col]]
            )
            fig.update_layout(
                height=500,
                showlegend=True,
                font=dict(size=12)
            )
            return fig

        elif chart_type == 'bar':
            # Bar chart
            label_col = text_cols[0] if text_cols else df.columns[0]
            value_col = numeric_cols[0] if numeric_cols else df.columns[1]

            # Sort by value descending
            df_sorted = df.sort_values(by=value_col, ascending=False).head(20)

            # Determine if horizontal or vertical
            use_horizontal = len(df_sorted) > 7

            # Add formatted values for tooltips
            df_sorted['formatted_value'] = df_sorted[value_col].apply(
                format_indian_number)

            if use_horizontal:
                fig = px.bar(
                    df_sorted,
                    y=label_col,
                    x=value_col,
                    orientation='h',
                    title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
                    labels={value_col: value_col.replace(
                        '_', ' ').title(), label_col: ''}
                )
                # Reverse order for horizontal bars (highest at top)
                fig.update_yaxes(autorange="reversed")
                # Format x-axis with Indian number format
                fig.update_xaxes(
                    tickformat='',
                    tickmode='auto',
                    title=value_col.replace('_', ' ').title()
                )
                # Update hover template
                fig.update_traces(
                    customdata=df_sorted[['formatted_value']],
                    hovertemplate='<b>%{y}</b><br>Value: %{customdata[0]}<extra></extra>'
                )
            else:
                fig = px.bar(
                    df_sorted,
                    x=label_col,
                    y=value_col,
                    title=f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
                    labels={value_col: value_col.replace('_', ' ').title(
                    ), label_col: label_col.replace('_', ' ').title()}
                )
                # Format y-axis with Indian number format
                fig.update_yaxes(
                    tickformat='',
                    tickmode='auto',
                    title=value_col.replace('_', ' ').title()
                )
                # Update hover template
                fig.update_traces(
                    customdata=df_sorted[['formatted_value']],
                    hovertemplate='<b>%{x}</b><br>Value: %{customdata[0]}<extra></extra>'
                )

            # Color gradient
            fig.update_traces(
                marker=dict(
                    color=df_sorted[value_col],
                    colorscale='Viridis',
                    showscale=False
                )
            )

            # Format axis tick labels with Indian number system
            if use_horizontal:
                fig.update_xaxes(ticktext=[format_indian_number(val) for val in fig.data[0].x],
                                 tickvals=fig.data[0].x)
            else:
                fig.update_yaxes(ticktext=[format_indian_number(val) for val in fig.data[0].y],
                                 tickvals=fig.data[0].y)

            fig.update_layout(height=500, showlegend=False)
            return fig

        elif chart_type == 'line':
            # Line chart
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
                    date_col = col
                    break

            if date_col is None:
                date_col = text_cols[0] if text_cols else df.columns[0]

            value_col = numeric_cols[0] if numeric_cols else df.columns[1]

            # Add formatted values for tooltips
            df['formatted_value'] = df[value_col].apply(format_indian_number)

            fig = px.line(
                df,
                x=date_col,
                y=value_col,
                title=f"{value_col.replace('_', ' ').title()} Over Time",
                labels={value_col: value_col.replace('_', ' ').title(
                ), date_col: date_col.replace('_', ' ').title()},
                markers=True
            )
            fig.update_traces(
                line=dict(width=3, color='#667eea'),
                marker=dict(size=8),
                customdata=df[['formatted_value']],
                hovertemplate='<b>%{x}</b><br>Value: %{customdata[0]}<extra></extra>'
            )

            # Format y-axis with Indian number format
            fig.update_yaxes(
                ticktext=[format_indian_number(val) for val in fig.data[0].y],
                tickvals=fig.data[0].y,
                title=value_col.replace('_', ' ').title()
            )

            fig.update_layout(height=500, hovermode='x unified')
            return fig

        elif chart_type == 'scatter':
            # Scatter plot
            x_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else df.columns[1]

            # Add formatted values for tooltips
            df['formatted_x'] = df[x_col].apply(format_indian_number)
            df['formatted_y'] = df[y_col].apply(format_indian_number)

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
                labels={x_col: x_col.replace('_', ' ').title(
                ), y_col: y_col.replace('_', ' ').title()},
                trendline="ols"  # Add trend line
            )
            fig.update_traces(
                marker=dict(size=10, color='#667eea', opacity=0.6),
                customdata=df[['formatted_x', 'formatted_y']],
                hovertemplate='<b>X: %{customdata[0]}<br>Y: %{customdata[1]}</b><extra></extra>',
                selector=dict(mode='markers')
            )
            fig.update_layout(height=500)
            return fig

    except Exception as e:
        # If chart creation fails, return None to fall back to table
        print(f"Chart creation failed: {e}")
        return None

    return None


# Page configuration
st.set_page_config(
    page_title="Tafe Analytical Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        min-height: 600px;
        max-height: 700px;
        overflow-y: auto;
    }
    
    /* Reasoning step styling */
    .reasoning-step {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 4px solid #667eea;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    .step-icon {
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .step-success {
        border-left-color: #28a745;
    }
    
    .step-running {
        border-left-color: #ffc107;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .step-error {
        border-left-color: #dc3545;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Message styling */
    .stChatMessage {
        background: transparent;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar text colors for better visibility */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1a202c !important;  /* Almost black */
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown {
        color: #2d3748 !important;  /* Very dark gray */
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: #4a5568 !important;  /* Dark gray */
    }
    
    /* Ensure button text in sidebar is visible */
    [data-testid="stSidebar"] button {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat input */
    .stChatInputContainer {
        border-top: 2px solid rgba(102, 126, 234, 0.2);
        padding-top: 1rem;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'schema' not in st.session_state:
    st.session_state.schema = None

# Header
st.markdown('<div class="main-header">🤖 Tafe Analytical Agent</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your database in natural language</div>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Health Check
    st.subheader("🔌 API Status")
    health = check_api_health()

    if health.get("status") == "ok":
        st.success("✅ All Systems Operational")
        st.caption(f"Database: Connected")
        st.caption(f"Agent: Ready")
    elif health.get("status") == "degraded":
        st.warning("⚠️ Degraded")
        st.caption(
            f"Database: {'✓' if health.get('database_connected') else '✗'}")
        st.caption(f"Agent: {'✓' if health.get('agent_ready') else '✗'}")
    else:
        st.error("❌ API Unavailable")
        st.caption("Cannot connect to API")

    st.divider()

    # Database Schema
    st.subheader("📊 Database Schema")
    if st.button("🔄 Load Schema", use_container_width=True):
        with st.spinner("Loading schema..."):
            schema = get_database_schema()
            if schema:
                st.session_state.schema = schema
                st.success("Schema loaded!")

    if st.session_state.schema:
        with st.expander("View Schema Details"):
            st.json(st.session_state.schema, expanded=False)

    st.divider()

    # Example Questions
    st.subheader("💡 Quick Questions")
    examples = [
        "How many records are in the database?",
        "Show me the top 10 customers",
        "What is the average order value?",
        "Show me recent orders"
    ]

    for example in examples:
        if st.button(f"📝 {example[:30]}...", key=f"ex_{hash(example)}", use_container_width=True):
            # Add to chat
            st.session_state.messages.append(
                {"role": "user", "content": example})
            st.rerun()

    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Chat Messages Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display reasoning steps if present
        if "reasoning_steps" in message and message["reasoning_steps"]:
            # Use st.status for consistent UI with streaming
            state = "complete"
            # Check if any step failed
            if any(step.get('status') == 'error' for step in message["reasoning_steps"]):
                state = "error"
                label = "Error occurred during reasoning"
            else:
                label = "Reasoning complete"

            with st.status(label, expanded=False, state=state):
                for step in message["reasoning_steps"]:
                    step_name = step['step_name']
                    status = step.get('status', 'success')

                    if status == 'error':
                        st.write(f"❌ **{step_name}** (Failed)")
                    else:
                        st.write(f"✅ **{step_name}**")

        # Display metadata if present
        if "metadata" in message:
            meta = message["metadata"]
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ {meta.get('execution_time', 0)}s")
            with col2:
                st.caption(f"🔄 {meta.get('reasoning_steps', 0)} steps")

# Chat Input
if prompt := st.chat_input("Ask a question about your database..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        # Use st.status for a "Gemini Deep Research" style reasoning display
        status_container = st.status("Thinking...", expanded=True)

        reasoning_steps = []
        answer_chunks = []
        metadata = {}

        try:
            # Stream the response
            for event in query_database_stream(prompt):
                event_type = event.get('type')

                if event_type == 'step_start':
                    step_name = event['step_name']
                    # Update status label to show current action
                    status_container.update(label=step_name, state="running")
                    status_container.write(f"**{step_name}**...")

                    step_info = {
                        'step_name': step_name,
                        'status': 'running'
                    }
                    reasoning_steps.append(step_info)

                elif event_type == 'step_complete':
                    step_name = event['step_name']
                    status = event['status']

                    # Update the last step status in our list
                    if reasoning_steps:
                        reasoning_steps[-1]['status'] = status

                    if status == 'error':
                        status_container.write(
                            f"❌ Error in {step_name}: {event.get('error', 'Unknown error')}")
                    else:
                        # We don't need to write success explicitly as the next step will appear or we finish
                        pass

                    # Check for SQL results to display
                    if event.get('tool_name') == 'execute_sql_tool' and status == 'success':
                        try:
                            tool_result_str = event.get('tool_result', '{}')
                            # The tool result is a JSON string, so we parse it
                            sql_data = json.loads(tool_result_str)

                            if 'columns' in sql_data and 'rows' in sql_data:
                                import pandas as pd
                                df = pd.DataFrame(
                                    sql_data['rows'], columns=sql_data['columns'])
                                if not df.empty:
                                    # Determine best chart type
                                    chart_type = determine_chart_type(
                                        df, prompt)

                                    # Create visualization
                                    fig = create_visualization(
                                        df, chart_type, prompt)

                                    if fig is not None:
                                        # Display the chart
                                        st.plotly_chart(
                                            fig, use_container_width=True)

                                        # Show table as expandable option below the chart
                                        df_formatted = format_dataframe_indian(
                                            df)
                                        with st.expander("📋 View Raw Data Table", expanded=False):
                                            st.dataframe(
                                                df_formatted, use_container_width=True)
                                            st.caption(
                                                f"Returned {len(df)} rows")
                                    else:
                                        # Fallback to table if visualization fails
                                        df_formatted = format_dataframe_indian(
                                            df)
                                        with st.expander("📊 View Query Results", expanded=True):
                                            st.dataframe(
                                                df_formatted, use_container_width=True)
                                            st.caption(
                                                f"Returned {len(df)} rows")
                        except Exception as e:
                            # If parsing fails, just ignore (don't break the chat)
                            pass

                elif event_type == 'answer_chunk':
                    # When we start getting the answer, we can collapse the status
                    status_container.update(
                        label="Reasoning complete", state="complete", expanded=False)
                    answer_chunks.append(event['content'])
                    answer_placeholder.markdown(''.join(answer_chunks))

                elif event_type == 'done':
                    metadata = {
                        'execution_time': event['execution_time'],
                        'reasoning_steps': event['reasoning_steps']
                    }
                    status_container.update(
                        label="Reasoning complete", state="complete", expanded=False)

                    # Show metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"⏱️ {metadata['execution_time']}s")
                    with col2:
                        st.caption(f"🔄 {metadata['reasoning_steps']} steps")

                elif event_type == 'error':
                    error_msg = event['error']
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        st.error(
                            "⚠️ OpenAI Rate Limit Reached. Please wait a moment and try again.")
                    else:
                        st.error(f"⚠️ Error: {error_msg}")
                    status_container.update(
                        label="Error occurred", state="error", expanded=True)

            # Save assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": ''.join(answer_chunks) if answer_chunks else "I encountered an error processing your question.",
                "reasoning_steps": reasoning_steps,
                "metadata": metadata
            }
            st.session_state.messages.append(assistant_message)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            status_container.update(label="Error", state="error")
