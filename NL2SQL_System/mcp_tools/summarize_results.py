"""MCP Tool D: summarize_results - Generate intelligent summaries of query results."""
from typing import Dict, List, Any
from loguru import logger
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
import json


def summarize_results(question: str, columns: List[str], rows: List[List[Any]], row_count: int) -> Dict[str, str]:
    """
    Generate an intelligent, human-readable summary of query results.

    This tool uses an LLM to analyze query results and provide insights, patterns,
    trends, and business implications in natural language.

    Args:
        question: Original natural language question
        columns: List of column names from query results
        rows: Query result rows (list of lists)
        row_count: Number of rows in results

    Returns:
        Dict containing the summary:
        {
            "summary": "Natural language summary with insights..."
        }

    Raises:
        Exception: If LLM call fails
    """
    logger.info(f"Generating summary for {row_count} rows")

    # Early exit for empty or all-NULL aggregates
    if row_count == 0:
        return {"summary": "No matching records were found for this query."}
    if (
        row_count == 1
        and rows
        and isinstance(rows[0], list)
        and all(val is None for val in rows[0])
    ):
        return {"summary": "No matching records were found for this query (the result returned NULL)."}

    # EARLY EXIT: multi-table COUNT aggregation
    if (
        isinstance(rows, list)
        and rows
        and isinstance(rows[0], list)
        and rows[0]
        and isinstance(rows[0][0], dict)
    ):
        return _summarize_multi_table_counts(question, rows)

    try:
        # Initialize LLM
        llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_ai_endpoint,
            api_key=SecretStr(settings.azure_ai_api_key),
            model=settings.azure_openai_deployment,
            api_version=settings.azure_ai_api_version,
            temperature=0.3,  # Slightly creative for better summaries
        )

        # Format results for prompt
        results_preview = _format_results_for_prompt(columns, rows, row_count)

        # Check if question is analysis-related
        is_analytical = _is_analytical_question(question)

        # Perform statistical analysis if needed
        statistical_insights = ""
        if is_analytical and row_count > 1:
            statistical_insights = _perform_statistical_analysis(columns, rows)

        # Always provide analysis (even for single-row results)
        stats_section = ""
        if statistical_insights:
            stats_section = f"""

Statistical Analysis:
{statistical_insights}
"""

        system_prompt = f"""You are a data analyst expert for an Indian-based application. Your task is to analyze query results and provide insightful, human-readable summaries.

Your summary should include:
1. Direct answer to the user's question
2. Key findings and patterns in the data
3. Statistical insights (counts, averages, trends, distributions, etc.)
4. Best results or recommendations based on the analysis
5. Business implications or notable observations
6. Any anomalies or interesting data points

Guidelines:
- Be concise but comprehensive
- Use specific numbers and data points from the statistical analysis
- Highlight the most important insights first and identify the BEST results
- Use clear, professional language
- Avoid technical jargon when possible
- If the result set is large, focus on top-level insights and representative examples
- For analytical questions, provide data-driven recommendations

INDIAN LOCALIZATION (MANDATORY):
- ALL monetary amounts MUST be in Indian Rupees (₹) unless specified otherwise
- ALWAYS use Indian number system: thousands, lakhs, crores
- NEVER use millions or billions - convert to lakhs and crores
- Number formatting rules:
  * Up to 99,999: Use thousands format (e.g., ₹45,000 or 45 thousand)
  * 1,00,000 to 99,99,999: Use lakhs (e.g., ₹12.5 lakhs or ₹50 lakhs)
  * 1,00,00,000 and above: Use crores (e.g., ₹2.5 crores or ₹100 crores)
- Examples:
  * 150000 → ₹1.5 lakhs (NOT $150K or 150 thousand)
  * 5000000 → ₹50 lakhs (NOT 5 million)
  * 75000000 → ₹7.5 crores (NOT 75 million)
  * 250000000 → ₹25 crores (NOT 250 million)
- For averages/statistics: ₹2.3 lakhs average, ₹15.6 crores total, etc.

Original Question: {{question}}
{stats_section}
Query Results:
Columns: {{columns}}
Total Rows: {{row_count}}
Data Preview:
{{results_preview}}
"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please provide the answer/summary based on these results.")
        ])

        # Generate summary
        chain = prompt_template | llm
        response = chain.invoke({
            "question": question,
            "columns": ", ".join(columns),
            "row_count": row_count,
            "results_preview": results_preview
        })

        summary = response.content.strip()

        logger.info("Successfully generated summary")
        return {"summary": summary}

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise


def _is_analytical_question(question: str) -> bool:
    """
    Determine if the question is asking for analysis.

    Args:
        question: The user's natural language question

    Returns:
        True if the question appears to be analytical in nature
    """
    analytical_keywords = [
        'analyz', 'analyse', 'compare', 'comparison', 'trend', 'pattern',
        'distribution', 'correlation', 'statistic', 'average', 'mean',
        'median', 'variance', 'standard deviation', 'insight', 'performance',
        'best', 'worst', 'top', 'bottom', 'rank', 'segment', 'breakdown',
        'vs', 'versus', 'difference', 'change', 'growth', 'decline'
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in analytical_keywords)


def _perform_statistical_analysis(columns: List[str], rows: List[List[Any]]) -> str:
    """
    Perform statistical analysis on numeric columns in the result set.

    Args:
        columns: Column names
        rows: Result rows

    Returns:
        Formatted string with statistical insights
    """
    if not rows:
        return ""

    insights = []

    # For each column, check if it contains numeric data
    for col_idx, col_name in enumerate(columns):
        numeric_values = []

        for row in rows:
            if col_idx < len(row) and row[col_idx] is not None:
                val = row[col_idx]
                # Try to convert to float
                try:
                    if isinstance(val, (int, float)):
                        numeric_values.append(float(val))
                    elif isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit():
                        numeric_values.append(float(val))
                except (ValueError, TypeError):
                    pass

        # If we have numeric values, calculate statistics
        if len(numeric_values) >= 2:
            try:
                import statistics

                mean_val = statistics.mean(numeric_values)
                median_val = statistics.median(numeric_values)
                min_val = min(numeric_values)
                max_val = max(numeric_values)

                stats_text = f"Column '{col_name}':"
                stats_text += f"\n  - Count: {len(numeric_values)}"
                stats_text += f"\n  - Mean: {mean_val:.2f}"
                stats_text += f"\n  - Median: {median_val:.2f}"
                stats_text += f"\n  - Min: {min_val:.2f}"
                stats_text += f"\n  - Max: {max_val:.2f}"

                # Add standard deviation if we have enough values
                if len(numeric_values) >= 2:
                    try:
                        stdev_val = statistics.stdev(numeric_values)
                        stats_text += f"\n  - Std Dev: {stdev_val:.2f}"
                    except statistics.StatisticsError:
                        pass

                insights.append(stats_text)
            except Exception as e:
                logger.debug(
                    f"Could not calculate statistics for column {col_name}: {e}")

    if insights:
        return "\n\n".join(insights)

    return ""


def _format_results_for_prompt(columns: List[str], rows: List[List[Any]], row_count: int, max_rows: int = 20) -> str:
    """
    Format query results into a readable string for the LLM prompt.

    Args:
        columns: Column names
        rows: Result rows
        row_count: Total number of rows
        max_rows: Maximum number of rows to include in preview (default: 20)

    Returns:
        Formatted results string
    """
    if row_count == 0:
        return "No results returned from the query."

    lines = []

    # Header
    lines.append(" | ".join(columns))
    lines.append("-" * (len(" | ".join(columns))))

    # Data rows (limited to max_rows)
    preview_rows = rows[:max_rows]
    for row in preview_rows:
        # Convert row values to strings, handling None/NULL values
        row_str = " | ".join(
            str(val) if val is not None else "NULL" for val in row)
        lines.append(row_str)

    # Add note if there are more rows
    if row_count > max_rows:
        lines.append(f"\n... and {row_count - max_rows} more rows")

    return "\n".join(lines)

# Summarize multi-table counts - helper function for _summarize_results


def _summarize_multi_table_counts(question: str, results: List[List[Dict[str, Any]]]) -> Dict[str, str]:
    total = 0
    breakdown = []

    for result in results:
        if result and isinstance(result[0], dict):
            for key, value in result[0].items():
                table_name = (
                    key.replace("_count", "")
                       .replace("record_", "")
                )
                breakdown.append(f"- {table_name}: {value}")
                total += int(value)

    summary = (
        f"The database contains a total of {total:,} records across multiple tables.\n\n"
        "Breakdown by table:\n" +
        "\n".join(breakdown)
    )

    return {"summary": summary}


if __name__ == "__main__":
    # Test the tool
    test_columns = ["id", "name", "age", "city"]
    test_rows = [
        [1, "Alice", 30, "New York"],
        [2, "Bob", 25, "Los Angeles"],
        [3, "Charlie", 35, "Chicago"],
        [4, "Diana", 28, "Houston"],
        [5, "Eve", 32, "Phoenix"]
    ]

    result = summarize_results(
        question="Show me all users in the database",
        columns=test_columns,
        rows=test_rows,
        row_count=len(test_rows)
    )

    print(result["summary"])
