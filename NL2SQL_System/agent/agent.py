"""LangChain agent setup with tool calling."""
from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from loguru import logger
from pydantic import SecretStr
from app.config import settings
from agent.tools import LANGCHAIN_TOOLS
from mcp_tools.get_schema import get_schema as mcp_get_schema
import time
import json


class MySQLAnalyticalAgent:
    """MySQL Analytical Agent using LangChain with tool calling."""

    def __init__(self):
        """Initialize the agent with LLM and tools."""
        logger.info("Initializing MySQL Analytical Agent")
        
        # Set Azure OpenAI API key as environment variable (required by langchain-openai)
        import os
        os.environ["OPENAI_API_KEY"] = settings.azure_ai_api_key

        # Initialize LLM with tools
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_ai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.azure_ai_api_version,
            temperature=0,
            timeout=180.0  # 3 minute timeout for complex analytical queries
        )

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(LANGCHAIN_TOOLS)

        logger.info("Agent initialized successfully")

    def query(self, question: str, role: str = "admin") -> Dict[str, Any]:
        """
        Process a natural language question and return the analysis.
        RBAC is currently disabled (forcing admin access).
        """
        # Force admin role for full access
        role = "admin"

        print(f"Processing query: {question}", flush=True)
        print(f"🚀 Starting End-to-End Pipeline for query: '{question}'", flush=True)
        
        start_time = time.time()
        try:
            # 1. Mandatory Schema Preloading
            # Fetch RBAC-filtered schema before planning starts
            schema_dict = mcp_get_schema(role)
            schema_str = json.dumps(schema_dict, indent=2)

            # System message with injected schema
            system_msg = SystemMessage(content=f"""You are a MySQL analytical agent for an Indian-based application. 
            
CURRENT USER ROLE: {role}

ALLOWED DATABASE SCHEMA:
```json
{schema_str}
```

WORKFLOW - Follow these steps IN ORDER:
1. Call pii_detect_tool(question) to find sensitive information.
2. Call pii_encode_tool(question) to mask PII with tokens.
   - Use the 'encoded_text' from this step for all subsequent logic.
3. Call get_schema_tool(role='{role}') to review allowed tables/columns.
4. Call generate_sql_tool(encoded_question, db_schema) using the MASKED question.
5. Call execute_sql_tool(sql, role='{role}') to run the query.
6. Call summarize_results_tool(encoded_question, results) to create insights.
7. Call pii_decode_tool(summary) to restore tokens to original values before answering.

IMPORTANT RULES:
- NEVER send plain-text PII to SQL generation tools. Use tokens (e.g., [PERSON_XXXX]).
- Use Indian number system (lakhs, crores) and Rupees (₹) for currency.
- If information is missing from schema, say "I cannot answer that with the available data."
""")

            human_msg = HumanMessage(content=question)

            messages = [system_msg, human_msg]
            steps = 0
            max_iterations = 10

            # Agent loop - call tools until we get a final answer
            while steps < max_iterations:
                response = self.llm_with_tools.invoke(messages)
                steps += 1

                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    messages.append(response)

                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']

                        # CUSTOM LOGGING: Step-specific logs matching user request
                        if tool_name == 'pii_detect_tool':
                            print("🔐 Step 1: Detecting and Encoding PII entities...", flush=True)
                        elif tool_name == 'pii_encode_tool':
                            print("   - Encoding PII entities...", flush=True)
                        elif tool_name == 'generate_sql_tool':
                            print("🧠 Step 2: Generating SQL query using LLM...", flush=True)
                            if 'question' in tool_args:
                                print(f"   - Encoded Query: {tool_args['question']}", flush=True)
                        elif tool_name == 'execute_sql_tool':
                            print("📊 Step 3: Executing SQL against Database...", flush=True)
                            if 'sql_query' in tool_args:
                                print(f"   - Generated SQL: {tool_args['sql_query']}", flush=True)
                        elif tool_name == 'summarize_results_tool':
                            print("💬 Step 4: Generating natural language summary...", flush=True)
                        elif tool_name == 'pii_decode_tool':
                            print("🔓 Step 5: Decoding PII entities in final response...", flush=True)

                        # Find and execute the tool
                        tool = next(
                            (t for t in LANGCHAIN_TOOLS if t.name == tool_name), None)
                        if tool:
                            try:
                                tool_result = tool.invoke(tool_args)

                                # Additional Logging for Results
                                if tool_name == 'pii_detect_tool':
                                    # Log detected count
                                    if isinstance(tool_result, dict) and 'count' in tool_result:
                                        print(f"   - Found {tool_result['count']} PII entities", flush=True)
                                    
                                elif tool_name == 'pii_encode_tool':
                                    print(f"   - Encoded Result: {str(tool_result)[:100]}...", flush=True)
                                    
                                elif tool_name == 'execute_sql_tool':
                                    try:
                                        res_len = 0
                                        if isinstance(tool_result, list):
                                            res_len = len(tool_result)
                                        elif isinstance(tool_result, str):
                                            try:
                                                parsed = json.loads(tool_result)
                                                if isinstance(parsed, list):
                                                    res_len = len(parsed)
                                            except:
                                                pass
                                        print(f"   - Execution Success: Found {res_len} rows", flush=True)
                                    except:
                                        print("   - Execution Success", flush=True)

                                # Add tool result to messages
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call['id']
                                ))
                            except Exception as e:
                                print(f"Tool {tool_name} failed: {e}", flush=True)
                                logger.error(f"Tool {tool_name} failed: {e}")
                                messages.append(ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_call['id']
                                ))
                else:
                    # No more tool calls - we have the final answer
                    final_answer = response.content
                    execution_time = time.time() - start_time
                    
                    print("   - Final Answer ready", flush=True)
                    print("✅ Pipeline Complete: Success", flush=True)

                    return {
                        "answer": final_answer,
                        "execution_time": round(execution_time, 2),
                        "reasoning_steps": steps
                    }

            # Max iterations reached
            execution_time = time.time() - start_time
            print("Pipeline failed: Max iterations reached", flush=True)
            return {
                "answer": "I reached the maximum number of steps without completing the analysis. Please try a simpler question.",
                "execution_time": round(execution_time, 2),
                "reasoning_steps": steps,
                "error": "Max iterations reached"
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")

            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "execution_time": round(execution_time, 2),
                "reasoning_steps": 0,
                "error": str(e)
            }

    def query_stream(self, question: str, role: str = "admin"):
        """
        Process a natural language question and yield streaming events.
        RBAC is currently disabled (forcing admin access).
        """
        # Force admin role for full access
        role = "admin"
        """
        Process a natural language question and yield streaming events.

        Yields events for:
        - Reasoning step start (tool being called)
        - Reasoning step complete (tool result)
        - Final answer chunks (for streaming display)

        Args:
            question: Natural language question about the database

        Yields:
            Dict events with types: 'step_start', 'step_complete', 'answer_chunk', 'done', 'error'
        """
        logger.info(f"Processing query: {question}")
        logger.info(f"🚀 Starting End-to-End Pipeline for query: '{question}'")
        print(f"Processing query: {question}", flush=True)
        print(f"🚀 Starting End-to-End Pipeline for query: '{question}'", flush=True)
        start_time = time.time()

        try:
            # System message with workflow instructions and role awareness
            system_msg = SystemMessage(content=f"""You are a MySQL analytical agent for an Indian-based application. 

WORKFLOW - Follow these steps IN ORDER:
1. Call pii_detect_tool(question) to find sensitive information.
2. Call pii_encode_tool(question) to mask PII with tokens.
3. Call get_schema_tool(role='{role}') to get the database schema.
4. Call generate_sql_tool(encoded_question, db_schema) using the masked question.
5. Call execute_sql_tool(sql, role='{role}') to run the query.
6. Call summarize_results_tool(encoded_question, results) to create insights.
7. Call pii_decode_tool(summary) to restore original values for the final answer.

IMPORTANT:
- Complete ALL steps in order.
- NEVER send plain-text names or numbers to the SQL tool.
- Your final response must be the DECODED summary.
- Use Indian number system (thousands, lakhs, crores).
- All monetary values in Indian Rupees (₹).
""")

            human_msg = HumanMessage(content=question)

            messages = [system_msg, human_msg]
            steps = 0
            max_iterations = 10

            # Map tool names to user-friendly step names
            step_names = {
                'pii_detect_tool': 'Analyzing Privacy Risk',
                'pii_encode_tool': 'Masking Sensitive Data',
                'get_schema_tool': 'Getting Database Schema',
                'generate_sql_tool': 'Generating SQL Query',
                'execute_sql_tool': 'Executing Query',
                'summarize_results_tool': 'Analyzing Results',
                'pii_decode_tool': 'Restoring Sensitive Data'
            }

            # Capture query results for visualization
            last_query_results = None
            last_query_results_decoded = None
            last_pii_summary = None

            # Agent loop - call tools until we get a final answer
            while steps < max_iterations:
                response = self.llm_with_tools.invoke(messages)
                steps += 1

                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    messages.append(response)

                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']

                        # CUSTOM LOGGING: Step-specific logs matching user request
                        if tool_name == 'pii_detect_tool':
                            logger.info("🔐 Step 1: Detecting and Encoding PII entities...")
                            print("🔐 Step 1: Detecting and Encoding PII entities...", flush=True)
                        elif tool_name == 'pii_encode_tool':
                            logger.info("   - Encoding PII entities...")
                            print("   - Encoding PII entities...", flush=True)
                        elif tool_name == 'generate_sql_tool':
                            logger.info("🧠 Step 2: Generating SQL query using LLM...")
                            print("🧠 Step 2: Generating SQL query using LLM...", flush=True)
                            if 'question' in tool_args:
                                logger.info(f"   - Encoded Query: {tool_args['question']}")
                                print(f"   - Encoded Query: {tool_args['question']}", flush=True)
                        elif tool_name == 'execute_sql_tool':
                            logger.info("📊 Step 3: Executing SQL against Database...")
                            print("📊 Step 3: Executing SQL against Database...", flush=True)
                            if 'sql_query' in tool_args:
                                logger.info(f"   - Generated SQL: {tool_args['sql_query']}")
                                print(f"   - Generated SQL: {tool_args['sql_query']}", flush=True)
                        elif tool_name == 'summarize_results_tool':
                            logger.info("💬 Step 4: Generating natural language summary...")
                            print("💬 Step 4: Generating natural language summary...", flush=True)
                        elif tool_name == 'pii_decode_tool':
                            logger.info("🔓 Step 5: Decoding PII entities in final response...")
                            print("🔓 Step 5: Decoding PII entities in final response...", flush=True)

                        # Emit step start event
                        yield {
                            'type': 'step_start',
                            'step_name': step_names.get(tool_name, tool_name),
                            'tool_name': tool_name,
                            'step_number': steps
                        }

                        # Find and execute the tool
                        tool = next(
                            (t for t in LANGCHAIN_TOOLS if t.name == tool_name), None)
                        if tool:
                            try:
                                tool_result = tool.invoke(tool_args)

                                # Additional Logging for Results
                                if tool_name == 'pii_detect_tool':
                                    # Log detected count
                                    if isinstance(tool_result, dict) and 'count' in tool_result:
                                        logger.info(f"   - Found {tool_result['count']} PII entities")
                                        print(f"   - Found {tool_result['count']} PII entities", flush=True)
                                        try:
                                            entities = tool_result.get("entities", [])
                                            last_pii_summary = {
                                                "count": tool_result.get("count", 0),
                                                "entities": [
                                                    {
                                                        "type": ent.get("entity_type"),
                                                        "value": question[ent.get("start", 0):ent.get("end", 0)],
                                                        "start": ent.get("start"),
                                                        "end": ent.get("end")
                                                    }
                                                    for ent in entities
                                                ]
                                            }
                                        except Exception:
                                            last_pii_summary = {
                                                "count": tool_result.get("count", 0),
                                                "entities": []
                                            }
                                    
                                elif tool_name == 'pii_encode_tool':
                                    logger.info(f"   - Encoded Result: {str(tool_result)[:100]}...")
                                    print(f"   - Encoded Result: {str(tool_result)[:100]}...", flush=True)

                                elif tool_name == 'execute_sql_tool':
                                    try:
                                        # Parse the result if it's JSON-like
                                        if isinstance(tool_result, str):
                                            last_query_results = json.loads(tool_result)
                                        else:
                                            last_query_results = tool_result
                                    except:
                                        last_query_results = tool_result
                                    
                                    # Decode for frontend display (LLM still receives masked results)
                                    try:
                                        from privacy.decoder import decode_results
                                        last_query_results_decoded = decode_results(last_query_results)
                                    except Exception:
                                        last_query_results_decoded = last_query_results
                                    
                                    try:
                                        res_len = 0
                                        if isinstance(last_query_results, list):
                                            res_len = len(last_query_results)
                                        logger.info(f"   - Execution Success: Found {res_len} rows")
                                        print(f"   - Execution Success: Found {res_len} rows", flush=True)
                                    except:
                                        logger.info("   - Execution Success")
                                        print("   - Execution Success", flush=True)

                                # Emit step complete event
                                yield {
                                    'type': 'step_complete',
                                    'step_name': step_names.get(tool_name, tool_name),
                                    'tool_name': tool_name,
                                    'step_number': steps,
                                    'status': 'success',
                                    'tool_result': str(tool_result)
                                }

                                # Add tool result to messages
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call['id']
                                ))
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}")

                                # Emit error event for this step
                                yield {
                                    'type': 'step_complete',
                                    'step_name': step_names.get(tool_name, tool_name),
                                    'tool_name': tool_name,
                                    'step_number': steps,
                                    'status': 'error',
                                    'error': str(e)
                                }

                                messages.append(ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_call['id']
                                ))
                else:
                    # No more tool calls - we have the final answer
                    final_answer = response.content
                    
                    logger.info("   - Final Answer ready")
                    logger.info("✅ Pipeline Complete: Success")
                    print("   - Final Answer ready", flush=True)
                    print("✅ Pipeline Complete: Success", flush=True)

                    if isinstance(final_answer, list):
                        final_answer = ' '.join(
                            str(item) for item in final_answer if isinstance(item, str))
                    execution_time = time.time() - start_time

                    # Stream the answer in chunks (split by sentences or words)
                    words = final_answer.split(' ')
                    for i, word in enumerate(words):
                        yield {
                            'type': 'answer_chunk',
                            'content': word + (' ' if i < len(words) - 1 else '')
                        }

                    # Emit done event with query results for visualization
                    done_event = {
                        'type': 'done',
                        'execution_time': round(execution_time, 2),
                        'reasoning_steps': steps
                    }

                    # Include query results if available
                    if last_query_results:
                        done_event['data'] = last_query_results_decoded or last_query_results
                        logger.info(f"Sending {len(last_query_results) if isinstance(last_query_results, list) else '?'} result rows to frontend")
                    if last_pii_summary is not None:
                        done_event['pii'] = last_pii_summary
                    
                    yield done_event
                    return

            # Max iterations reached
            execution_time = time.time() - start_time
            yield {
                'type': 'error',
                'error': 'Max iterations reached',
                'execution_time': round(execution_time, 2),
                'reasoning_steps': steps
            }

        except Exception as e:
            import traceback
            
            execution_time = time.time() - start_time
            error_msg = f"Streaming query processing failed: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")

            yield {
                'type': 'error',
                'error': str(e),
                'execution_time': round(execution_time, 2),
                'reasoning_steps': 0
            }


# Global agent instance (initialized on first use)
_agent_instance = None


def get_agent() -> MySQLAnalyticalAgent:
    """
    Get or create the global agent instance.

    Returns:
        MySQLAnalyticalAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MySQLAnalyticalAgent()
    return _agent_instance


if __name__ == "__main__":
    # Test the agent
    agent = get_agent()

    test_question = "My name is Sreekanth and I live in Bangalore. How many orders have I placed?"
    result = agent.query(test_question)

    print("\n" + "="*80)
    print("ANSWER:")
    print(result["answer"])
    print(f"\nExecution Time: {result['execution_time']}s")
    print(f"Reasoning Steps: {result['reasoning_steps']}")
    print("="*80)
