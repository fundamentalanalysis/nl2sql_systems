"""FastAPI application for MySQL Analytical Agent."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import json
import asyncio
from loguru import logger
from app.logger import app_logger
from app.config import settings
from app.auth import get_current_user
from agent.agent import get_agent
from database.connection import db_manager


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(...,
                          description="Natural language question about the database")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How many users are in the database?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str = Field(...,
                        description="Natural language answer with insights")
    execution_time: float = Field(...,
                                  description="Time taken to process the query in seconds")
    reasoning_steps: int = Field(...,
                                 description="Number of reasoning steps the agent took")
    error: Optional[str] = Field(
        None, description="Error message if query failed")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The database contains 1,250 users. The user base has grown by 15% this month...",
                "execution_time": 3.45,
                "reasoning_steps": 4,
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    database_connected: bool
    agent_ready: bool
    redis_connected: bool = False


# Create FastAPI app
app = FastAPI(
    title="MySQL Analytical Agent",
    description="LLM-powered agent that converts natural language questions to SQL queries and provides intelligent insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware using configuration from .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins.split(
        ",") if settings.cors_allow_origins != "*" else ["*"],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(
        ",") if settings.cors_allow_methods != "*" else ["*"],
    allow_headers=settings.cors_allow_headers.split(
        ",") if settings.cors_allow_headers != "*" else ["*"],
)


# Global agent instance
agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on application startup."""
    global agent

    logger.info("Starting MySQL Analytical Agent API")
    logger.info(
        f"Configuration: {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")

    # Test database connection
    if not db_manager.test_connection():
        logger.warning("Database connection test failed during startup")

    # Initialize agent
    try:
        agent = get_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        # Continue startup even if agent fails - will handle in requests


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "MySQL Analytical Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Verifies that the database is connected and the agent is ready.
    """
    database_connected = db_manager.test_connection()
    agent_ready = agent is not None
    
    from app.services.redis import redis_client
    redis_connected = redis_client.is_connected

    status = "healthy" if database_connected and agent_ready and redis_connected else "degraded"

    return HealthResponse(
        status=status,
        database_connected=database_connected,
        agent_ready=agent_ready,
        redis_connected=redis_connected
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_database(request: QueryRequest, user: dict = Depends(get_current_user)):
    """Process a natural language question about the database with RBAC enforcement.

    This endpoint:
    1. Retrieves the database schema (filtered by user role)
    2. Converts the question to SQL using LLM
    3. Executes the SQL query safely with RBAC validation
    4. Generates an intelligent summary with insights

    Args:
        request: QueryRequest containing the natural language question
        user: Authenticated user context from JWT

    Returns:
        QueryResponse with answer, execution time, and reasoning steps

    Raises:
        HTTPException: If agent is not initialized or query processing fails critically
    """
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please check server logs."
        )

    logger.info(f"Received query: {request.question}")

    try:
        # Process the query through the agent with role context
        result = agent.query(request.question, role=user["role"])

        # Log the result
        logger.info(
            f"Query completed: {result['execution_time']}s, {result['reasoning_steps']} steps")

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Critical error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/query/stream", tags=["Query"])
async def query_database_stream(request: QueryRequest):
    """
    Process a natural language question with streaming response.
    
    **TEMPORARY**: Authentication disabled for demo purposes.
    All requests are treated as admin role.

    Returns Server-Sent Events (SSE) stream with real-time updates:
    - Reasoning step start/complete events
    - Answer chunks for streaming display
    - Final done event with metadata

    Args:
        request: QueryRequest containing the natural language question

    Returns:
        StreamingResponse with SSE events

    Raises:
        HTTPException: If agent is not initialized
    """
    if agent is None:
        logger.error("Agent not initialized")
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please check server logs."
        )

    logger.info(f"Received streaming query: {request.question}")
    print(f"DEBUG: Endpoint /query/stream reached for question: {request.question}", flush=True)
    
    # TEMPORARY: Hardcode admin role (remove authentication for demo)
    user = {"user_id": "demo_user", "role": "admin"}

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from agent stream."""
        try:
            if agent is None:
                raise RuntimeError("Agent is not initialized")
            for event in agent.query_stream(request.question, role=user["role"]):
                # Format as SSE: data: {json}\n\n
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_event = {
                'type': 'error',
                'error': str(e),
                'execution_time': 0,
                'reasoning_steps': 0
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/schema", tags=["Database"])
async def get_schema(user: dict = Depends(get_current_user)):
    """
    Get the current database schema filtered by user role.

    Returns the schema including tables, columns, and data types,
    filtered according to the user's RBAC permissions.

    Args:
        user: Authenticated user context from JWT

    Returns:
        Filtered schema based on user role
    """
    from mcp_tools.get_schema import get_schema as mcp_get_schema

    try:
        schema = mcp_get_schema(role=user["role"])
        return schema
    except Exception as e:
        logger.error(f"Failed to retrieve schema: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schema: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )
