import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from clari_gen.orchestrator.ambiguity_pipeline import AmbiguityPipeline
from clari_gen.api.schemas import (
    QueryRequest,
    ClarifyRequest,
    ConfirmRequest,
    QueryResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: AmbiguityPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Initializing AmbiguityPipeline...")
    # Initialize with default settings.
    # In a real app, you might want to load config from env vars.
    pipeline = AmbiguityPipeline()
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Clarification API", lifespan=lifespan)


@app.post("/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a new query.
    If ambiguous, returns 'status': 'clarification_needed' and a 'context' blob.
    If clear, returns 'status': 'completed' and the original query as 'reformulated_query'.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    logger.info(f"Received query: {request.text}")

    # Process query without a callback (stops at AWAITING_CLARIFICATION if ambiguous)
    query_result = pipeline.process_query(request.text, clarification_callback=None)

    response = QueryResponse(
        status=query_result.status.value.lower(),
        original_query=query_result.original_query,
        error_message=query_result.error_message,
    )

    if query_result.status == "COMPLETED" or query_result.status == "NOT_AMBIGUOUS":
        response.status = "completed"
        response.reformulated_query = query_result.get_final_output()

    elif query_result.status == "AWAITING_CLARIFICATION":
        response.status = "clarification_needed"
        response.clarifying_question = query_result.clarifying_question
        response.context = query_result.to_dict()

    elif query_result.status == "ERROR":
        response.status = "error"
        raise HTTPException(status_code=500, detail=query_result.error_message)

    return response


@app.post("/v1/clarify", response_model=QueryResponse)
async def process_clarification(request: ClarifyRequest):
    """
    Process a clarification response.
    Requires 'context' blob from the previous /query response.
    Returns the reformulated query with AWAITING_CONFIRMATION status.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    logger.info(
        f"Received clarification for query: {request.context.get('original_query', 'unknown')}"
    )

    query_result = pipeline.continue_with_clarification(
        query_dict=request.context, user_clarification=request.answer
    )

    response = QueryResponse(
        status=query_result.status.value.lower(),
        original_query=query_result.original_query,
        error_message=query_result.error_message,
    )

    if query_result.status.value == "AWAITING_CONFIRMATION":
        response.status = "confirmation_needed"
        response.reformulated_query = query_result.reformulated_query
        response.context = query_result.to_dict()

    elif query_result.status.value == "ERROR":
        response.status = "error"
        raise HTTPException(status_code=500, detail=query_result.error_message)

    return response


@app.post("/v1/confirm", response_model=QueryResponse)
async def process_confirmation(request: ConfirmRequest):
    """
    Process a confirmation response.
    Requires 'context' blob from the previous /clarify response.
    Accepts user confirmation (yes/no) and optional alternative query.
    Returns the final confirmed query.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    logger.info(
        f"Received confirmation for query: {request.context.get('original_query', 'unknown')}"
    )

    query_result = pipeline.confirm_reformulation(
        query_dict=request.context,
        confirmation=request.confirmation,
        alternative_query=request.alternative_query,
    )

    response = QueryResponse(
        status=query_result.status.value.lower(),
        original_query=query_result.original_query,
        reformulated_query=query_result.reformulated_query,
        confirmed_query=query_result.confirmed_query,
        error_message=query_result.error_message,
    )

    if query_result.status.value == "COMPLETED":
        response.status = "completed"

    elif query_result.status.value == "ERROR":
        response.status = "error"
        raise HTTPException(status_code=500, detail=query_result.error_message)

    return response


@app.get("/health")
async def health_check():
    return {"status": "ok"}
