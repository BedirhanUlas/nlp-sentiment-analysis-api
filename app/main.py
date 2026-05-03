import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import ModelConfig, load_model, predict_batch, predict_sentiment, unload_model
from app.schemas import (
    BatchPredictRequest,
    BatchSentimentResponse,
    HealthResponse,
    PredictRequest,
    SentimentResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading NLP model...")
    load_model()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down — releasing model...")
    unload_model()


app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Production-ready REST API for real-time sentiment analysis "
        "powered by DistilBERT (HuggingFace Transformers)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with a unique ID and latency."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms}ms)"
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"Runtime error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Model inference unavailable. Please try again later."},
    )


@app.get("/", tags=["root"])
def root():
    return {
        "service": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_check():
    """Service liveness check — returns model name and device info."""
    return HealthResponse(
        status="ok",
        model=ModelConfig.NAME,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=SentimentResult,
    tags=["inference"],
    summary="Single text sentiment prediction",
)
def predict(request: PredictRequest):
    """Classify the sentiment of a single text (max 512 characters).

    Returns label (POSITIVE/NEGATIVE), confidence score, and
    friendly sentiment string.
    """
    if len(request.text) > ModelConfig.MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Text exceeds {ModelConfig.MAX_LENGTH} character limit.",
        )
    result = predict_sentiment(request.text)
    return SentimentResult(**result)


@app.post(
    "/predict/batch",
    response_model=BatchSentimentResponse,
    tags=["inference"],
    summary="Batch sentiment prediction (up to 32 texts)",
)
def predict_batch_endpoint(request: BatchPredictRequest):
    """Classify sentiment for up to 32 texts in a single request.

    More efficient than calling /predict repeatedly — runs a single
    forward pass through the model.
    """
    results = predict_batch(request.texts)
    return BatchSentimentResponse(
        results=[SentimentResult(**r) for r in results],
        total=len(results),
    )
