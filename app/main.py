from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.model import load_model, predict_sentiment, predict_batch
from app.schemas import (
    PredictRequest,
    BatchPredictRequest,
    SentimentResult,
    BatchSentimentResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading NLP model...")
    load_model()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for sentiment analysis powered by DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {"message": "Sentiment Analysis API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        version="1.0.0",
    )


@app.post("/predict", response_model=SentimentResult, tags=["Sentiment"])
def predict(request: PredictRequest):
    """Predict sentiment for a single text input."""
    try:
        result = predict_sentiment(request.text)
        return SentimentResult(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Sentiment"])
def predict_batch_endpoint(request: BatchPredictRequest):
    """Predict sentiment for a batch of texts (up to 32)."""
    try:
        results = predict_batch(request.texts)
        return BatchSentimentResponse(
            results=[SentimentResult(**r) for r in results],
            total=len(results),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
