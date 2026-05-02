from transformers import pipeline
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_sentiment_pipeline = None


def load_model():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        _sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            truncation=True,
            max_length=512,
        )
        logger.info("Model loaded successfully.")
    return _sentiment_pipeline


def predict_sentiment(text: str) -> Dict:
    model = load_model()
    result = model(text)[0]
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 4),
        "sentiment": "Positive" if result["label"] == "POSITIVE" else "Negative",
    }


def predict_batch(texts: List[str]) -> List[Dict]:
    model = load_model()
    results = model(texts)
    return [
        {
            "text": text,
            "label": r["label"],
            "score": round(r["score"], 4),
            "sentiment": "Positive" if r["label"] == "POSITIVE" else "Negative",
        }
        for text, r in zip(texts, results)
    ]
