import logging
from enum import Enum
from typing import Dict, List

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class ModelConfig:
    NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    MAX_LENGTH: int = 512
    SCORE_DECIMALS: int = 4
    DEVICE: int = 0 if torch.cuda.is_available() else -1


_pipeline = None


def load_model() -> pipeline:
    """Load DistilBERT sentiment pipeline (singleton — initializes once).

    Returns:
        transformers.Pipeline ready for inference.

    Raises:
        RuntimeError: If model cannot be loaded from HuggingFace Hub.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        logger.info(f"Loading model: {ModelConfig.NAME} | device: {ModelConfig.DEVICE}")
        _pipeline = pipeline(
            task="sentiment-analysis",
            model=ModelConfig.NAME,
            truncation=True,
            max_length=ModelConfig.MAX_LENGTH,
            device=ModelConfig.DEVICE,
        )
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        raise RuntimeError(f"Could not initialize sentiment model: {exc}") from exc

    return _pipeline


def unload_model() -> None:
    """Release model from memory (called on app shutdown)."""
    global _pipeline
    _pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded.")


def _format_result(text: str, raw: Dict) -> Dict:
    label = SentimentLabel(raw["label"])
    return {
        "text": text,
        "label": label.value,
        "score": round(float(raw["score"]), ModelConfig.SCORE_DECIMALS),
        "sentiment": "Positive" if label == SentimentLabel.POSITIVE else "Negative",
    }


def predict_sentiment(text: str) -> Dict:
    """Predict sentiment for a single text input.

    Args:
        text: Non-empty string up to MAX_LENGTH characters.

    Returns:
        Dict with keys: text, label, score, sentiment.

    Raises:
        ValueError: If text is empty or exceeds max length.
        RuntimeError: If model inference fails.
    """
    if not text or not isinstance(text, str):
        raise ValueError("text must be a non-empty string.")
    if len(text) > ModelConfig.MAX_LENGTH:
        raise ValueError(f"text exceeds {ModelConfig.MAX_LENGTH} characters.")

    try:
        model = load_model()
        result = model(text)[0]
        return _format_result(text, result)
    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"Inference failed (len={len(text)}): {exc}")
        raise RuntimeError(f"Inference failed: {exc}") from exc


def predict_batch(texts: List[str]) -> List[Dict]:
    """Predict sentiment for a batch of texts.

    Args:
        texts: List of 1–32 non-empty strings.

    Returns:
        List of prediction dicts in the same order as input.

    Raises:
        ValueError: If batch is empty, exceeds 32 items, or has invalid text.
        RuntimeError: If model inference fails.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list.")
    if len(texts) > 32:
        raise ValueError("Maximum 32 texts per batch request.")
    for i, t in enumerate(texts):
        if not t or not isinstance(t, str):
            raise ValueError(f"texts[{i}] must be a non-empty string.")

    try:
        model = load_model()
        results = model(texts)
        return [_format_result(text, raw) for text, raw in zip(texts, results)]
    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"Batch inference failed ({len(texts)} texts): {exc}")
        raise RuntimeError(f"Batch inference failed: {exc}") from exc
