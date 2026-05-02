from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        examples=["This movie was absolutely amazing!"],
    )


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        examples=[["Great product!", "Terrible experience."]],
    )


class SentimentResult(BaseModel):
    text: str
    label: str          # POSITIVE or NEGATIVE
    score: float        # confidence 0.0 - 1.0
    sentiment: str      # human-friendly: "Positive" / "Negative"


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
