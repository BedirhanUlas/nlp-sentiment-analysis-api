from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="This movie was absolutely amazing!")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32, example=["Great product!", "Terrible experience."])


class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    sentiment: str


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
