# Sentiment Analysis API

A production-ready REST API for NLP-based sentiment analysis, powered by **DistilBERT** (HuggingFace Transformers) and served with **FastAPI**. Containerized with **Docker** for easy deployment.

---

## Overview

This project fine-tunes and deploys a transformer-based NLP model that classifies text as **Positive** or **Negative** with a confidence score. Built with production best practices: typed schemas, health checks, batch inference, and full test coverage.

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`
**Dataset:** Stanford Sentiment Treebank (SST-2) — 67,349 movie review sentences
**Accuracy:** ~91.3% on SST-2 validation set

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | HuggingFace Transformers (DistilBERT) |
| API Framework | FastAPI |
| Data Validation | Pydantic v2 |
| Server | Uvicorn (ASGI) |
| Containerization | Docker + Docker Compose |
| Testing | Pytest |
| Language | Python 3.11 |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Welcome message and links |
| GET | /health | Health check with model info |
| POST | /predict | Single text sentiment prediction |
| POST | /predict/batch | Batch prediction (up to 32 texts) |

---

## Getting Started

### Run with Docker

```bash
git clone https://github.com/BedirhanUlas/nlp-sentiment-analysis-api.git
cd nlp-sentiment-analysis-api
docker-compose up --build
```

API: http://localhost:8000 | Docs: http://localhost:8000/docs

### Run Locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Author

**Bedirhan Ulas** — Data Scientist & Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/bedirhan-ulas) | [GitHub](https://github.com/BedirhanUlas)
