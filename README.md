# Sentiment Analysis API

Production-ready REST API for real-time NLP sentiment analysis, powered by **DistilBERT** (HuggingFace Transformers) and served via **FastAPI**. Containerized with Docker for one-command deployment.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Client                            │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────────────┐
│               FastAPI Application                     │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  /predict  │  │/predict/batch│  │   /health   │  │
│  └─────┬──────┘  └──────┬───────┘  └─────────────┘  │
│        └────────────────┘                             │
│                 │                                     │
│    ┌────────────▼────────────────────┐               │
│    │  DistilBERT (distilbert-base-   │               │
│    │  uncased-finetuned-sst-2-english│               │
│    │  via HuggingFace Transformers)  │               │
│    └─────────────────────────────────┘               │
└──────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
  Docker Container         Pydantic v2
  (Uvicorn ASGI)         (type-safe schemas)
```

## Model Performance

| Metric | Value |
|---|---|
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Training Dataset | Stanford Sentiment Treebank (SST-2), 67,349 sentences |
| Validation Accuracy | ~91.3% |
| Inference latency (CPU, p50) | ~120ms |
| Inference latency (GPU, p50) | ~8ms |
| Model size | 255MB |

DistilBERT retains 97% of BERT's accuracy at 40% of the size — the right tradeoff for production serving.

## Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | HuggingFace Transformers (DistilBERT) |
| API Framework | FastAPI |
| Data Validation | Pydantic v2 |
| Server | Uvicorn (ASGI) |
| Containerization | Docker + Docker Compose |
| Testing | pytest |
| Language | Python 3.11 |

## Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/BedirhanUlas/nlp-sentiment-analysis-api.git
cd nlp-sentiment-analysis-api
docker-compose up --build
```

API: http://localhost:8000 | Swagger Docs: http://localhost:8000/docs

### Option 2: Local

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product completely exceeded my expectations!"}'
```

**Response:**
```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "text_length": 51
}
```

### `POST /predict/batch`

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Absolutely loved it!", "Worst purchase ever.", "It was okay."]}'
```

**Response:**
```json
{
  "results": [
    {"label": "POSITIVE", "score": 0.9997, "text_length": 20},
    {"label": "NEGATIVE", "score": 0.9994, "text_length": 20},
    {"label": "NEGATIVE", "score": 0.6821, "text_length": 12}
  ],
  "count": 3
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "device": "cpu"
}
```

## Project Structure

```
nlp-sentiment-analysis-api/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── model.py         # DistilBERT model loader + inference
│   └── schemas.py       # Pydantic request/response models
├── tests/
│   └── test_api.py      # API integration tests
├── Dockerfile           # Production container
├── docker-compose.yml   # One-command deployment
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Use Cases

- **E-commerce** — Real-time product review scoring
- **Social listening** — Brand sentiment monitoring from Twitter/Reddit feeds
- **Customer support** — Automatic ticket priority based on message sentiment
- **Content moderation** — Flag negative/toxic content for review

## License

MIT

## Author

**Bedirhan Ulas** — Data Scientist & ML Engineer
[LinkedIn](https://www.linkedin.com/in/bedirhan-ulas) · [GitHub](https://github.com/BedirhanUlas)
