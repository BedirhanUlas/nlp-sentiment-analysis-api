from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive():
    response = client.post("/predict", json={"text": "This is absolutely fantastic, I love it!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["score"] > 0.5
    assert data["sentiment"] == "Positive"


def test_predict_negative():
    response = client.post("/predict", json={"text": "This was a terrible experience, I hated it."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert data["score"] > 0.5


def test_predict_batch():
    response = client.post("/predict/batch", json={
        "texts": [
            "Great product, highly recommend!",
            "Worst purchase I have ever made.",
            "It was okay, nothing special."
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
