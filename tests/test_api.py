
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "SepsisGuard-X" in response.text


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "Hour": 5,
        "HR": 90,
        "O2Sat": 97,
        "SBP": 120,
        "MAP": 85,
        "Resp": 18,
        "Age": 55,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": 5,
    }

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert 0 <= data["probability"] <= 1
    assert data["model_name"] == "Calibrated Logistic Regression"
    assert "disclaimer" in data


def test_prediction_rejects_missing_features():
    response = client.post("/api/predict", json={"HR": 90})

    assert response.status_code == 422


def test_explain_endpoint():
    payload = {
        "Hour": 5,
        "HR": 90,
        "O2Sat": 97,
        "SBP": 120,
        "MAP": 85,
        "Resp": 18,
        "Age": 55,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": 5,
    }

    response = client.post("/api/explain", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert 0 <= data["probability"] <= 1
    assert len(data["feature_contributions"]) == 10
    assert data["model_name"] == "Calibrated Logistic Regression"
    assert "disclaimer" in data
    assert "not causal effects" in data["warning"]


def test_explain_endpoint_rejects_missing_features():
    response = client.post("/api/explain", json={"HR": 90})

    assert response.status_code == 422
