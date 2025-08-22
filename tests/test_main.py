# tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app

# NO client is created here anymore.

def test_predict_success():
    
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_class" in response.json()


def test_predict_malformed():
    
    with TestClient(app) as client:
        payload = {"sepal_length": "invalid_data"} 
        response = client.post("/predict", json=payload)
        assert response.status_code == 422