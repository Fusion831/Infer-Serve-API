# tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app
import pytest


@pytest.fixture(autouse=True)
def mock_redis(mocker):
    """
    Fixture to mock the redis.Redis client in the application's main module.
    It automatically runs for every test function in this file.
    """
    
    mock_redis_client = mocker.MagicMock()
    
    
    mocker.patch('app.main.redis.Redis', return_value=mock_redis_client)
    
    
    mock_redis_client.incr.return_value = 5 
    
    
    yield mock_redis_client

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

def test_predict_rate_limit_exceeded(mock_redis):
    """
    Tests that a 429 error is returned when the rate limit is exceeded.
    """
    # INSTEAD of using mocker.patch, we directly configure the mock
    # object that the fixture provided to us.
    mock_redis.incr.return_value = 11  # A value OVER the limit of 10

    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        
        # Now, the app will have received '11' from the mock, and this assert will pass.
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]