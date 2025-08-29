# InferServe-Stack: A Multi-Service ML API with CI/CD

[![Python CI](https://github.com/Fusion831/Infer-Serve-API/actions/workflows/main.yml/badge.svg)](https://github.com/Fusion831/Infer-Serve-API/actions/workflows/main.yml)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Orchestration](https://img.shields.io/badge/Orchestration-Docker%20Compose-blue.svg)](https://docs.docker.com/compose/)
[![Testing](https://img.shields.io/badge/Testing-Pytest-blueviolet.svg)](https://pytest.org/)

This project is a complete, multi-service, production-style machine learning API stack. It features a FastAPI application that serves a Scikit-learn model, a Redis database for stateful rate-limiting, and a professional CI/CD pipeline using GitHub Actions for automated testing. The entire stack is orchestrated with Docker Compose for easy, one-command deployment.

## Overview

This project evolves a simple model-serving API into a robust, interconnected system. The FastAPI application exposes a `/predict` endpoint that is protected by a rate limiter. On receiving a request, the API first checks the client's IP against a counter in Redis. If the request count is within the limit (10 requests/minute), it serves a prediction from the ML model. If the limit is exceeded, it returns a `429 Too Many Requests` error. The entire process is automated and validated by a GitHub Actions workflow that runs the `pytest` suite on every push.

### Key Features
*   **FastAPI & Pydantic:** For a high-performance API with robust data validation.
*   **Redis:** For fast, in-memory data storage to handle stateful tasks like rate limiting.
*   **Docker & Docker Compose:** For containerizing the individual services and orchestrating the multi-container application stack.
*   **Professional CI/CD:** A GitHub Actions workflow automatically runs the test suite on every push.
*   **Advanced Testing:** The `pytest` suite uses mocking to test the rate-limiting logic in isolation.
*   **Secure Configuration:** Uses a `.env` file to manage secrets, keeping them separate from version-controlled code.

## Project Structure
```
infer-serve-stack/
├── .github/
│   └── workflows/
│       └── main.yml        # GitHub Actions CI workflow
├── app/
│   ├── main.py             # FastAPI logic with Redis integration
│   └── model.pkl
├── images/
│   └── confusion_matrix.png # Model performance visualization
├── requirements/
│   ├── requirements-dev.txt # Testing dependencies
│   └── requirements.txt     # Production dependencies
├── tests/
│   └── test_main.py        # Pytest suite with rate-limit tests
├── .dockerignore
├── .env.example            # Example environment variables
├── Dockerfile
├── docker-compose.yml      # Orchestrates the API and Redis services
└── ...
```

## Getting Started

### Prerequisites
*   Git
*   Docker & Docker Compose

### How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fusion831/Infer-Serve-API.git
    cd Infer-Serve-API
    ```

2.  **Create your environment file:**
    Copy the example file to a new `.env` file. This file will hold your local configuration and secrets.
    ```bash
    cp .env.example .env
    ```
    *You can modify the `REDIS_PASSWORD` in the `.env` file if you wish.*

3.  **Launch the application stack:**
    This command will build the API image from its Dockerfile and start both the `api` and `redis` containers as defined in `docker-compose.yml`.
    ```bash
    docker-compose up --build
    ```

4.  **Access the API:**
    The API will now be running. You can access the interactive documentation (Swagger UI) in your browser at:
    **[http://localhost:8000/docs](http://localhost:8000/docs)**
    Try sending more than 10 requests in a minute to see the rate limiting in action!

## Running the Tests
The test suite includes checks for the prediction logic, input validation, and rate limiting.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\Activate.ps1`
    ```

2.  **Install all dependencies:**
    ```bash
    pip install -r requirements/requirements.txt
    pip install -r requirements/requirements-dev.txt
    ```

3.  **Run the test suite:**
    From the project root directory, run `pytest`.
    ```bash
    pytest
    ```

## API Endpoint Documentation

### `POST /predict`
Accepts a JSON object with Iris flower features and returns the predicted class, subject to rate limiting.

#### Request Body
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Success Response (`200 OK`)
```json
{
  "predicted_class": "setosa"
}
```

#### Rate Limit Error Response (`429 Too Many Requests`)
Returned if the client IP has made more than 10 requests in the last minute.
```json
{
  "detail": "Rate limit exceeded. Please try again later. Limit is 10 requests per minute."
}
```

## Model Performance
The served Logistic Regression model achieves 100% accuracy on the test set for the Iris dataset. The confusion matrix below visualizes its performance, showing zero misclassifications.

![Confusion Matrix for Iris Classification](images/confusion_matrix.png)

## Design Decisions & Engineering Insights

This project demonstrates a series of design patterns that are fundamental to building robust, production-grade services.

*   **Why Feature Scaling?**
    Logistic Regression is sensitive to the scale of input features. By using `StandardScaler` during model training, we ensure that the model learns the true predictive power of each feature, leading to a more reliable model.

*   **Efficient Model Loading:**
    Loading a model from disk can be slow. This API uses FastAPI's `lifespan` event handler to load the model into memory **once at startup**, minimizing latency for every API call and ensuring high performance.

*   **Orchestration with Docker Compose:**
    Docker Compose allows us to define our entire multi-service application in a single YAML file. It automatically handles networking, startup order (`depends_on`), and configuration, which is essential for creating a reliable and reproducible development environment.

*   **Secrets Management with `.env`:**
    Secrets like database passwords should never be committed to Git. This project uses a `.env` file to store local secrets. The `docker-compose.yml` file then reads these secrets and injects them into the appropriate containers as environment variables. This is a standard practice for separating configuration from code.

*   **The CI/CD Pipeline:**
    The GitHub Actions workflow serves as an automated safety net. It runs our entire test suite in a clean, isolated environment on every push. This guarantees that new code doesn't break existing functionality and provides immediate feedback, allowing for confident and rapid development.