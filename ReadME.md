# InferServe API: Foundational Drill

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Testing](https://img.shields.io/badge/Testing-Pytest-blueviolet.svg)](https://pytest.org/)

A lightweight, containerized REST API built with FastAPI and Docker, designed to serve a pre-trained Scikit-learn model for real-time predictions.

## Overview

This project isolates and masters the foundational skill of building a clean, production-style API for a machine learning model. A FastAPI application loads a pre-trained Scikit-learn model upon startup. It exposes a single `POST` endpoint, `/predict`, which accepts a JSON payload of features. The application uses Pydantic for robust data validation, passes the data to the model, and returns the prediction as a JSON response. The entire application is encapsulated in a single, efficient Docker image.

### Key Features
*   **FastAPI:** For building a high-performance, modern API.
*   **Pydantic:** For robust, type-hint-based data validation and error handling.
*   **Scikit-learn:** For serving a pre-trained machine learning model.
*   **Docker:** For containerizing the application, ensuring portability and reproducible builds.
*   **Pytest:** For a comprehensive test suite that validates endpoint functionality.

## Project Structure
```
infer-serve-api/
├── app/
│   ├── __init__.py         # Makes 'app' a Python package
│   ├── main.py             # FastAPI application logic
│   └── model.pkl           # Pre-trained Scikit-learn model
├── images/
│   └── confusion_matrix.png # Model performance visualization
├── tests/
│   └── test_main.py        # Pytest suite for the API
├── .dockerignore           # Specifies files to exclude from Docker image
├── Dockerfile              # Instructions for building the Docker image
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
└── train_model.py          # Script to train and save the model
```

## Getting Started

### Prerequisites
*   Git
*   Docker Desktop

### How to Run
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd infer-serve-api
    ```

2.  **Build the Docker image:**
    This command builds the image from the `Dockerfile` and tags it with the name `infer-serve-api`.
    ```bash
    docker build -t infer-serve-api .
    ```

3.  **Run the Docker container:**
    This command starts a container from the image. The `-p 8000:8000` flag maps port 8000 on your local machine to port 8000 inside the container.
    ```bash
    docker run -p 8000:8000 infer-serve-api
    ```

4.  **Access the API:**
    The API will now be running. You can access the interactive documentation (Swagger UI) in your browser at:
    **[http://localhost:8000/docs](http://localhost:8000/docs)**

## Running the Tests
To run the automated test suite locally, you'll need a Python virtual environment.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\Activate.ps1`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install pytest httpx
    ```

3.  **Run the test suite:**
    From the project root directory, run `pytest`.
    ```bash
    pytest
    ```

## API Endpoint Documentation

### `POST /predict`
Accepts a JSON object with four features of the Iris flower and returns the predicted class.

#### Request Body
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Success Response (200 OK)
```json
{
  "predicted_class": "setosa"
}
```

#### Error Response (422 Unprocessable Entity)
If the request body is malformed (e.g., missing a field or has the wrong data type), Pydantic provides a detailed error response.
```json
{
  "detail": [
    {
      "loc": [
        "body",
        "petal_width"
      ],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
## Model Performance
The served Logistic Regression model achieves 100% accuracy on the test set for the Iris dataset. The confusion matrix below visualizes its performance, showing zero misclassifications.

![Confusion Matrix for Iris Classification](images/confusion_matrix.png)

## Design Decisions & Engineering Insights

During development, several key questions guided the final implementation, reflecting a focus on robust, production-oriented practices.

*   **Why Feature Scaling?**
    Logistic Regression is sensitive to the scale of input features. By using `StandardScaler`, we ensure that the model learns the true predictive power of each feature, leading to a more reliable model.

*   **How should the ML model be loaded in a production API?**
    Loading a model from disk is a slow operation. This API uses FastAPI's `lifespan` event handler to load the model into memory **once at startup**. This minimizes latency for every API call, ensuring high performance.

*   **What's the best way to handle class label mapping?**
    The model predicts a number (e.g., `1`), but the API must return a string (e.g., `"versicolor"`). Instead of loading the entire training dataset into the API, a more professional pattern would be to save and load a separate, lightweight `class_names.json` file. This decouples the API from the training data and keeps its memory footprint low.