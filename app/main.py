from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris
import os
import redis
from contextlib import asynccontextmanager

Rate_limit = 10
REDIS_HOST = os.getenv("Redis_host","localhost")
REDIS_PORT = int(os.getenv("Redist_port",6379))
REDIS_PASSWORD = os.getenv("REDIS_Password",None)



class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load('app/model.pkl')
    print(f"Connecting to Redis port at: {REDIS_PORT}")
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)



@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API"}



@app.post("/predict")
def predict(request: Request,features: IrisFeatures):
    model = request.app.state.model
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(input_data)
    iris = load_iris()
    if isinstance(iris, tuple):
        iris_data = iris[0]
    else:
        iris_data = iris
    predicted_class = iris_data.target_names[prediction[0]]
    return {"predicted_class": predicted_class}


