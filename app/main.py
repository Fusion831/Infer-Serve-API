from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load('app/model.pkl')
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


