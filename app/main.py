from fastapi import FastAPI, Request,HTTPException
from pydantic import BaseModel
import joblib
from sklearn.datasets import load_iris
import os
import redis
from redis.exceptions import ConnectionError
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
    try: 
        app.state.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        app.state.redis.ping()
        app.state.redis.ping()
    except ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        app.state.redis = None
        app.state.redis = None
    yield
    app.state.model = None
    if app.state.redis:
        app.state.redis.close()

app = FastAPI(lifespan=lifespan)



@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API"}



@app.post("/predict")
def predict(request: Request,features: IrisFeatures):
    redis_client = request.app.state.redis
    if not redis_client:
        raise HTTPException(status_code = 500, detail = "Internal Server Error, Redis not connected.")
    client_ip = request.client.host if request.client else "unknown"
    redis_key = f"rate-limit:{client_ip}"
    current_request = redis_client.incr(redis_key)
    if current_request == 1:
        redis_client.expire(redis_key, 60)
    if current_request > Rate_limit:
        raise HTTPException(
            status_code = 429,
            detail = f"Too many request, limit is {Rate_limit} per minute"
        )
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


