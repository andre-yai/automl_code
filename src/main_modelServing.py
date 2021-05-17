from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from libs.model_serving.ModelInference  import Prediction

app = FastAPI()


# Schema para o endpoint para o intervalo de datas
class PredictionData(BaseModel):
    data: list = []

# Endpoint padrao

print("CONFIG FILE:: ",os.environ["ConfigFile"])

pred = Prediction(os.environ["ConfigFile"])

@app.get("/")
def read_root():
        return {"API": f"API ML Serving"}

@app.post("/predict/")
def predict_item(item: PredictionData):
    return pred.predict(item)