
#==============================
# Main Class for model Serving
#=============================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

from libs.model_serving.ModelInference  import Prediction

app = FastAPI()


class PredictionData(BaseModel):
    data: List = [[]]

pred = Prediction(os.environ["ConfigFile"], os.environ["ModelFolder"])

@app.get("/")
def read_root():
        return {"API": f"API ML Serving"}

@app.post(f"/predict/")
def predict_item(item: PredictionData):
    #======================================
    # This is responsable for calling model predictions in /predict 
    # It receives a data metrics example in diabetes model  data = [[2,180,74,24,21,23.9091702,1.488172308,22]] and returns it result for the model. 
    #============================

    return pred.predict(item.data)