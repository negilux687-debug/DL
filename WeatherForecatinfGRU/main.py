from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import numpy as np


app=FastAPI()


with open("gru_model.pkl","rb") as f:
    model=pickle.load(f)


with open("scaler.pkl","rb") as f:
    scaler=pickle.load(f)



class WeatherInput(BaseModel):
    temperature: List[float]



@app.get("/")
def home():
    return {
        "message":"GRU Weather Prediction API Running"
    }



@app.post("/predict")
def predict(data:WeatherInput):

    values=data.temperature

    arr=np.array(values).reshape(-1,1)

    scaled=scaler.transform(arr)

    X=np.array([scaled[-3:]])

    prediction=model.predict(X)

    result=scaler.inverse_transform(prediction)


    return {
        "next_temperature":float(result[0][0])
    }