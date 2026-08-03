from fastapi import FastAPI
import pickle
import numpy as np

from typing import List

app = FastAPI()


with open("lstm_model.pkl", "rb") as file:
    model = pickle.load(file)



with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)



@app.get("/")
def home():
    return {
        "message": "LSTM Prediction API Running"
    }



@app.post("/predict")


@app.post("/predict")
def predict(prices: List[float]):

    data = np.array(prices).reshape(-1,1)

    scaled_data = scaler.transform(data)

    X = np.array([scaled_data[-3:]])

    prediction = model.predict(X)

    result = scaler.inverse_transform(prediction)

    return {
        "predicted_price": float(result[0][0])
    }