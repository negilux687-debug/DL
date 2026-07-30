from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = FastAPI(title="Breast Cancer Detection API")


model = load_model("breast_cancer_ann_model.keras")

scaler = pickle.load(open("breast_cancer_scaler.pkl", "rb"))


class CancerInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Breast cancer detection API is running"}


@app.post("/predict")
def predict(data: CancerInput):

  
    input_data = np.array(data.features).reshape(1, -1)

    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)

    
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        result = "Cancer Detected "
    else:
        result = "No Cancer Detected "

    return {
        "input": data.features,
        "prediction": result,
        "value": int(predicted_class)
    }