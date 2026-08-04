from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np


app = FastAPI()

model = tf.keras.models.load_model("FaceMaskDetector.keras")


@app.get("/")
def home():
    return {"message": "API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = Image.open(BytesIO(await file.read()))
    img = img.resize((128,128))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)

    if result[0][0] > 0.5:
        prediction = "without Mask"
    else:
        prediction = "With Mask"

    return {
        "result": prediction
    }