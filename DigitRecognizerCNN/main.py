from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io


app = FastAPI()


model = load_model("DigitRecognizer.keras")


@app.get("/")
def home():
    return {
        "message": "Digit Recognizer API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(
        io.BytesIO(await file.read())
    )

    image = image.convert("L")


    image = image.resize((28,28))


    image = np.array(image)


    image = image / 255.0

    image = image.reshape(1,28,28,1)


    prediction = model.predict(image)

    digit = np.argmax(prediction)


    return {
        "predicted_digit": int(digit)
    }