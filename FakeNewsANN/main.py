from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pickle


app = FastAPI(title="Fake News Detection API")



model = load_model("fake_news_ann_model.keras")


with open("fake_news_tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


class NewsInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake news detection API is running"}


@app.post("/predict")
def predict(data: NewsInput):

    text_vector = tfidf.transform([data.text]).toarray()

    prediction = model.predict(text_vector)

    probability = float(prediction[0][0])

    if probability > 0.5:
        result = "fake News"
        value = 1
    else:
        result = "real News"
        value = 0

    return {
        "input": data.text,
        "prediction": result,
        "value": value,
        "probability": probability
    }