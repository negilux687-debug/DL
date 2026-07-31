from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = FastAPI()



model = tf.keras.models.load_model(
    "sentiment_model.keras",
    compile=False
)



with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)



with open("maxlen.pkl", "rb") as f:
    maxlen = pickle.load(f)



class TextInput(BaseModel):
    text: str



@app.get("/")
def home():
    return {"message": "Sentiment API is running"}



@app.post("/predict")
def predict(data: TextInput):

    seq = tok.texts_to_sequences([data.text])

    padded = pad_sequences(
        seq,
        maxlen=maxlen,
        padding="post"
    )

    pred = model.predict(padded)

    score = pred[0][0]

    if score > 0.5:
        result = "negative"
    else:
        result = "positive"


    return {
        "text": data.text,
        "sentiment": result,
        "confidence": float(score)
    }







