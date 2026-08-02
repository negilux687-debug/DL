from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences




model = tf.keras.models.load_model(
    "sentiment_rnn_model.keras"
)




with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)




with open("maxlen.pkl", "rb") as file:
    maxlen = pickle.load(file)





app = FastAPI(title="Sentiment Analysis API",)





class TextInput(BaseModel):
    text: str





@app.get("/")
def home():
    return {"message": "Sentiment Analysis API Running" }



@app.post("/predict")
def predict(data: TextInput):

    text = data.text




    sequence = tokenizer.texts_to_sequences(
        [text]
    )



    padded = pad_sequences(
        sequence,
        maxlen=maxlen,
        padding="post"
    )


 

    prediction = model.predict(padded)[0][0]


    if prediction > 0.5:
        result = "Positive"
    else:
        result = "Negative"


    return {
        "text": text,
        "sentiment": result,
        "confidence": float(prediction)
    }
