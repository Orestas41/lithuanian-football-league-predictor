"""
The script defines a FastAPI application with two endpoints,
where the "/predict" endpoint performs predictions using a trained machine learning model
based on the input data received in a POST request.
"""
# pylint: disable=E0401, R0903, C0103
import os
from datetime import datetime
import pickle
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
import mlflow

NOW = pd.Timestamp.now()
DATE = NOW.timestamp() / 10**18


class Predict(BaseModel):
    """
    Structure and validation rules for the input data used in the prediction endpoint of an API.
    """
    Date: str = Query(default=datetime.now().strftime('%Y-%m-%d, %H:%M'))
    Home: str
    Away: str

    class Config:
        """
        Provide additional configuration options and metadata
        """
        schema_extra = {
            'example': {
                'Home': 'Panevėžys',
                'Away': 'Palanga'
            }
        }


app = FastAPI()


@app.get('/')
async def say_hello():
    """
    Return a greeting message when the root URL is accessed.
    """
    return {'greeting': 'Hello World!'}


@app.post("/predict")
async def model_inference(data: Predict):
    """
    Perform inference using a trained model on the input data and generate a prediction result.
    """

    dirname = os.path.dirname(__file__)
    model = mlflow.sklearn.load_model(os.path.join(
        dirname, "prod_model_dir"))
    with open('pre-processing/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    sample = pd.DataFrame(data).transpose()
    sample.columns = sample.iloc[0]
    sample = sample.drop(0, axis=0)

    sample['Date'] = (pd.to_datetime(
        sample['Date'], format="%Y-%m-%d, %H:%M")).astype(int) / 10**18

    sample['Home'] = encoder.transform(sample['Home'])
    sample['Away'] = encoder.transform(sample['Away'])

    pred = model.predict(sample)

    if pred[0] > 0.5:
        pred[0] = int(pred[0] * 100)
        winner = data.Away
        loser = data.Home
    else:
        pred[0] = int((100 - pred[0] * 100))
        winner = data.Home
        loser = data.Away

    return {f"{winner} has a chance of {pred[0]}% to win against {loser}"}
