from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Union, List

import os
import joblib
from datetime import datetime
import wandb
import mlflow
import pickle
import pandas as pd
import numpy as np


now = pd.Timestamp.now()
date = now.timestamp() / 10**18


class Predict(BaseModel):
    Date: str = Query(default=datetime.now().strftime('%Y-%m-%d, %H:%M'))
    Home: str
    Away: str

    class Config:
        schema_extra = {
            'example': {
                'Home': 'Panevėžys',
                'Away': 'Palanga'
            }
        }


app = FastAPI()


@app.get('/')
async def say_hello():
    return {'greeting': 'Hello World!'}


@app.post("/predict")
async def model_inference(data: Predict):

    dirname = os.path.dirname(__file__)
    model = mlflow.sklearn.load_model(os.path.join(
        dirname, "training_validation/model_dir"))
    with open('pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    sample = pd.DataFrame(data)
    sample = sample.T

    sample.columns = sample.iloc[0]
    sample = sample.drop(0, axis=0)

    home = sample['Home'].iat[0]
    away = sample['Away'].iat[0]

    sample['Date'] = (pd.to_datetime(
        sample['Date'], format="%Y-%m-%d, %H:%M")).astype(int) / 10**18

    sample['Home'] = encoder.transform(sample['Home'])
    sample['Away'] = encoder.transform(sample['Away'])

    pred = model.predict(sample)

    if pred[0] > 0.5:
        pred[0] = (pred[0]*100).astype(int)
        winner = away
        loser = home
    else:
        pred[0] = (100 - pred[0]*100).astype(int)
        winner = home
        loser = away

    return {f"{winner} has a chance of {pred[0]}% to win against {loser}"}
