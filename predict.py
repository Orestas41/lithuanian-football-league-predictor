from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List

import os
import joblib
from datetime import datetime
import wandb
import xgboost as xgb
import mlflow
import pickle
import pandas as pd
import numpy as np


now = pd.Timestamp.now()
date = now.timestamp() / 10**18


class Predict(BaseModel):
    date: str
    home: str
    away: str

    class Config:
        schema_extra = {
            'example': {
                'date': datetime.now().strftime('%Y-%m-%d, %H:%M'),
                'home': 'Panevėžys',
                'away': 'Šiauliai'
            }
        }


app = FastAPI()


@app.get('/')
async def say_hello():
    return {'greeting': 'Hello World!'}


@app.post("/predict")
async def model_inference(data: Predict):

    dirname = os.path.dirname(__file__)
    xgboost = mlflow.xgboost.load_model(os.path.join(
        dirname, "training_validation/xgboost_dir/model.xgb"))
    with open('pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    sample = pd.DataFrame(data)
    sample = sample.T
    sample.columns = sample.iloc[0]
    sample = sample.drop(0, axis=0)

    sample['date'] = (pd.to_datetime(
        sample['date'], format="%Y-%m-%d, %H:%M")).astype(int) / 10**18

    sample['home'] = encoder.transform(sample['home'])
    sample['away'] = encoder.transform(sample['away'])

    sample['homeResult'] = np.nan
    sample['awayResult'] = np.nan

    X = sample

    pred = xgboost.predict(X)
    print(pred)
    pred = [round(result) for result in pred]

    if pred[0] == 1:
        pred = sample['home'].iat[0]
        pred = encoder.inverse_transform([pred])
    elif pred[0] == 3:
        pred = sample['away'].iat[0]
        pred = encoder.inverse_transform([pred])
    else:
        pred = 'Draw'

    return {"Winner": pred[0]}
