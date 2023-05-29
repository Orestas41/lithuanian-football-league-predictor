from fastapi import FastAPI
from pydantic import BaseModel

import os
import joblib
import wandb
import mlflow
import pandas as pd
import numpy as np

app = FastAPI()


@app.get('/')
async def say_hello():
    return {'greeting': 'Hello World!'}


@app.post("/predict")
async def model_inference(text: str):

    run = wandb.init(
        project='project-FootballPredict',
        group='production',
        job_type="prediction")
    run.config.update(args)

    model_local_path = run.use_artifact(args.mlflow_model).download()

    xgboost = mlflow.xgboost.load_model(model_local_path)

    pred = xgboost.predict(X)

    return {"prediction": pred}
