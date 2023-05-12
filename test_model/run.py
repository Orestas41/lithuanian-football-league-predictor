#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import os
from datetime import datetime
import argparse
import logging
import joblib
import wandb
# import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

# from wandb_utils.log_artifact import log_artifact


log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{log_folder.split('/')[-1]}-{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        project='project-FootballPredict',
        group='development',
        job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")

    # Downloading input artifact
    # model_local_path = run.use_artifact(args.mlflow_model).download()
    model_local_path = '../inference/trainedmodel'

    # Downloading test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Reading test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("Winner")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = joblib.load(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    # Logging MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
