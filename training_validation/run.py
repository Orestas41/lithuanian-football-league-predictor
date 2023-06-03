"""
This script trains and validates the model
"""
import argparse
import logging
import os
import shutil

import mlflow
import joblib
import json

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

log_folder = os.getcwd()
# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        job_type="training_validation")
    run.config.update(args)

    logger.info("5 - Running training and validation step")

    # Getting the Linear Regression configuration and updating W&B
    with open(args.model_config) as fp:
        model_config = json.load(fp)
    run.config.update(model_config)

    logger.info(
        f"Fetching {args.trainval_artifact} and setting it as dataframe")
    # Fetching the training/validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)

    logger.info("Setting winner column as target")
    # Removing the column "Winner" from X and putting it into y
    y = X.pop('Winner')

    logger.info(f"Number of outcomes: {y.nunique()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size)

    logger.info("Preparing Linear Regression model")

    model = LinearRegression(**model_config)

    # Fitting it to the X_train, y_train data
    logger.info("Fitting")

    # Fitting the inference pipeline
    model.fit(X_train, y_train)

    # Computing r2 and MAE
    logger.info("Scoring")
    r_squared = model.score(X_val, y_val)

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    if os.path.exists("model_dir"):
        shutil.rmtree("model_dir")

    mlflow.sklearn.save_model(model, "model_dir")

    # Uploading inference pipeline artifact to W&B
    logger.info("Saving and exporting the model")
    artifact = wandb.Artifact(
        args.output_artifact,
        type='model_export',
        description='model pipeline',
        metadata=model_config
    )
    artifact.add_dir("model_dir")
    run.log_artifact(artifact)

    # Saving r_squared under the "r2" key
    run.summary['r2'] = r_squared

    # Logging the variable "mae" under the key "mae".
    run.summary['mae'] = mae

    logger.info("Finished training and valdiation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--model_config",
        help="model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for Linear Regression.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
