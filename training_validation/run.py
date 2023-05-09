#!/usr/bin/env python
"""
This script trains and validates the model
"""
import argparse
import logging
import os
import shutil
import joblib
import pickle
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{log_folder.split('/')[-1]}-{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        project='project-FootballPredict',
        group='development',
        job_type="training_validation")
    run.config.update(args)

    # Getting the XGBoost configuration and updating W&B
    with open(args.xgb_config) as fp:
        xgb_config = json.load(fp)
    run.config.update(xgb_config)

    # Fixing the random seed for the XGBoost, so we get reproducible results
    xgb_config['random_state'] = args.random_seed

    # Fetching the training/validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)

    # Removing the column "Winner" from X and putting it into y
    y = X.pop("Winner")

    logger.info(f"Number of outcomes: {y.nunique()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe = get_inference_pipeline(xgb_config)

    # Fitting it to the X_train, y_train data
    logger.info("Fitting")

    # Fitting the inference pipeline
    sk_pipe.fit(X_train, y_train)

    # Computing r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")
    joblib.dump(sk_pipe, "./inference")

    # Uploading inference pipeline artifact to W&B
    logger.info("Saving and exporting the model")
    artifact = wandb.Artifact(
        args.output_artifact,
        type='model_export',
        description='XGBoost pipeline',
        metadata=xgb_config
    )
    artifact.add_dir("./inference/xgboost_dir")
    run.log_artifact(artifact)

    # Saving r_squared under the "r2" key
    run.summary['r2'] = r_squared

    # Logging the variable "mae" under the key "mae".
    run.summary['mae'] = mae


def plot_feature_importance(pipe, feat_names):

    # Collecting the feature importance for all non-nlp features
    feat_imp = pipe["xgboost"].feature_importances_[
        : len(feat_names) - 1]

    # NLP importance
    # Summing up across all the TF-IDF dimensions into a global
    nlp_importance = sum(
        pipe["xgboost"].feature_importances_[
            len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp,
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(xgb_config):

    # Creating xgboost
    xgboost = xgb.XGBRegressor(**xgb_config)

    # Creating inference pipeline
    sk_pipe = Pipeline([('xgboost', xgboost)])

    return sk_pipe


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
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--xgb_config",
        help="XGBoost configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for XGBRegressor.",
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
