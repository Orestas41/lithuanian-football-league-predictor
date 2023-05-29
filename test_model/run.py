#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import os
import csv
from datetime import datetime
import argparse
import logging
import joblib
import mlflow
import wandb
import numpy as np
# import mlflow
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        project='project-FootballPredict',
        group='development',
        job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")

    # Downloading input artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Downloading test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Reading test dataset
    """    test_data = pd.read_csv(test_dataset_path)
    X_test = test_data[['homeResult', 'awayResult']] = ''
    y_test = test_data.pop("Winner")"""

    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("Winner")

    logger.info("Loading model and performing inference on test set")
    xgboost = mlflow.xgboost.load_model(model_local_path)
    y_pred = xgboost.predict(X_test)
    y_pred = [round(result) for result in xgboost.predict(X_test)]

    logger.info("Scoring")
    r_squared = xgboost.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    # Data slice testing
    # iterate each value and record the metrics
    slice_mae = {}
    for val in y_test.unique():
        # Fix the feature
        idx = y_test == val

        # Do the inference and Compute the metrics
        preds = [round(result) for result in xgboost.predict(X_test[idx])]
        slice_mae[val] = mean_absolute_error(y_test[idx], preds)

    date = datetime.now().strftime('%Y-%m-%d')
    with open('../reports/model_performance.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([date, r_squared, mae])

    perf = pd.read_csv('../reports/model_performance.csv', index_col=0)
    raw_comp = r_squared < np.min(perf['Score'])
    param_signific = r_squared < np.mean(
        perf['Score']) - 2*np.std(perf['Score'])
    iqr = np.quantile(perf['Score'], 0.75) - np.quantile(perf['Score'], 0.25)
    nonparam = r_squared < np.quantile(perf['Score'], 0.25) - iqr*1.5

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")
    logger.info(f"MAE of slices: {slice_mae}")
    logger.info(f"Raw comparison: {raw_comp}")
    logger.info(f"Parametric significance: {param_signific}")
    logger.info(f"Non-parametric outlier: {nonparam}")

    # Logging MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    run.summary["Raw comparison"] = raw_comp
    run.summary["Parametric significance"] = param_signific
    run.summary["Non-parametric outlier"] = nonparam


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
