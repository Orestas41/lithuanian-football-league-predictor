#!/usr/bin/env python
"""
This step takes the latest model and tests it against the test dataset. If it performs better than previous models, it is promoted to a production.
"""
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import logging
import joblib
import shutil
import mlflow
import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

log_folder = os.getcwd()
# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="model_test")
    run.config.update(args)

    logger.info("6 - Running model testing step")

    logger.info(
        f"Downloading model-{args.mlflow_model} and data{args.test_dataset} artifacts")

    # Downloading input artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Downloading test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Reading test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("Winner")

    logger.info("Loading model and performing inference on test set")
    model = mlflow.sklearn.load_model(model_local_path)
    y_pred = model.predict(X_test)

    logger.info("Scoring")
    r_squared = model.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Running data slice tests")
    # Data slice testing
    # iterate each value and record the metrics
    slice_mae = {}
    for val in y_test.unique():
        # Fix the feature
        idx = y_test == val

        # Do the inference and Compute the metrics
        preds = model.predict(X_test[idx])
        slice_mae[val] = mean_absolute_error(y_test[idx], preds)

    logger.info(f"MAE of slices: {slice_mae}")

    # Setting current date
    date = datetime.now().strftime('%Y-%m-%d')

    logger.info("Testing data drift. Expecting results to be False")
    # Opening model performance log
    perf = pd.read_csv('../reports/model_performance.csv', index_col=0)
    # Raw comprison test
    raw_comp = r_squared < np.min(perf['Score'])
    # Parametric significance test
    param_signific = r_squared < np.mean(
        perf['Score']) - 2*np.std(perf['Score'])
    # Non-parametric outlier test
    iqr = np.quantile(perf['Score'], 0.75) - np.quantile(perf['Score'], 0.25)
    nonparam = r_squared < np.quantile(perf['Score'], 0.25) - iqr*1.5

    logger.info(f"Raw comparison: {raw_comp}")
    logger.info(f"Parametric significance: {param_signific}")
    logger.info(f"Non-parametric outlier: {nonparam}")

    logger.info(
        "Saving the latest model performance metrics")
    date = datetime.now().strftime('%Y-%m-%d')
    with open('../reports/model_performance.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([date, r_squared, mae])

    logger.info(
        "Checking if the new model is perfroming better than previous models")
    # If the MAE score of the latest model is smaller (better performace) than any other models MAE, then this model is promoted to production model
    if mae <= perf['MAE'].min() or raw_comp == True or param_signific == True or nonparam == True:
        if os.path.exists("../prod_model_dir"):
            shutil.rmtree("prod_model_dir")
        logger.info("Saving model locally")
        mlflow.sklearn.save_model(model, "../prod_model_dir")
        artifact = wandb.Artifact(
            args.mlflow_model, type='wandb.Artifact', name='model_export')
        logger.info("Promoting model to production in wandb")
        artifact.add_alias("prod")
        artifact.save()
    else:
        pass

    performance = pd.read_csv("../reports/model_performance.csv")
    logger.info(
        "Generating plot that displays the change in model performance over time")
    plt.plot(performance["Date"], performance["Score"], label="Score")
    plt.plot(performance["Date"], performance["MAE"], label="MAE")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Score/MAE")
    plt.title("Change in ML Model Performance")

    # Save the plot
    plt.savefig("../reports/model_performance.png")

    # Logging MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    run.summary["Raw comparison"] = raw_comp
    run.summary["Parametric significance"] = param_signific
    run.summary["Non-parametric outlier"] = nonparam

    logger.info("Finished testing the model")

    # Finish the run
    run.finish()


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
