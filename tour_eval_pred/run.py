"""
Evaluating previous tour results against models predictions and batch predicting next tour match results
"""

import csv
import os.path
import logging
import wandb
import mlflow
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        job_type="data_scraping")
    run.config.update(args)
    logger.info("7 - Running tour evaluation and prediction step")

    logger.info("Configuring webdriver")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")

    homedir = os.path.expanduser("~")
    webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

    logger.info("Setting browser")
    driver = webdriver.Chrome(
        service=webdriver_service, options=chrome_options)

    result_website = "https://alyga.lt/rezultatai/1"
    logger.info(
        f"Opening website for of the latest results - {result_website}")
    driver.get(result_website)

    logger.info("Scraping the data")
    rows = driver.find_elements(By.TAG_NAME, "tr")

    # Saving the data to results csv file
    with open(f"../reports/tours/result.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the data rows
        for row in rows[1:6]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    logger.info("Inserting the data to dataframe")
    df = pd.read_csv(
        f"../reports/tours/result.csv", header=None)

    logger.info("Applying pre-processing")
    df.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    # Converting dates to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d, %H:%M')

    # Converting dates to timestamps
    df['Date'] = df['Date'].astype(int) / 10**18

    # Converting Results columns into separate columns for Home and Away goals
    score_strings = df['Result']
    homeResult = []
    awayResult = []

    for score_string in score_strings:
        scores = score_string.split(' : ')
        home = int(scores[0])
        away = int(scores[1])
        homeResult.append(home)
        awayResult.append(away)

    logger.info("Creating Winner column with the team that won or draw")
    Winner = [0] * len(df)
    for i in range(len(df)):
        if homeResult[i] > awayResult[i]:
            Winner[i] = 0
        elif homeResult[i] < awayResult[i]:
            Winner[i] = 1
        else:
            Winner[i] = 0.5

    df['Winner'] = Winner

    # Removing uneccessary columns
    df = df.drop(['Blank', 'Location', 'Result'], axis=1)

    logger.info("Adding previous predictions for this tour")
    prev_preds = pd.read_csv(
        f"../reports/tours/predictions.csv", header=None)

    prev_preds = prev_preds.values.tolist()

    df['Prediction'] = prev_preds[0]

    logger.info("Calculating prediction error")
    df['Model Performance'] = abs(df['Winner'] - df['Prediction'])

    logger.info("Saving the report on the latest tour prediction evaluations")
    df.to_csv(
        f"../reports/tours/{datetime.now().strftime('%Y-%m-%d')}.csv", index=None)

    fixture_website = "https://alyga.lt/tvarkarastis/1"
    logger.info(
        f"Opening website of the future tour fixtures - {fixture_website}")
    driver.get(fixture_website)

    logger.info("Scraping the data")
    rows = driver.find_elements(By.TAG_NAME, "tr")

    # Saving the data to next_tour csv file
    with open(f"../reports/tours/next_tour.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the data rows
        for row in rows[1:6]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    logger.info("Inserting the data into a dataframe")
    df1 = pd.read_csv(
        f"../reports/tours/next_tour.csv", header=None)

    logger.info("Applying pre-processing")
    df1.columns = ["Date", "Blank", "Home", "TV", "Away", "Location"]

    # Converting dates to datetime format
    df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d, %H:%M')

    # Converting dates to timestamps
    df1['Date'] = df1['Date'].astype(int) / 10**18

    logger.info("Loading the encoder")
    with open('../pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    home = df1['Home']
    away = df1['Away']

    logger.info("Encoding home and away team names")
    df1['Home'] = encoder.transform(df1['Home'])
    df1['Away'] = encoder.transform(df1['Away'])

    # Removing unecessary
    df1 = df1.drop(['Blank', 'Location', 'TV'], axis=1)

    logger.info(f"Downloading the production model-{args.mlflow_model}")
    model_local_path = run.use_artifact(args.mlflow_model).download()

    logger.info(
        "Loading the model and predicting the results for next tour fixtures")
    model = mlflow.sklearn.load_model(model_local_path)
    pred = model.predict(df1)

    preds = [pred]
    logger.info("Saving the predictions to csv file")
    with open("../reports/tours/predictions.csv", "w") as f:
        writer = csv.writer(f)

        for pred in preds:
            writer.writerow(pred)

    logger.info("Writing models predictions to a presentable txt file")
    with open("../reports/tours/predictions_for_next_tour.txt", "w") as f:
        for i in range(len(pred)):
            f.write(
                f"For the match between {home[i]} and {away[i]}, model predicts that {home[i]}'s chance of winning is {(pred[i]*100).astype(int)}%.\n")

    logger.info("Removing temporary files")
    os.remove("../reports/tours/next_tour.csv")
    os.remove("../reports/tours/result.csv")

    logger.info("Batch tour evaluations and predictions finished")
    driver.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument("--step_description", type=str,
                        help="Description of the step")

    args = parser.parse_args()

    go(args)
