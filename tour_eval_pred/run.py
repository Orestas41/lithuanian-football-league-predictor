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

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        job_type="data_scraping")
    run.config.update(args)
    logger.info("7 - Running tour evaluation and prediction step")
    # Setup chrome options
    logger.info("Configuring webdriver")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")

    homedir = os.path.expanduser("~")
    webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

    # Choose Chrome Browser
    logger.info("Setting browser")
    driver = webdriver.Chrome(
        service=webdriver_service, options=chrome_options)

    logger.info("Opening website")
    driver.get("https://alyga.lt/rezultatai/1")

    logger.info("Scraping the data")
    rows = driver.find_elements(By.TAG_NAME, "tr")

    with open(f"../reports/tours/result.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the data rows
        for row in rows[1:6]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    df = pd.read_csv(
        f"../reports/tours/result.csv", header=None)

    df.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d, %H:%M')

    df['index'] = df['Date'].copy()

    df['Date'] = df['Date'].astype(int) / 10**18

    df = df.set_index('index')

    score_strings = df['Result']
    homeResult = []
    awayResult = []

    for score_string in score_strings:
        scores = score_string.split(' : ')
        home = int(scores[0])
        away = int(scores[1])
        homeResult.append(home)
        awayResult.append(away)

    with open('../pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    df['Home'] = encoder.transform(df['Home'])
    df['Away'] = encoder.transform(df['Away'])

    Winner = [0] * len(df)
    for i in range(len(df)):
        if homeResult[i] > awayResult[i]:
            Winner[i] = 0
        elif homeResult[i] < awayResult[i]:
            Winner[i] = 1
        else:
            Winner[i] = 0.5

    df['Winner'] = Winner

    logger.info("Dropping unnecessary columns")
    df = df.drop(['Blank', 'Location', 'Result'], axis=1)

    df['Prediction'] = np.nan

    df['Model Performance'] = abs(df['Winner'] - df['Prediction'])

    driver.get("https://alyga.lt/tvarkarastis/1")

    logger.info("Scraping the data")
    rows = driver.find_elements(By.TAG_NAME, "tr")

    with open(f"../reports/tours/next_tour.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the data rows
        for row in rows[1:6]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    df1 = pd.read_csv(
        f"../reports/tours/next_tour.csv", header=None)

    df1.columns = ["Date", "Blank", "Home", "TV", "Away", "Location"]

    df1['Date'] = pd.to_datetime(
        df1['Date'], format='%Y-%m-%d, %H:%M')

    df['index'] = df['Date'].copy()

    df1['Date'] = df1['Date'].astype(int) / 10**18

    df = df.set_index('index')

    with open('../pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    df1['Home'] = encoder.transform(df1['Home'])
    df1['Away'] = encoder.transform(df1['Away'])

    df1 = df1.drop(['Blank', 'Location', 'TV'], axis=1)

    model_local_path = run.use_artifact(args.mlflow_model).download()

    model = mlflow.sklearn.load_model(model_local_path)
    pred = model.predict(df1)

    df1['Prediction'] = pred

    df['Home'] = encoder.inverse_transform(df['Home'])
    df['Away'] = encoder.inverse_transform(df['Away'])
    df1['Home'] = encoder.inverse_transform(df1['Home'])
    df1['Away'] = encoder.inverse_transform(df1['Away'])

    all_old = pd.read_csv(
        f"../reports/tours/tour_eval_pred.csv")

    all = pd.merge(all_old, df, on='Date', how='inner')

    all = pd.concat([all, df1], axis=0)

    all.to_csv(
        f"../reports/tours/tour_eval_pred.csv", index=None)

    print(all)

    os.remove("../reports/tours/result.csv")
    os.remove("../reports/tours/next_tour.csv")

    logger.info("Scraping finished")
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
