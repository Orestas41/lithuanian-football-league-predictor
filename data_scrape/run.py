
import csv
import os.path
import logging
import wandb
import argparse
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
    logger.info("1 - Running data scrape step")
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

    with open(f"../raw_data/{datetime.now().strftime('%Y-%m-%d')}.csv", 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the data rows
        for row in rows[1:]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    logger.info("Scraping finished")
    driver.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    parser.add_argument("--step_description", type=str,
                        help="Description of the step")

    args = parser.parse_args()

    go(args)
