"""
Merges all available data. Performs data cleaning and saves it in W&B
"""

import os
import yaml
from datetime import datetime
import argparse
import logging
import pickle
import wandb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()

# Load config.json and get input and output paths
with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# Setting-up the wandb experiment
input_folder_path = config['directories']['input_folder_path']
output_folder_path = config['directories']['output_folder_path']
file_record = open(
    f"../reports/ingestedfiles/{datetime.now().strftime('%Y-%m-%d')}.txt", "w")


def go(args):

    # Creating instance
    run = wandb.init(
        job_type="pre-processing")
    run.config.update(args)

    logger.info("2 - Running pre-processing step")

    logger.info("Merging multiple dataframes")
    data = pd.DataFrame()
    data_dir = os.path.abspath('..')
    datasets = os.listdir(data_dir+'/'+input_folder_path)
    for each_dataset in datasets:
        file_record.write(str(each_dataset)+'\n')
        df = pd.read_csv(data_dir+'/'+input_folder_path +
                         '/'+each_dataset, header=None)
        data = pd.concat([data, df], axis=0)
    result = data.drop_duplicates()
    result.to_csv(f'../{output_folder_path}/raw_data.csv', index=None)

    logger.info("Creating dataframe")
    df = pd.read_csv(f'../{output_folder_path}/raw_data.csv', header=None)

    logger.info("Adding headers")
    df.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    # Removing unnecessary row
    df = df.drop(0, axis=0)

    logger.info("Removeing rows with missing values")
    df = df.dropna()

    logger.info("Converting Date column into datetime format")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d, %H:%M')

    logger.info("Sorting dataframe by date")
    df = df.sort_values(by='Date')

    # df['index'] = df['Date'].copy()

    df['Date'] = df['Date'].astype(int) / 10**18

    logger.info("Setting Date column as index")
    # df = df.set_index('index')

    logger.info(
        "Converting Results columns into separate columns for Home and Away goals")
    score_strings = df['Result']
    homeResult = []
    awayResult = []

    for score_string in score_strings:
        scores = score_string.split(' : ')
        home = int(scores[0])
        away = int(scores[1])
        homeResult.append(home)
        awayResult.append(away)

    logger.info("Encoding unique strings")
    encoder = LabelEncoder()
    encoder.fit(df['Home'])
    df['Home'] = encoder.transform(df['Home'])
    df['Away'] = encoder.transform(df['Away'])

    encoder_file = 'encoder.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)

    encoder_artifact = wandb.Artifact(
        'encoder',
        type='encoder'
    )
    encoder_artifact.add_file('encoder.pkl')
    run.log_artifact(encoder_artifact)

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

    logger.info("Dropping unnecessary columns")
    df = df.drop(['Blank', 'Location', 'Result'], axis=1)

    logger.info("Saving dataframe as a csv file")
    df.to_csv(f'../{output_folder_path}/processed_data.csv', index=None)

    logger.info("Uploading processed_data.csv file to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(f'../{output_folder_path}/processed_data.csv')
    run.log_artifact(artifact)

    logger.info("Finished pre-processing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This step merges and cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Fully-qualified name for the input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Name of the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Description of the output artifact',
        required=True
    )

    args = parser.parse_args()

    go(args)
