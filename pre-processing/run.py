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
# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()



def go(args):

    # Load config.json and get input and output paths
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # Setting-up directory paths
    input_folder_path = config['directories']['input_folder_path']
    output_folder_path = config['directories']['output_folder_path']
    # Setting-up ingested file recording
    file_record = open(
        f"../reports/ingestedfiles/{datetime.now().strftime('%Y-%m-%d')}.txt", "w")


    # Creating instance
    run = wandb.init(
        job_type="pre-processing")
    run.config.update(args)

    logger.info("2 - Running pre-processing step")

    logger.info("Merging multiple dataframes")
    data = pd.DataFrame()
    data_dir = os.path.abspath('..')
    datasets = os.listdir(data_dir+'/'+input_folder_path)
    # Iterating through each dataset
    for each_dataset in datasets:
        file_record.write(str(each_dataset)+'\n')
        df = pd.read_csv(data_dir+'/'+input_folder_path +
                         '/'+each_dataset, header=None)
        # Concatinating all datasets into one
        # data = pd.concat([data, df], axis=0)
        data = merge_datasets(data, df)
    # Removing dublicates
    result = drop_duplicates(data)
    # Saving merged datasets as one file
    result.to_csv(f'../{output_folder_path}/raw_data.csv', index=None)

    logger.info("Creating dataframe")
    df = pd.read_csv(f'../{output_folder_path}/raw_data.csv', header=None)

    logger.info("Adding headers")
    df.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    # Removing unnecessary row
    df = df.drop(0, axis=0)

    logger.info("Removeing rows with missing values")
    df = remove_na(df)

    logger.info("Converting Date column into datetime format")
    df = convert_date_column(df)

    logger.info("Sorting dataframe by date")
    df = sort_dataframe(df)

    # Converting dates to  timestamps
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

    logger.info("Encoding unique strings")
    encoder = LabelEncoder()
    encoder.fit(df['Home'])
    df = encode_team_names(df, encoder)

    logger.info('Saving encoder locally')
    encoder_file = 'encoder.pkl'
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)

    logger.info('Saving encoder to wandb')
    encoder_artifact = wandb.Artifact(
        'encoder',
        type='encoder'
    )
    encoder_artifact.add_file('encoder.pkl')
    run.log_artifact(encoder_artifact)

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


def merge_datasets(df1, df2):
    data = pd.concat([df1, df2], axis=0)
    return data


def drop_duplicates(df):
    result = df.drop_duplicates()
    return result


def remove_na(df):
    df = df.dropna()
    return df


def convert_date_column(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d, %H:%M')
    return df


def sort_dataframe(df):
    df = df.sort_values(by='Date')
    return df


def encode_team_names(df, encoder):
    df['Home'] = encoder.transform(df['Home'])
    df['Away'] = encoder.transform(df['Away'])
    return df


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
