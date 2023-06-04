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


def merge_datasets(df1, df2):
    """
    Merges two DataFrames.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.

    Returns:
        The merged DataFrame.
    """
    return pd.concat([df1, df2], axis=0)


def drop_duplicates(df):
    """
    Drops duplicate rows from a DataFrame.

    Args:
        df: The DataFrame.

    Returns:
        The DataFrame with duplicate rows dropped.
    """
    return df.drop_duplicates()


def remove_na(df):
    """
    Removes rows with missing values from a DataFrame.

    Args:
        df: The DataFrame.

    Returns:
        The DataFrame with missing values removed.
    """
    return df.dropna()


def convert_date_column(df):
    """
    Convert a date column in a DataFrame to a datetime object.

    Args:
        df: The DataFrame.
        format: The format of the date column.

    Returns:
        The DataFrame with the date column converted to a datetime object.
    """
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d, %H:%M')
    return df


def sort_dataframe(df, col):
    """
    Sorts a DataFrame by a column.

    Args:
        df: The DataFrame.
        col: The column to sort by.

    Returns:
        The sorted DataFrame.
    """
    return df.sort_values(by=col)


def encode_team_names(df, encoder):
    """
    Encodes the team names in a DataFrame using a LabelEncoder.

    Args:
        df: The DataFrame.
        encoder: A LabelEncoder object.

    Returns:
        The DataFrame with the team names encoded.
    """
    df['Home'] = encoder.transform(df['Home'])
    df['Away'] = encoder.transform(df['Away'])
    return df


def go(args):

    # Load config.yaml and get input and output paths
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
    # Get the datasets
    raw_data = pd.DataFrame()
    raw_datasets = []
    for dataset in os.listdir('../'+input_folder_path):
        raw_datasets.append(pd.read_csv(os.path.join(
            '../'+input_folder_path, dataset), header=None))
        file_record.write(str(dataset)+'\n')

    # Merge the datasets
    for i in range(len(raw_datasets)):
        raw_data = merge_datasets(raw_data, raw_datasets[i])

    # Remove duplicate rows
    raw_data = drop_duplicates(raw_data)

    # Save merged datasets as one file
    raw_data.to_csv(f'../{output_folder_path}/raw_data.csv', index=None)

    logger.info("Adding headers")
    raw_data.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    # Removing unnecessary row
    # df = df.drop(0, axis=0)

    logger.info("Removeing rows with missing values")
    raw_data = remove_na(raw_data)

    logger.info("Converting Date column into datetime format")
    raw_data = convert_date_column(raw_data)

    logger.info("Sorting dataframe by date")
    raw_data = sort_dataframe(raw_data, 'Date')

    # Converting dates to  timestamps
    raw_data['Date'] = raw_data['Date'].astype(int) / 10**18

    # Converting Results columns into separate columns for Home and Away goals
    score_strings = raw_data['Result']
    homeResult = []
    awayResult = []

    for score_string in score_strings:
        scores = score_string.split(' : ')
        home = int(scores[0])
        away = int(scores[1])
        homeResult.append(home)
        awayResult.append(away)

    logger.info("Creating Winner column with the team that won or draw")
    Winner = [0] * len(raw_data)
    for i in range(len(raw_data)):
        if homeResult[i] > awayResult[i]:
            Winner[i] = 0
        elif homeResult[i] < awayResult[i]:
            Winner[i] = 1
        else:
            Winner[i] = 0.5

    raw_data['Winner'] = Winner

    logger.info("Encoding unique strings")
    encoder = LabelEncoder()
    encoder.fit(raw_data['Home'])
    raw_data = encode_team_names(raw_data, encoder)

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
    raw_data = raw_data.drop(['Blank', 'Location', 'Result'], axis=1)

    logger.info("Saving dataframe as a csv file")
    raw_data.to_csv(f'../{output_folder_path}/processed_data.csv', index=None)

    logger.info("Uploading processed_data.csv file to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(f'../{output_folder_path}/processed_data.csv')
    run.log_artifact(artifact)

    logger.info("Successfully pre-processed the data")


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
