"""
Merges all available data. Performs data cleaning and saves it in W&B
"""

import os
import yaml
from datetime import datetime
import argparse
import logging
import wandb
import pandas as pd

log_folder = os.getcwd()

#logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logging.basicConfig(filename=f"../reports/logs/{log_folder.split('/')[-1]}-{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('../config.yaml','r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# Setting-up the wandb experiment
input_folder_path = config['directories']['input_folder_path']
output_folder_path = config['directories']['output_folder_path']
file_record=open(f"../reports/logs/ingestedfiles-{datetime.now().strftime('%Y-%m-%d')}.txt","w")

def go(args):

    # Creating instance
    run = wandb.init(
        project='project-FootballPredict',
        group='dev',
        job_type="pre-processing")
    run.config.update(args)

    logger.info("Merging multiple dataframes")
    data = pd.DataFrame()
    data_dir = os.path.abspath('..')
    datasets = os.listdir(data_dir+'/'+input_folder_path)
    for each_dataset in datasets:
        file_record.write(str(each_dataset)+'\n')
        df = pd.read_csv(data_dir+'/'+input_folder_path+'/'+each_dataset)
        data = data.append(df)
    result=data.drop_duplicates()
    result.to_csv(f'../{output_folder_path}/trainingdata.csv', index=True)    

    logger.info("Creating dataframe")
    df = pd.read_csv(f'../{output_folder_path}/trainingdata.csv')

    logger.info("Removeing rows with missing values")
    df = df.dropna()

    logger.info("Converting Date column into datetime format")
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d, %H:%M")

    logger.info("Sorting dataframe by date")
    df = df.sort_values(by='Date')     

    logger.info("Setting Date column as index")
    df = df.set_index('Date')

    logger.info("Checking if teams are correct. Changing if incorrect")
    for i in range(len(df)):
        if df['Home'][i] != df['Away'][i]:
            pass
        else:
            df['Home'][i] = df['Home-missing'][i]
            df['Away'][i] = df['Away-missing'][i]

    logger.info("Checking the teams are on the correct side. Changing if incorrect")
    for i in range(len(df)):
        if df['Home'][i] != df['Home-missing'][i] and df['Away'][i] != df['Away-missing'][i]:
            df['Home'][i] = df['Home-missing'][i]
            df['Away'][i] = df['Away-missing'][i]

    logger.info("Converting Results columns into separate columns for Home and Away goals")
    score_strings = df['Result']
    homeResult = []
    awayResult = []

    for score_string in score_strings:
        scores = score_string.split(' : ')
        home = int(scores[0])
        away = int(scores[1])
        homeResult.append(home)
        awayResult.append(away)

    df['homeResult'] = homeResult
    df['awayResult'] = awayResult

    logger.info("Encoding unique strings")
    encoder = {}
        
    for i in range(0, df['Home'].nunique()):
        encoder[df['Home'].unique()[i]] = i
        
    encoder['Draw'] = df['Home'].nunique() + 1

    logger.info("Creating Winner column with the team that won or draw")
    Winner = [0] * len(df)
    for i in range(len(df)):
        if df['homeResult'][i] > df['awayResult'][i]:
            Winner[i] = df['Home'][i]
        elif df['homeResult'][i] < df['awayResult'][i]:
            Winner[i] = df['Away'][i]
        else:
            Winner[i] = 'Draw'

    df['Winner'] = Winner

    for i in range(0, df['Home'].nunique()):
        df = df.replace(encoder)

    logger.info("Dropping unnecessary columns")
    df = df.drop(['Position', 'Sanity check', 'Home-missing', 'Away-missing', 'Result', 'Unnamed: 0'], axis=1)    

    logger.info("Saving dataframe as a csv file")
    df.to_csv(f'../{output_folder_path}/trainingdata.csv', index=True)

    logger.info("Uploading trainingdata.csv file to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(f'../{output_folder_path}/trainingdata.csv')
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step merges and cleans the data")

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