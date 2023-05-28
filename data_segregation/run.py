"""
This script splits the provided dataframe in test and remainder
"""
import os
import argparse
from datetime import datetime
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()


def go(args):

    run = wandb.init(
        project='project-FootballPredict',
        group='development',
        job_type="data_segregation")
    run.config.update(args)

    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
    )

    # Saving to output files
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            df.to_csv(fp.name, index=False)

            artifact = wandb.Artifact(
                f"{k}_data.csv",
                type=f"{k}_data",
                description=f"{k} split of dataset",
            )
            artifact.add_file(fp.name)
            run.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    args = parser.parse_args()

    go(args)
