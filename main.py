import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "pre-processing",
    "data_check",
    "data_segregation"
]


# Reading the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setting-up the wandb experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Moving to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "pre-processing" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "pre-processing"),
                "main",
                parameters={
                    "input_artifact": "raw_data.csv:latest",
                    "output_artifact": "processed_data.csv",
                    "output_type": "processed_data",
                    "output_description": "Merged and cleaned data", },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_check"),
                "main",
                parameters={
                    "csv": "processed_data.csv:latest",
                    "ref": "processed_data.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"]}
            )

        if "data_segregation" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_segregation"),
                "main",
                parameters={
                    "input": "processed_data.csv:latest",
                    "test_size": config["data_segregation"]["test_size"],
                    "random_seed": config["data_segregation"]["random_seed"]}
            )


if __name__ == "__main__":
    go()
