import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "pre-processing"
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
                    "input_artifact": "trainingdata.csv:latest",
                    "output_artifact": "trainingdata.csv",
                    "output_type": "trainingdata",
                    "output_description": "Merged and cleaned data",},
            )

if __name__ == "__main__":
    go()