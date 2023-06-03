import json

import mlflow
import tempfile
import os

import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "data_scrape",
    "pre-processing",
    "data_check",
    "data_segregation",
    "training_validation",
    "model_test",
    "tour_eval_pred"
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
    with tempfile.TemporaryDirectory():

        if "data_scrape" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_scrape"),
                "main",
                parameters={
                    "step_description": "This step scrapes the latest data from the web"
                },
            )

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
                    "output_description": "Merged and cleaned data"
                },
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
                    "test_size": config["data_segregation"]["test_size"]}
            )

        if "training_validation" in active_steps:

            model_config = os.path.abspath("config.yaml")
            with open(model_config, "w+") as fp:
                json.dump(
                    dict(
                        config["modeling"]["linearRegression"].items()),
                    fp)

            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "training_validation"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "model_config": model_config,
                    "output_artifact": "model_export"},
            )

        if "model_test" in active_steps:

            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "model_test"),
                "main",
                parameters={
                    "mlflow_model": "model_export:latest",
                    "test_dataset": "test_data.csv:latest"},
            )

        if "tour_eval_pred" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "tour_eval_pred"),
                "main",
                parameters={
                    "mlflow_model": "model_export:prod",
                    "step_description": "This step checks if the previous tour predictions were correct and predicts the result of the next tour matches"
                },
            )


if __name__ == "__main__":
    go()
