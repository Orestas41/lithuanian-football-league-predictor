import pytest
import os
import pandas as pd
from datetime import datetime
import logging
import wandb

log_folder = os.getcwd()
# Setting up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store", type=float)


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")
    return kl_threshold
