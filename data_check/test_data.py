"""
This script runs data tests
"""
# pylint: disable=E0401, E1101
import logging
from datetime import datetime
import pandas as pd
import wandb
import scipy

# Setting up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.ERROR)
LOGGER = logging.getLogger()

RUN = wandb.init(
    job_type="data_check")

LOGGER.info("3 - Running data checks")


def test_column_names(data):
    """
    Testing columns are what is expected
    """
    LOGGER.info("Testing if the column names are correct")
    expected_colums = [
        "Date",
        "Home",
        "Away",
        "Winner"
    ]
    these_columns = data.columns.values
    assert list(expected_colums) == list(these_columns)


def test_format(data):
    """
    Test the format of values is correct
    """
    LOGGER.info("Testing if the format of the values are correct")
    # Convert the index of the DataFrame to a datetime
    data.index = pd.to_datetime(data.index)

    # Check if the index is in correct format
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.dtype == 'datetime64[ns]'
    # Checking if columns that are not dates have either integer or float
    # values
    for column in data.columns:
        if column != 'Date':
            assert data[column].dtype in (int, float)


def test_number_of_teams(data):
    """
    Test if number of unique home teams is same as away teams
    """
    LOGGER.info("Testing if number of teams are correct")
    assert data['Home'].nunique() == data['Away'].nunique()


def test_winner_range(data):
    """
    Test the range of winner values
    """
    LOGGER.info("Testing if the values of Winner column are correct")
    assert data['Winner'].nunique() == 3
    # Checking that winner values are between 0 and 1
    assert data['Winner'].min() >= 0 and data['Winner'].max() <= 1


def test_similar_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        threshold: float):
    """
    Applying a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """

    LOGGER.info(
        "Testing of the distribution of the dataset is similar to what is expected")
    dist1 = data['Winner'].value_counts().sort_index()
    dist2 = ref_data['Winner'].value_counts().sort_index()
    # Checking if the distirbution difference is less than the k1 threshold
    assert scipy.stats.entropy(dist1, dist2, base=2) < threshold

    LOGGER.info("Finished data checks")
