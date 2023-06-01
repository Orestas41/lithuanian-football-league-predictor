import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.INFO)
logger = logging.getLogger()

logger.info("3 - Running data checks")


def test_column_names(data):
    """
    Test columns
    """
    logger.error("Testing if the column names are correct")
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
    logger.error("Testing if the format of the values are correct")
    # Convert the index of the DataFrame to a datetime
    data.index = pd.to_datetime(data.index)

    # Check if the index is in correct format
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.dtype == 'datetime64[ns]'

    for column in data.columns:
        if column != 'Date':
            assert data[column].dtype == int or float


def test_number_of_teams(data):
    """
    Test if number of unique home teams is same as away
    """
    logger.error("Testing if number of teams are correct")
    assert data['Home'].nunique() == data['Away'].nunique()


def test_winner_range(data):
    """
    Test the range of winner values
    """
    logger.error("Testing if the values of Winner column are correct")
    assert data['Winner'].nunique() == 3


"""def test_similar_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):

    # Apply a threshold on the KL divergence to detect if the distribution of the new data is
    # significantly different than that of the reference dataset

    logger.errors(
        "Testing of the distribution of the dataset is similar to what is expected")

    dist1 = data['Winner'].value_counts().sort_index()
    dist2 = ref_data['Winner'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold"""

logger.info("Finished data checks")
