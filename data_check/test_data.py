import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np
import scipy.stats

log_folder = os.getcwd()

logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log", level=logging.DEBUG)
logger = logging.getLogger()

logger.info("Testing if the column names are correct")


def test_column_names(data):
    """
    Test columns
    """

    expected_colums = [
        "Date",
        "Home",
        "Away",
        "homeResult",
        "awayResult",
        "Winner"
    ]

    these_columns = data.columns.values

    assert list(expected_colums) == list(these_columns)


logger.info("Testing if the format of the values are correct")


def test_format(data):
    """
    Test the format of values is correct
    """

    # Convert the index of the DataFrame to a datetime
    data.index = pd.to_datetime(data.index)

    # Check if the index is in correct format
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.dtype == 'datetime64[ns]'

    # assert isinstance(data.index, pd.DatetimeIndex)

    for column in data.columns:
        if column != 'Date':
            assert data[column].dtype == int


logger.info("Testing if number of teams are correct")


def test_number_of_teams(data):
    """
    Test if number of unique home teams is same as away
    """

    assert data['Home'].nunique() == data['Away'].nunique()


logger.info("Testing the results are withing possible limits")


def test_result_range(data):
    """
    Test the range of results
    """
    results = ['homeResult', 'awayResult']

    for result in results:
        assert 0 <= data[result].any() < 15


logger.info("Testing if the values of Winner column are correct")


def test_winner_range(data):
    """
    Test the range of winner values
    """

    assert data['Winner'].nunique() == 3


logger.info("Testing if the winner column is set up correctly")


def test_winner(data):
    """
    Test the winner column is correct
    """

    for i in range(len(data)):
        if data['homeResult'][i] > data['awayResult'][i]:
            assert data['Winner'][i] == 1
        elif data['homeResult'][i] < data['awayResult'][i]:
            assert data['Winner'][i] == 3
        else:
            assert data['Winner'][i] == 2


logger.info(
    "Testing of the distribution of the dataset is similar to what is expected")


def test_similar_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['Winner'].value_counts().sort_index()
    dist2 = ref_data['Winner'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold
