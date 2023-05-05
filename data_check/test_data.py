import pandas as pd
import numpy as np
import scipy.stats


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


def test_number_of_teams(data):
    """
    Test if number of unique home teams is same as away
    """

    assert data['Home'].nunique() == data['Away'].nunique()


def test_result_range(data):
    """
    Test the range of results
    """
    results = ['homeResult', 'awayResult']

    for result in results:
        assert 0 <= data[result].any() < 15


def test_winner_range(data):
    """
    Test the range of winner values
    """

    assert data['Winner'].nunique() == data['Home'].nunique() + 1


def test_winner(data):
    """
    Test the winner column is correct
    """

    for i in range(len(data)):
        if data['homeResult'][i] > data['awayResult'][i]:
            assert data['Winner'][i] == data['Home'][i]
        elif data['homeResult'][i] < data['awayResult'][i]:
            assert data['Winner'][i] == data['Away'][i]
        else:
            assert data['Winner'][i] == data['Winner'].max()


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
