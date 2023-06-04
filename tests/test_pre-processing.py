import pytest
import pandas as pd
import numpy as np
import pickle
import sys

sys.path.append(
    "/home/orestas41/project-FootballPredict/lithuanian-football-league-predictor/pre-processing")
    
from run import remove_na
from run import drop_duplicates
from run import merge_datasets
from run import convert_date_column
from run import sort_dataframe
from run import encode_team_names


def test_merge_datasets():
    """Test that the `merge_datasets` function merges the dataframes correctly."""

    # Create two dataframes
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})

    # Merge the dataframes
    merged_df = merge_datasets(df1, df2)

    # Assert that the merged dataframe has the correct data
    assert merged_df.shape == (6, 2)
    assert merged_df['A'].tolist() == [1, 2, 3, 7, 8, 9]
    assert merged_df['B'].tolist() == [4, 5, 6, 10, 11, 12]


def test_drop_duplicates():
    """Test that the `drop_duplicates` function removes duplicate rows correctly."""

    # Create a dataframe with duplicate rows
    df = pd.DataFrame({'A': [1, 1, 2, 3, 3]})

    # Drop the duplicate rows
    unique_df = drop_duplicates(df)

    # Assert that the unique dataframe has no duplicate rows
    assert len(unique_df) == 3


def test_remove_na():
    """Test that the `remove_na` function removes rows with missing values correctly."""

    # Create a dataframe with missing values
    df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5]})

    # Remove the rows with missing values
    non_na_df = remove_na(df)

    # Assert that the non-na dataframe has no missing values
    assert np.all(~pd.isna(non_na_df))


def test_convert_date_column():
    """Test that the `convert_date_column` function converts the date column to the correct format correctly."""

    # Create a dataframe with a date column in the wrong format
    df = pd.DataFrame({'Date': ['2023-03-08, 12:53', '2023-03-09, 13:52']})

    # Convert the date column to the correct format
    converted_df = convert_date_column(df)

    # Assert that the converted date column is in the correct format
    assert converted_df['Date'].dtypes == 'datetime64[ns]'


def test_sort_dataframe():
    """Test that the `sort_dataframe` function sorts the dataframe by the specified column correctly."""

    # Create a dataframe with unsorted data
    df = pd.DataFrame({'Date': [3, 1, 2]})

    # Sort the dataframe by the `A` column
    sorted_df = sort_dataframe(df)

    # Assert that the sorted dataframe is sorted by the `A` column
    assert sorted_df['Date'].tolist() == [1, 2, 3]


def test_encode_team_names():
    """Test that the `encode_strings` function encodes the unique strings in the specified column correctly."""

    # Create a dataframe with a column of unique strings
    df = pd.DataFrame({'Home': ['K. Žalgiris', 'Nevėžis', 'Palanga'], 'Away': [ 'Stumbras', 'Sūduva', 'Šiauliai']})
    with open('./pre-processing/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    # Encode the unique strings in the `Strings` column
    encoded_df = encode_team_names(df, encoder)

    # Assert that the encoded strings are encoded correctly
    assert encoded_df['Home'].dtypes == 'int'
    assert encoded_df['Away'].dtypes == 'int'

if __name__ == "__main__":
    pytest.main()
