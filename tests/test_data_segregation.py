import unittest
import csv
import pandas as pd
import argparse
from datetime import datetime
import sys

sys.path.append(
    "/home/orestas41/project-FootballPredict/lithuanian-football-league-predictor/data_segregation")
    
from run import go

class TestGo(unittest.TestCase):

    def test_split_data(self):
        """Test that the go function correctly splits the input dataframe into trainval and test sets."""
        # Create a small dataframe
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})

        # Call the go function
        args = argparse.Namespace(input='processed_data.csv:latest', test_size=0.2)
        trainval, test = go(args)

        # Check the sizes of the trainval and test sets.
        trainval_size = len(trainval)
        test_size = len(test)

        print(trainval_size)
        print(test_size)

        self.assertTrue(trainval_size == round(0.8 * (test_size + trainval_size)))
        self.assertTrue(test_size == round(0.2 * (test_size + trainval_size)))

        # Check that the data in the trainval and test sets are not overlapping.
        trainval_unique_values = trainval.drop_duplicates()
        test_unique_values = test.drop_duplicates()

        self.assertTrue(len(trainval_unique_values.merge(test_unique_values, how='inner')) == 0)

if __name__ == "__main__":
    unittest.main()
