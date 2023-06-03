

from run import go
import unittest
import csv
from argparse import ArgumentParser
from datetime import datetime
import sys

sys.path.append(
    "/home/orestas41/project-FootballPredict/lithuanian-football-league-predictor/data_scrape")


class TestGo(unittest.TestCase):

    def test_can_scrape_data(self):
        # Arrange
        args = ArgumentParser().parse_args(
            ["--step_description", "Test data scraping"])

        # Act
        go(args)

        # Assert
        with open(f"../raw_data/{datetime.now().strftime('%Y-%m-%d')}.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
