import unittest
from pathlib import Path

import pandas as pd

from analysis import UNEMPLOYMENT, clean_data, covid_summary, filter_data


DATA_PATH = Path(__file__).parents[1] / "data" / "unemployment_india.csv"


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = clean_data(pd.read_csv(DATA_PATH))

    def test_clean_data_removes_blank_rows_and_normalizes_columns(self):
        self.assertEqual(len(self.data), 740)
        self.assertIn("Date", self.data.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.data["Date"]))
        self.assertFalse(self.data.isna().any().any())

    def test_filter_data_applies_region_area_and_date(self):
        result = filter_data(
            self.data,
            regions=["Delhi"],
            areas=["Urban"],
            date_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")),
        )
        self.assertFalse(result.empty)
        self.assertEqual(set(result["Region"]), {"Delhi"})
        self.assertEqual(set(result["Area"]), {"Urban"})
        self.assertGreaterEqual(result["Date"].min(), pd.Timestamp("2020-01-01"))

    def test_covid_unemployment_is_higher_than_pre_covid(self):
        result = covid_summary(self.data)
        self.assertGreater(result["covid_mean"], result["pre_covid_mean"])
        self.assertGreater(result["percentage_point_change"], 0)
        self.assertLess(result["p_value"], 0.05)

    def test_unemployment_values_are_in_expected_range(self):
        self.assertGreaterEqual(self.data[UNEMPLOYMENT].min(), 0)
        self.assertLessEqual(self.data[UNEMPLOYMENT].max(), 100)


if __name__ == "__main__":
    unittest.main()
