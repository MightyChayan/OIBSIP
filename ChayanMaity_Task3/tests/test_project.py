import pandas as pd

from data_utils import MODEL_FEATURES, load_data, make_prediction_row, prepare_features


def test_dataset_is_clean_after_loading():
    data = load_data()
    assert len(data) == 299
    assert not data.isna().any().any()
    assert not data.duplicated().any()


def test_prepare_features_returns_expected_columns():
    data = pd.DataFrame(
        [
            {
                "Year": 2015,
                "Present_Price": 8.5,
                "Driven_kms": 32000,
                "Owner": 0,
                "Fuel_Type": "Petrol",
                "Selling_type": "Dealer",
                "Transmission": "Manual",
            }
        ]
    )
    result = prepare_features(data)
    assert list(result.columns) == MODEL_FEATURES
    assert result.loc[0, "Car_Age"] == 4


def test_prediction_row_matches_model_schema():
    row = make_prediction_row(8.5, 32000, 0, 2015, "Petrol", "Dealer", "Manual")
    assert list(row.columns) == MODEL_FEATURES
    assert row.loc[0, "Present_Price"] == 8.5
