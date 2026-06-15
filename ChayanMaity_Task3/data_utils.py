from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "car_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

REFERENCE_YEAR = 2019
TARGET = "Selling_Price"
NUMERIC_FEATURES = ["Present_Price", "Driven_kms", "Owner", "Car_Age"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and lightly clean the internship dataset."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data.drop_duplicates().reset_index(drop=True)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rows into the columns expected by the saved pipeline."""
    prepared = data.copy()
    if "Car_Age" not in prepared.columns:
        prepared["Car_Age"] = REFERENCE_YEAR - prepared["Year"]
    return prepared[MODEL_FEATURES]


def make_prediction_row(
    present_price: float,
    driven_kms: int,
    owner: int,
    year: int,
    fuel_type: str,
    selling_type: str,
    transmission: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Present_Price": present_price,
                "Driven_kms": driven_kms,
                "Owner": owner,
                "Car_Age": REFERENCE_YEAR - year,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
            }
        ]
    )
