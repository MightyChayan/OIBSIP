"""Make an Iris species prediction from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("outputs/iris_model.joblib")


def load_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a trained model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python -m src.train' first."
        )
    return joblib.load(path)


def predict_species(
    measurements: list[float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[str, dict[str, float]]:
    """Predict one species and return class probabilities."""
    artifact = load_artifact(model_path)
    features = artifact["features"]
    if len(measurements) != len(features):
        raise ValueError(f"Expected {len(features)} measurements, got {len(measurements)}.")
    if any(value <= 0 for value in measurements):
        raise ValueError("All measurements must be greater than zero.")

    sample = pd.DataFrame([measurements], columns=features)
    model = artifact["model"]
    prediction = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    probability_map = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    return prediction, probability_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one Iris flower.")
    parser.add_argument("sepal_length", type=float, help="Sepal length in cm")
    parser.add_argument("sepal_width", type=float, help="Sepal width in cm")
    parser.add_argument("petal_length", type=float, help="Petal length in cm")
    parser.add_argument("petal_width", type=float, help="Petal width in cm")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    ]
    prediction, probabilities = predict_species(values, args.model)
    print(f"Predicted species: Iris-{prediction.lower()}")
    print("Class probabilities:")
    for label, probability in sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()

