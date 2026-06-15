from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
    load_data,
    prepare_features,
)


RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=450,
            min_samples_leaf=1,
            max_features=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=450,
            min_samples_leaf=1,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.035,
            max_depth=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }


def train_and_save() -> dict:
    data = load_data()
    X = prepare_features(data)
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    best_name = ""
    best_score = -np.inf

    for name, regressor in candidate_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor()), ("regressor", regressor)]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, scoring="r2", cv=folds, n_jobs=-1
        )
        comparison[name] = {
            "cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std": round(float(scores.std()), 4),
        }
        if scores.mean() > best_score:
            best_name = name
            best_score = float(scores.mean())

    final_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", candidate_models()[best_name]),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    predictions = final_pipeline.predict(X_test)

    metrics = {
        "selected_model": best_name,
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 4
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(data)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_comparison": comparison,
        "actual": [round(float(value), 3) for value in y_test],
        "predicted": [round(float(value), 3) for value in predictions],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_and_save()
    print(f"Selected model: {result['selected_model']}")
    print(f"Test R2: {result['r2']:.3f}")
    print(f"Test MAE: {result['mae']:.3f} lakh")
    print(f"Test RMSE: {result['rmse']:.3f} lakh")
