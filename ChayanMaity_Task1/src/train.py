"""Train, evaluate, and save models for Iris flower classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "Species"
ID_COLUMN = "Id"
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
DISPLAY_NAMES = {
    "SepalLengthCm": "Sepal Length (cm)",
    "SepalWidthCm": "Sepal Width (cm)",
    "PetalLengthCm": "Petal Length (cm)",
    "PetalWidthCm": "Petal Width (cm)",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the Iris dataset."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    data = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data.copy()
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype(str).str.replace("Iris-", "", regex=False).str.title()
    )

    if data[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Required feature or target columns contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(data[col]) for col in FEATURE_COLUMNS):
        raise ValueError("All measurement columns must contain numeric values.")
    if (data[FEATURE_COLUMNS] <= 0).any().any():
        raise ValueError("Flower measurements must be greater than zero.")

    return data


def build_models() -> dict[str, BaseEstimator]:
    """Return candidate models with appropriate preprocessing."""
    return {
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier()),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return compact grids suitable for the small Iris dataset."""
    return {
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": list(range(3, 16, 2)),
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 2, 3, 4, 5],
            "min_samples_split": [2, 4, 6],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 4],
            "max_features": ["sqrt", None],
        },
    }


def save_eda_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Create presentation-ready exploratory data analysis charts."""
    sns.set_theme(style="whitegrid", palette="Set2")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for feature, axis in zip(FEATURE_COLUMNS, axes.flat):
        sns.histplot(
            data=data,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            element="step",
            ax=axis,
        )
        axis.set_title(f"Distribution of {DISPLAY_NAMES[feature]}")
        axis.set_xlabel(DISPLAY_NAMES[feature])
    fig.suptitle("Iris Feature Distributions by Species", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pair_grid = sns.pairplot(
        data[FEATURE_COLUMNS + [TARGET_COLUMN]],
        hue=TARGET_COLUMN,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.8, "s": 45},
    )
    pair_grid.fig.suptitle("Pairwise Relationships Between Iris Features", y=1.02)
    pair_grid.savefig(output_dir / "pairplot.png", dpi=180, bbox_inches="tight")
    plt.close(pair_grid.fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    correlation = data[FEATURE_COLUMNS].corr()
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
        ax=axis,
    )
    axis.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x=TARGET_COLUMN, hue=TARGET_COLUMN, legend=False, ax=axis)
    axis.set_title("Class Distribution", fontweight="bold")
    axis.set_xlabel("Species")
    axis.set_ylabel("Number of Samples")
    for container in axis.containers:
        axis.bar_label(container)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare_models(
    models: dict[str, BaseEstimator],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """Compare candidates with cross-validated training accuracy."""
    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "mean_cv_accuracy": scores.mean(),
                "std_cv_accuracy": scores.std(),
                "min_cv_accuracy": scores.min(),
                "max_cv_accuracy": scores.max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_cv_accuracy", ascending=False, ignore_index=True
    )


def save_model_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Save a model comparison chart."""
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        comparison["model"],
        comparison["mean_cv_accuracy"],
        yerr=comparison["std_cv_accuracy"],
        capsize=6,
        color=["#2a9d8f", "#e9c46a", "#f4a261"],
    )
    axis.set_ylim(0.8, 1.01)
    axis.set_ylabel("Mean Cross-Validation Accuracy")
    axis.set_title("Model Comparison (5-Fold Cross-Validation)", fontweight="bold")
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in bars.datavalues])
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(
    data_path: str | Path = "Iris.csv",
    output_dir: str | Path = "outputs",
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Run the complete training workflow and return summary metrics."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    save_eda_plots(data, output_path)

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()
    comparison = compare_models(models, x_train, y_train, cv)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    save_model_comparison(comparison, output_path)

    selected_name = str(comparison.iloc[0]["model"])
    search = GridSearchCV(
        estimator=models[selected_name],
        param_grid=parameter_grids()[selected_name],
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(x_test)

    labels = sorted(y.unique())
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_path / "classification_report.csv"
    )

    prediction_frame = x_test.copy()
    prediction_frame["ActualSpecies"] = y_test
    prediction_frame["PredictedSpecies"] = predictions
    prediction_frame["Correct"] = prediction_frame["ActualSpecies"].eq(
        prediction_frame["PredictedSpecies"]
    )
    prediction_frame.sort_index().to_csv(
        output_path / "test_predictions.csv", index_label="OriginalRow"
    )

    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix - {selected_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "selected_model": selected_name,
        "best_parameters": search.best_params_,
        "best_cross_validation_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "classes": labels,
        "random_state": RANDOM_STATE,
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "classes": labels,
        "metrics": metrics,
    }
    joblib.dump(artifact, output_path / "iris_model.joblib")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Iris flower classifiers."
    )
    parser.add_argument("--data", default="Iris.csv", help="Path to the CSV dataset.")
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for generated artifacts."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for final testing (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(args.data, args.output_dir, args.test_size)
    print("\nTraining complete")
    print(f"Selected model: {metrics['selected_model']}")
    print(
        "Best cross-validation accuracy: "
        f"{metrics['best_cross_validation_accuracy']:.3f}"
    )
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Artifacts saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

