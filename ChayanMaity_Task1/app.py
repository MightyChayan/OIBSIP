from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_artifact

MODEL_PATH = Path("outputs/iris_model.joblib")
OUTPUT_DIR = Path("outputs")

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="I",
    layout="centered",
)


@st.cache_resource
def load_saved_model() -> dict:
    """Load the saved model once."""
    return load_artifact(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.train` first.")
    st.stop()

artifact = load_saved_model()
model = artifact["model"]
metrics = artifact["metrics"]

st.title("Iris Flower Classification")
st.write(
    "This project uses machine learning to predict the species of an Iris "
    "flower from its sepal and petal measurements."
)

prediction_tab, results_tab = st.tabs(["Prediction", "Project Results"])

with prediction_tab:
    st.subheader("Enter Flower Measurements")
    st.write("Adjust the values in the sidebar, then click the prediction button.")

    with st.sidebar:
        st.header("Input Measurements")
        st.caption("All measurements are in centimeters.")

        sepal_length = st.slider(
            "Sepal length",
            min_value=4.0,
            max_value=8.0,
            value=5.8,
            step=0.1,
        )
        sepal_width = st.slider(
            "Sepal width",
            min_value=2.0,
            max_value=4.5,
            value=3.0,
            step=0.1,
        )
        petal_length = st.slider(
            "Petal length",
            min_value=1.0,
            max_value=7.0,
            value=4.3,
            step=0.1,
        )
        petal_width = st.slider(
            "Petal width",
            min_value=0.1,
            max_value=2.5,
            value=1.3,
            step=0.1,
        )

    input_data = pd.DataFrame(
        [
            {
                "SepalLengthCm": sepal_length,
                "SepalWidthCm": sepal_width,
                "PetalLengthCm": petal_length,
                "PetalWidthCm": petal_width,
            }
        ]
    )

    st.write("**Current measurements**")
    display_data = input_data.rename(
        columns={
            "SepalLengthCm": "Sepal Length",
            "SepalWidthCm": "Sepal Width",
            "PetalLengthCm": "Petal Length",
            "PetalWidthCm": "Petal Width",
        }
    )
    st.dataframe(display_data, hide_index=True, width="stretch")

    if st.button("Predict Species", type="primary", width="stretch"):
        prediction = str(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities.max())

        st.success(f"Predicted Species: Iris-{prediction.lower()}")
        st.write(f"Prediction confidence: **{confidence:.1%}**")

        probability_data = pd.DataFrame(
            {
                "Species": model.classes_,
                "Probability": probabilities,
            }
        ).set_index("Species")

        st.write("**Probability for each species**")
        st.bar_chart(probability_data)

with results_tab:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
    col2.metric(
        "Cross-Validation",
        f"{metrics['best_cross_validation_accuracy']:.1%}",
    )
    col3.metric("Macro F1-Score", f"{metrics['test_f1_macro']:.1%}")

    st.write(f"**Selected model:** {metrics['selected_model']}")
    st.write(
        "I compared K-Nearest Neighbors, Decision Tree, and Random Forest. "
        "KNN gave the best cross-validation result."
    )

    st.subheader("Model Comparison")
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    comparison_display = comparison[
        ["model", "mean_cv_accuracy", "std_cv_accuracy"]
    ].copy()
    comparison_display.columns = [
        "Model",
        "Mean CV Accuracy",
        "Standard Deviation",
    ]
    st.dataframe(
        comparison_display.style.format(
            {
                "Mean CV Accuracy": "{:.3f}",
                "Standard Deviation": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Confusion Matrix")
    st.image(
        str(OUTPUT_DIR / "confusion_matrix.png"),
        caption="Performance on the 30 test samples",
        width="stretch",
    )

    with st.expander("How the model was trained"):
        st.markdown(
            """
            1. Loaded and checked the Iris dataset.
            2. Removed the `Id` column because it is not a flower measurement.
            3. Split the data into 80% training and 20% testing data.
            4. Compared three classification algorithms using 5-fold
               cross-validation.
            5. Tuned the best model and evaluated it on the test data.
            6. Saved the trained model for future predictions.
            """
        )
