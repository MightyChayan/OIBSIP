from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data_utils import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    REFERENCE_YEAR,
    load_data,
    make_prediction_row,
)


st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --ink: #111827;
        --muted: #5b6778;
        --primary: #1f3a5f;
        --primary-dark: #172a45;
        --accent: #b56a2a;
        --page: #eef1f4;
        --panel: #ffffff;
        --line: #cfd6df;
        --soft: #e8edf3;
    }
    .stApp {
        background:
            linear-gradient(180deg, #e3e8ee 0%, var(--page) 280px),
            var(--page);
        color: var(--ink);
    }
    .block-container {
        max-width: none;
        width: 100%;
        padding: 1.6rem clamp(1.1rem, 3vw, 3rem) 3rem;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: 0;
    }
    p, label, div { font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu { visibility: hidden; }
    .hero {
        position: relative;
        overflow: hidden;
        min-height: 210px;
        padding: clamp(2rem, 4vw, 3.2rem);
        border: 1px solid var(--line);
        border-left: 5px solid var(--primary);
        border-radius: 10px;
        background:
            linear-gradient(135deg, #ffffff 0%, #f4f6f8 58%, #f7efe7 100%);
        color: var(--ink);
        margin-bottom: 1.35rem;
        box-shadow: 0 12px 30px rgba(16, 42, 67, .08);
    }
    .hero h1 {
        color: var(--ink) !important;
        font-size: clamp(2.3rem, 4vw, 4rem);
        line-height: 1.08;
        margin: 0 0 .75rem;
        white-space: nowrap;
    }
    .hero p {
        max-width: none;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0;
        white-space: nowrap;
    }
    .soft-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 8px 22px rgba(16, 42, 67, .06);
    }
    .result-card {
        background: var(--panel);
        color: var(--ink);
        border: 1px solid var(--line);
        border-left: 5px solid var(--primary);
        border-radius: 10px;
        padding: 1.5rem 1.7rem;
        margin-top: .5rem;
        box-shadow: 0 8px 22px rgba(16, 42, 67, .06);
    }
    .result-label { color: var(--muted); font-size: .78rem; letter-spacing: .08em; font-weight: 700; }
    .result-price {
        color: var(--ink);
        font-family: 'Inter', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        margin: .1rem 0;
    }
    .result-note { color: var(--muted); font-size: .88rem; }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 3px solid var(--accent);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(16, 42, 67, .05);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(180deg, #24466f 0%, var(--primary-dark) 100%);
        color: white;
        border: 1px solid var(--primary-dark);
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0;
        min-height: 48px;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #111f34; color: white; border-color: #111f34;
    }
    [data-baseweb="tab-list"] {
        gap: .7rem;
        padding: .35rem 0 .9rem;
    }
    [data-baseweb="tab"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .65rem 1.2rem;
    }
    [aria-selected="true"] {
        background: var(--soft) !important;
        color: var(--ink) !important;
        border-bottom: 3px solid var(--primary) !important;
    }
    [data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1.35rem;
        box-shadow: 0 8px 22px rgba(16, 42, 67, .06);
    }
    [data-testid="stDataFrame"], [data-testid="stAlert"] {
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    [data-testid="stVerticalBlock"] { gap: 1rem; }
    [data-testid="stHorizontalBlock"] { gap: 1.25rem; }
    [data-testid="stPlotlyChart"], [data-testid="stImage"], [data-testid="stDataFrame"] {
        width: 100%;
    }
    @media (min-width: 1400px) {
        .hero { min-height: 240px; }
        [data-testid="stForm"] { padding: 1.55rem; }
        .soft-card { padding: 1.55rem 1.65rem; }
    }
    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.5rem; min-height: auto; }
        .hero h1, .hero p { white-space: normal; }
    }
    hr { border-color: var(--line) !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def rupees_from_lakh(value: float) -> str:
    return f"₹{value * 100_000:,.0f}"


def draw_prediction_plot(metrics: dict):
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    actual = np.asarray(metrics["actual"])
    predicted = np.asarray(metrics["predicted"])
    sns.scatterplot(x=actual, y=predicted, color="#1f3a5f", s=65, ax=axis)
    bounds = [0, max(actual.max(), predicted.max()) + 1]
    axis.plot(bounds, bounds, "--", color="#b56a2a", linewidth=1.7)
    axis.set(xlabel="Actual price (lakh)", ylabel="Predicted price (lakh)")
    axis.tick_params(colors="#5b6778")
    axis.xaxis.label.set_color("#111827")
    axis.yaxis.label.set_color("#111827")
    sns.despine()
    return figure


def draw_price_distribution(data: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    sns.histplot(
        data=data,
        x="Selling_Price",
        bins=24,
        color="#7d8ea3",
        edgecolor="#ffffff",
        ax=axis,
    )
    axis.set(xlabel="Selling price (lakh)", ylabel="Number of cars")
    axis.tick_params(colors="#5b6778")
    axis.xaxis.label.set_color("#111827")
    axis.yaxis.label.set_color("#111827")
    sns.despine()
    return figure


data = get_data()
model = load_model()
metrics = get_metrics()

st.markdown(
    """
    <section class="hero">
        <h1>CAR VALUE PREDICTOR</h1>
        <p>Estimate a used car's resale value from its age, original price, mileage, transmission and ownership details.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

predict_tab, insights_tab, model_tab = st.tabs(
    ["Price estimator", "Dataset insights", "Model report"]
)

with predict_tab:
    st.subheader("Enter the vehicle details")
    form_col, guide_col = st.columns([2.1, 1], gap="large")

    with form_col:
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                present_price = st.number_input(
                    "Original showroom price (₹ lakh)",
                    min_value=0.30,
                    max_value=100.00,
                    value=8.00,
                    step=0.25,
                    help="The price of the car when it was new.",
                )
                year = st.selectbox(
                    "Year of purchase",
                    options=list(range(2018, 2002, -1)),
                    index=4,
                    help="This dataset contains vehicles from 2003 to 2018.",
                )
                fuel_type = st.selectbox(
                    "Fuel type", sorted(data["Fuel_Type"].unique())
                )
                owner = st.selectbox(
                    "Previous owners",
                    options=sorted(data["Owner"].unique()),
                    format_func=lambda value: {
                        0: "First owner",
                        1: "Second owner",
                        3: "Fourth owner",
                    }.get(value, f"{int(value) + 1} owners"),
                )
            with right:
                driven_kms = st.number_input(
                    "Distance driven (km)",
                    min_value=500,
                    max_value=500_000,
                    value=35_000,
                    step=1_000,
                )
                transmission = st.radio(
                    "Transmission",
                    options=sorted(data["Transmission"].unique()),
                    horizontal=True,
                )
                selling_type = st.radio(
                    "Seller type",
                    options=sorted(data["Selling_type"].unique()),
                    horizontal=True,
                )
            submitted = st.form_submit_button(
                "Estimate resale price", width="stretch"
            )

        if submitted:
            prediction_row = make_prediction_row(
                present_price,
                driven_kms,
                int(owner),
                int(year),
                fuel_type,
                selling_type,
                transmission,
            )
            predicted_lakh = max(0.0, float(model.predict(prediction_row)[0]))
            typical_error = metrics["mae"]
            low = max(0.0, predicted_lakh - typical_error)
            high = predicted_lakh + typical_error
            depreciation = max(0.0, 100 * (1 - predicted_lakh / present_price))
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">ESTIMATED RESALE VALUE</div>
                    <div class="result-price">{rupees_from_lakh(predicted_lakh)}</div>
                    <div class="result-note">
                        Model estimate: {predicted_lakh:.2f} lakh · Typical range
                        {low:.2f}–{high:.2f} lakh · Approx. depreciation {depreciation:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with guide_col:
        st.markdown(
            """
            <div class="soft-card">
                <h3 style="margin-top:0">Before you estimate</h3>
                <p>The model was trained on 299 unique listings collected for
                this internship dataset.</p>
                <p><b>Price unit:</b> one lakh equals ₹100,000.</p>
                <p><b>Best use:</b> comparing ordinary used cars represented by
                the 2003–2018 sample.</p>
                <p><b>Remember:</b> condition, city, service history and market
                demand are not available in the data, so this is a guide rather
                than a dealer quotation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Dataset range: ₹{data['Selling_Price'].min():.2f}–"
            f"₹{data['Selling_Price'].max():.2f} lakh"
        )

with insights_tab:
    st.subheader("What is inside the dataset?")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Unique listings", f"{len(data):,}")
    metric_cols[1].metric("Average resale price", f"₹{data['Selling_Price'].mean():.2f} L")
    metric_cols[2].metric("Median kilometres", f"{data['Driven_kms'].median():,.0f}")
    metric_cols[3].metric(
        "Vehicle years", f"{int(data['Year'].min())}–{int(data['Year'].max())}"
    )

    chart_col, summary_col = st.columns([1.75, 1], gap="large")
    with chart_col:
        st.pyplot(draw_price_distribution(data), width="stretch")
    with summary_col:
        st.markdown("#### Median resale price by fuel")
        fuel_summary = (
            data.groupby("Fuel_Type", as_index=False)["Selling_Price"]
            .median()
            .sort_values("Selling_Price", ascending=False)
            .rename(
                columns={
                    "Fuel_Type": "Fuel type",
                    "Selling_Price": "Median price (lakh)",
                }
            )
        )
        st.dataframe(fuel_summary, hide_index=True, width="stretch")
        st.info(
            "Only two CNG cars are present, so that category should be "
            "interpreted cautiously."
        )

    with st.expander("View cleaned data"):
        st.dataframe(data, hide_index=True, width="stretch")

with model_tab:
    st.subheader("How well does the model perform?")
    score_cols = st.columns(4)
    score_cols[0].metric("Selected model", metrics["selected_model"])
    score_cols[1].metric("Test R²", f"{metrics['r2']:.3f}")
    score_cols[2].metric("Mean absolute error", f"₹{metrics['mae']:.2f} L")
    score_cols[3].metric("Test records", metrics["test_rows"])

    plot_col, detail_col = st.columns([1.75, 1], gap="large")
    with plot_col:
        st.pyplot(draw_prediction_plot(metrics), width="stretch")
    with detail_col:
        st.markdown("#### Cross-validation comparison")
        comparison = pd.DataFrame(metrics["model_comparison"]).T.reset_index()
        comparison.columns = ["Model", "Mean CV R²", "CV variation"]
        st.dataframe(
            comparison.sort_values("Mean CV R²", ascending=False),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            """
            The pipeline standardizes numeric values, one-hot encodes categories
            and then applies the selected tree-based regressor. Model selection
            used five-fold cross-validation on the training set; the displayed
            R² and error are from a separate 20% test split.
            """
        )
        st.caption(
            f"Car age is measured against {REFERENCE_YEAR}, one year after the "
            "latest vehicle in this historical dataset."
        )
