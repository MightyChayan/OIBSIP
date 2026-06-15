from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    EMPLOYED,
    PARTICIPATION,
    UNEMPLOYMENT,
    clean_data,
    filter_data,
    monthly_summary,
    region_summary,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "unemployment_india.csv"
COLORS = {
    "ink": "#1f2937",
    "indigo": "#243b67",
    "saffron": "#e58a2b",
    "green": "#3f7d6b",
    "paper": "#f6f3ed",
    "red": "#b84a4a",
}

st.set_page_config(
    page_title="India Unemployment Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1f2937;
            --indigo: #243b67;
            --saffron: #e58a2b;
            --muted: #667085;
            --paper: #f6f3ed;
            --line: #ded8ce;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        #MainMenu { visibility: hidden; }
        html, body, [class*="css"] {
            font-family: "Segoe UI", Arial, sans-serif;
        }
        h1, h2, h3 {
            color: var(--indigo);
            letter-spacing: -0.015em;
        }
        [data-testid="stSidebar"] {
            background: #1f3763;
            border-right: 4px solid var(--saffron);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #ffffff;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #d8e1ef;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, .22);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border-color: #c7d0dc;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div {
            color: #475467;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: #eef2f7;
            border: 1px solid #aeb9c8;
            color: #243b67;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
            background: #e1e7ef;
            border-color: #8291a6;
            color: #172b4d;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [data-baseweb="input"] svg {
            color: #1f2937;
            fill: #475467;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #e58a2b;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] span,
        [data-testid="stSidebar"] [data-baseweb="tag"] svg {
            color: #172033;
            fill: #172033;
        }
        [data-testid="stSidebar"] button[kind="header"] svg {
            color: #ffffff;
            fill: #ffffff;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 3px solid var(--saffron);
            border-radius: 7px;
            padding: 15px 17px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--indigo); }
        .hero {
            padding: 28px 32px;
            border-radius: 8px;
            margin-bottom: 22px;
            background: var(--indigo);
            border-bottom: 6px solid var(--saffron);
        }
        .hero .eyebrow {
            color: #f4bd75;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .11em;
        }
        .hero h1 {
            color: white;
            margin: .4rem 0 .55rem;
            font-size: 2.15rem;
        }
        .hero p {
            color: #e5e7eb;
            margin: 0;
            max-width: 780px;
            line-height: 1.55;
        }
        .insight {
            background: #fffaf2;
            border: 1px solid #ead8bb;
            border-left: 4px solid var(--saffron);
            border-radius: 5px;
            padding: 14px 16px;
            color: var(--ink);
            margin: 8px 0 20px;
        }
        div[data-testid="stPlotlyChart"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 6px;
        }
        .footer {
            text-align: center;
            color: var(--muted);
            font-size: .82rem;
            padding: 28px 0 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_default_data(path: str) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return clean_data(pd.read_csv(BytesIO(file_bytes)))


def chart_layout(fig: go.Figure, height: int = 410) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=55, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color=COLORS["ink"]),
        title_font=dict(size=17, color=COLORS["indigo"]),
        hoverlabel=dict(bgcolor="white"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d6d3d1")
    fig.update_yaxes(gridcolor="#ebe7df", zeroline=False)
    return fig


def fmt_number(value: float) -> str:
    if value >= 10_000_000:
        return f"{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"{value / 100_000:.2f} L"
    return f"{value:,.0f}"


apply_styles()

with st.sidebar:
    st.markdown("## Unemployment Study")
    st.caption("India, May 2019 to June 2020")
    uploaded_file = st.file_uploader("Upload a different CSV", type="csv")

try:
    data = (
        load_uploaded_data(uploaded_file.getvalue())
        if uploaded_file is not None
        else load_default_data(str(DEFAULT_DATA))
    )
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"The selected CSV could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")
    all_regions = sorted(data["Region"].unique())
    all_areas = sorted(data["Area"].unique())
    selected_regions = st.multiselect("State / region", all_regions, placeholder="All regions")
    selected_areas = st.multiselect("Area", all_areas, default=all_areas)
    selected_dates = st.date_input(
        "Study period",
        value=(data["Date"].min().date(), data["Date"].max().date()),
        min_value=data["Date"].min().date(),
        max_value=data["Date"].max().date(),
    )
    st.markdown("---")
    st.caption("Source: CMIE unemployment sample supplied with the internship task.")

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    date_filter = (pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1]))
else:
    date_filter = (data["Date"].min(), data["Date"].max())

filtered = filter_data(data, selected_regions, selected_areas, date_filter)
if filtered.empty:
    st.warning("No records match the current filters. Widen the selection to continue.")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <h1>Unemployment Analysis in India</h1>
      <p>This dashboard examines how unemployment changed across Indian states,
      with separate views for rural and urban areas and a closer look at the
      months affected by the first COVID-19 wave.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

average_rate = filtered[UNEMPLOYMENT].mean()
average_participation = filtered[PARTICIPATION].mean()
average_employed = filtered[EMPLOYED].mean()
peak_row = filtered.loc[filtered[UNEMPLOYMENT].idxmax()]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Average unemployment", f"{average_rate:.2f}%")
kpi2.metric("Average employed", fmt_number(average_employed))
kpi3.metric("Labour participation", f"{average_participation:.2f}%")
kpi4.metric("Peak recorded rate", f"{peak_row[UNEMPLOYMENT]:.2f}%", peak_row["Region"])

st.markdown("### How unemployment changed over time")
monthly = monthly_summary(filtered)
trend = px.line(
    monthly,
    x="Date",
    y="unemployment_rate",
    markers=True,
    labels={"Date": "", "unemployment_rate": "Unemployment rate (%)"},
    title="Average unemployment rate over time",
)
trend.update_traces(line=dict(color=COLORS["indigo"], width=3), marker=dict(size=7))
trend.add_vrect(
    x0="2020-03-01",
    x1="2020-06-30",
    fillcolor=COLORS["saffron"],
    opacity=0.13,
    line_width=0,
    annotation_text="COVID period",
    annotation_position="top left",
)
st.plotly_chart(chart_layout(trend), width="stretch")

left, right = st.columns((1.1, 0.9))
with left:
    regions = region_summary(filtered).head(12).sort_values("unemployment_rate")
    ranking = px.bar(
        regions,
        x="unemployment_rate",
        y="Region",
        orientation="h",
        color="unemployment_rate",
        color_continuous_scale=["#f2dfc3", COLORS["saffron"], COLORS["red"]],
        labels={"unemployment_rate": "Average unemployment (%)", "Region": ""},
        title="Regions with the highest average unemployment",
        text_auto=".1f",
    )
    ranking.update_layout(coloraxis_showscale=False)
    st.plotly_chart(chart_layout(ranking, 470), width="stretch")

with right:
    area_monthly = (
        filtered.groupby(["Date", "Area"], as_index=False)[UNEMPLOYMENT].mean()
    )
    area_chart = px.line(
        area_monthly,
        x="Date",
        y=UNEMPLOYMENT,
        color="Area",
        markers=True,
        color_discrete_map={"Rural": COLORS["green"], "Urban": COLORS["saffron"]},
        labels={"Date": "", UNEMPLOYMENT: "Unemployment rate (%)"},
        title="Rural vs urban unemployment",
    )
    area_chart.update_traces(line=dict(width=3))
    st.plotly_chart(chart_layout(area_chart, 470), width="stretch")

st.markdown("### Additional comparisons")
tab1, tab2, tab3 = st.tabs(["Distribution", "Participation", "Data table"])

with tab1:
    distribution = px.box(
        filtered,
        x="Area",
        y=UNEMPLOYMENT,
        color="Area",
        points="outliers",
        color_discrete_map={"Rural": COLORS["green"], "Urban": COLORS["saffron"]},
        labels={"Area": "", UNEMPLOYMENT: "Unemployment rate (%)"},
        title="Unemployment distribution by area",
    )
    distribution.update_layout(showlegend=False)
    st.plotly_chart(chart_layout(distribution), width="stretch")

with tab2:
    scatter = px.scatter(
        filtered,
        x=PARTICIPATION,
        y=UNEMPLOYMENT,
        color="Area",
        hover_data=["Region", "Date"],
        opacity=0.72,
        trendline="ols",
        color_discrete_map={"Rural": COLORS["green"], "Urban": COLORS["saffron"]},
        labels={
            PARTICIPATION: "Labour participation rate (%)",
            UNEMPLOYMENT: "Unemployment rate (%)",
        },
        title="Labour participation and unemployment",
    )
    st.plotly_chart(chart_layout(scatter), width="stretch")

with tab3:
    display_columns = ["Date", "Region", "Area", UNEMPLOYMENT, EMPLOYED, PARTICIPATION]
    st.dataframe(
        filtered[display_columns].sort_values("Date", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn(format="MMM YYYY"),
            EMPLOYED: st.column_config.NumberColumn(format="localized"),
            UNEMPLOYMENT: st.column_config.NumberColumn(format="%.2f%%"),
            PARTICIPATION: st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.download_button(
        "Download filtered data",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_unemployment_india.csv",
        mime="text/csv",
    )

st.markdown(
    '<div class="footer">Unemployment Analysis in India</div>',
    unsafe_allow_html=True,
)
