"""Data preparation and analysis helpers for the unemployment dashboard."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


UNEMPLOYMENT = "Estimated Unemployment Rate (%)"
EMPLOYED = "Estimated Employed"
PARTICIPATION = "Estimated Labour Participation Rate (%)"
REQUIRED_COLUMNS = {
    "Region",
    "Date",
    "Frequency",
    UNEMPLOYMENT,
    EMPLOYED,
    PARTICIPATION,
    "Area",
}


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the supplied internship dataset and validate its schema."""
    data = raw_data.copy()
    data.columns = data.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = data.dropna(how="all").drop_duplicates()
    for column in ("Region", "Frequency", "Area"):
        data[column] = data[column].astype("string").str.strip()

    data["Date"] = pd.to_datetime(
        data["Date"].astype("string").str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    for column in (UNEMPLOYMENT, EMPLOYED, PARTICIPATION):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Region", "Date", UNEMPLOYMENT, EMPLOYED, PARTICIPATION, "Area"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.strftime("%b %Y")
    data["Period"] = np.select(
        [
            data["Date"] < pd.Timestamp("2020-03-01"),
            data["Date"].between(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30")),
        ],
        ["Pre-COVID", "COVID period"],
        default="Other",
    )
    return data.sort_values(["Date", "Region", "Area"]).reset_index(drop=True)


def filter_data(
    data: pd.DataFrame,
    regions: Iterable[str] | None = None,
    areas: Iterable[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters without mutating the source frame."""
    filtered = data.copy()
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if areas:
        filtered = filtered[filtered["Area"].isin(areas)]
    if date_range:
        start, end = (pd.Timestamp(value) for value in date_range)
        filtered = filtered[filtered["Date"].between(start, end)]
    return filtered


def covid_summary(data: pd.DataFrame) -> dict[str, float]:
    """Compare mean unemployment before and during the first COVID wave."""
    pre_covid = data.loc[data["Period"] == "Pre-COVID", UNEMPLOYMENT]
    covid = data.loc[data["Period"] == "COVID period", UNEMPLOYMENT]
    pre_mean = float(pre_covid.mean()) if not pre_covid.empty else np.nan
    covid_mean = float(covid.mean()) if not covid.empty else np.nan
    change = covid_mean - pre_mean if not (np.isnan(pre_mean) or np.isnan(covid_mean)) else np.nan

    p_value = np.nan
    if len(pre_covid) > 1 and len(covid) > 1:
        p_value = float(stats.ttest_ind(pre_covid, covid, equal_var=False, nan_policy="omit").pvalue)

    return {
        "pre_covid_mean": pre_mean,
        "covid_mean": covid_mean,
        "percentage_point_change": change,
        "p_value": p_value,
    }


def monthly_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the core measures by month."""
    return (
        data.groupby("Date", as_index=False)
        .agg(
            unemployment_rate=(UNEMPLOYMENT, "mean"),
            employed=(EMPLOYED, "sum"),
            labour_participation=(PARTICIPATION, "mean"),
        )
        .sort_values("Date")
    )


def region_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the core measures by region."""
    return (
        data.groupby("Region", as_index=False)
        .agg(
            unemployment_rate=(UNEMPLOYMENT, "mean"),
            employed=(EMPLOYED, "mean"),
            labour_participation=(PARTICIPATION, "mean"),
        )
        .sort_values("unemployment_rate", ascending=False)
    )
