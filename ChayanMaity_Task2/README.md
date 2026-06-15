# Unemployment Analysis with Python

This is my Task 2 project for the Oasis Infobyte Data Science internship. The
project studies unemployment in India between May 2019 and June 2020 using the
sample CMIE dataset provided with the task.

I used Streamlit for the interface, Pandas for cleaning and aggregation, Plotly
for the charts, and SciPy for the pre-COVID and COVID-period comparison.

## Features

- Filters for region, rural/urban area, and date range
- Monthly change in the unemployment rate
- Comparison of pre-COVID and COVID-period unemployment
- Ranking of states and union territories
- Rural versus urban comparison
- Distribution and labour-participation relationship charts
- Filtered data preview and CSV export
- Support for uploading a compatible replacement dataset

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
.
|-- app.py
|-- analysis.py
|-- data/
|   `-- unemployment_india.csv
|-- tests/
|   `-- test_analysis.py
|-- requirements.txt
`-- README.md
```

## Dataset

The source file contains 768 rows. During cleaning, 28 completely blank rows
are removed, leaving 740 usable observations for 28 states and union
territories. Column names are trimmed, dates are converted to datetime values,
and the numeric columns are checked before the analysis is displayed.
