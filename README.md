# OIBSIP

This repository contains my work for the Oasis Infobyte internship tasks.

The projects are kept in separate folders so each task can be opened and run
individually.

## Repository Structure

```text
OIBSIP/
|-- ChayanMaity_Task1/
|-- ChayanMaity_Task2/
|-- ChayanMaity_Task3/
`-- README.md
```

## Tasks

### Task 1: Iris Flower Classification

This task is a machine learning project that classifies Iris flowers into three
species:

- Setosa
- Versicolor
- Virginica

The project uses the Iris dataset and compares different machine learning
models. The final model is saved and a simple Streamlit app is provided to make
predictions.

Main files:

- `Iris.csv` - dataset
- `app.py` - Streamlit application
- `src/train.py` - model training script
- `src/predict.py` - prediction helper
- `outputs/` - saved model, charts, and result files
- `notebooks/` - Jupyter notebook for analysis

To run Task 1:

```powershell
cd ChayanMaity_Task1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

To train the model again:

```powershell
python -m src.train
```

### Task 2: Unemployment Analysis with Python

This task is a data analysis project that studies unemployment in India from
May 2019 to June 2020.

The project uses a CMIE unemployment dataset and shows how unemployment changed
across different regions, areas, and time periods. It also compares the
pre-COVID period with the COVID period.

The Streamlit app includes:

- Region, area, and date filters
- Monthly unemployment trend
- Pre-COVID and COVID-period comparison
- State and union territory ranking
- Rural and urban comparison
- Data preview and CSV export
- Option to upload a compatible dataset

Main files:

- `data/unemployment_india.csv` - unemployment dataset
- `app.py` - Streamlit application
- `analysis.py` - data cleaning and analysis code
- `tests/test_analysis.py` - unit tests
- `requirements.txt` - required Python libraries

To run Task 2:

```powershell
cd ChayanMaity_Task2
python -m pip install -r requirements.txt
streamlit run app.py
```

To run the tests:

```powershell
python -m unittest discover -s tests -v
```

### Task 3: Car Price Prediction

This task is a machine learning project that predicts the resale price of a
used car.

The prediction is based on details such as original showroom price, purchase
year, kilometres driven, fuel type, seller type, transmission type, and
ownership history.

The Streamlit app includes:

- Used car resale price estimator
- Dataset summary
- Visual analysis of car price patterns
- Model performance report
- Saved machine learning pipeline

Main files:

- `data/car_data.csv` - car dataset
- `app.py` - Streamlit application
- `train_model.py` - model training script
- `data_utils.py` - helper functions for data handling
- `car_price_analysis.ipynb` - notebook for analysis
- `models/` - saved model and metrics
- `tests/test_project.py` - project tests
- `requirements.txt` - required Python libraries

To run Task 3:

```powershell
cd ChayanMaity_Task3
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

To run the tests:

```powershell
python -m pytest tests -v
```

## Requirements

The projects use Python libraries for data analysis, visualization, machine
learning, testing, and app development.

Some of the main libraries used are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit
- jupyter
- plotly
- scipy
- pytest

The exact versions are listed in each task folder:

```text
ChayanMaity_Task1/requirements.txt
ChayanMaity_Task2/requirements.txt
ChayanMaity_Task3/requirements.txt
```

## About

This repository was created as part of the Oasis Infobyte internship program.
Each task shows a different practical project and includes the required files,
code, and outputs.
