# Car Price Prediction

This project estimates the resale value of a used car from its original
showroom price, purchase year, kilometres driven, fuel type, transmission,
seller type and ownership history.

The Streamlit app includes:

- A guided resale price estimator
- Dataset summaries and visual exploration
- A model report with test metrics and cross-validation results
- A saved scikit-learn preprocessing and regression pipeline

## Project structure

```text
.
|-- app.py                 
|-- car_price_analysis.ipynb 
|-- data_utils.py     
|-- train_model.py     
|-- data/car_data.csv    
|-- models/               
|-- tests/test_project.py 
|-- requirements.txt
`-- README.md
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The dataset records prices in lakhs of Indian rupees. For example, `4.75`
means ₹4,75,000.

## Modeling approach

The two exact duplicate rows are removed before training. `Car_Age` is derived
from the purchase year, numeric features are standardized and categorical
features are one-hot encoded. Random Forest, Extra Trees and Gradient Boosting
models are compared using five-fold cross-validation on the training split.
The strongest average model is fitted and evaluated on a separate 20% test
split.

`Car_Name` is intentionally excluded. There are almost 100 car names in only
301 source rows, which makes it easy for a model to memorize individual
listings. The original showroom price provides a more stable representation
of the vehicle's market segment.

## Limitations

The source data is small and contains vehicles from 2003 to 2018. It does not
include location, service history, accident history, trim level or present
condition. Predictions should therefore be treated as educational estimates,
not formal valuations.
