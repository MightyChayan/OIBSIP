# Iris Flower Classification

This project uses machine learning to classify Iris flowers as Setosa,
Versicolor, or Virginica using four flower measurements.

## Results

| Metric | Score |
|---|---:|
| Selected model | K-Nearest Neighbors |
| Tuned cross-validation accuracy | 96.7% |
| Held-out test accuracy | 93.3% |
| Macro precision | 94.4% |
| Macro F1-score | 93.3% |

## What I Did

- Loaded and checked the Iris dataset
- Studied the data using graphs and summary statistics
- Compared K-Nearest Neighbors, Decision Tree, and Random Forest
- Selected and tuned the model with the best cross-validation score
- Tested the final model on data that was not used during training
- Created a simple Streamlit app to make predictions
- Saved the results, graphs, test predictions, and trained model

## Dataset

The dataset contains 150 observations, with 50 examples from each species:

| Feature | Description |
|---|---|
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm` | Sepal width in centimeters |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm` | Petal width in centimeters |
| `Species` | Target flower species |

`Id` is excluded from training because it is an identifier, not a biological
measurement.

## Project Structure

```text
.
|-- Iris.csv
|-- app.py
|-- requirements.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml
|-- notebooks/
|   `-- Iris_Flower_Classification.ipynb
|-- src/
|   |-- __init__.py
|   |-- predict.py
|   `-- train.py
`-- outputs/                 
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python -m src.train
```

The command creates:

- `outputs/iris_model.joblib`
- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/classification_report.csv`
- `outputs/test_predictions.csv`
- EDA and evaluation charts in `outputs/`

## Run the Application

```powershell
streamlit run app.py
```

The application includes:

- Flower measurement sliders
- Species prediction with confidence
- Model performance metrics
- Model comparison table
- Confusion matrix
- A short explanation of the training process

## Steps Followed

1. Checked the columns, missing values, duplicate rows, and class counts.
2. Removed the `Id` column because it does not describe the flower.
3. Used 80% of the data for training and 20% for testing.
4. Compared three algorithms using 5-fold cross-validation.
5. Tuned the best-performing algorithm.
6. Checked the final accuracy, precision, recall, F1-score, and confusion
   matrix.

I used `StandardScaler` with KNN because KNN calculates distances between data
points and can be affected by differences in feature scale.

## Conclusion

K-Nearest Neighbors gave the best result for this dataset. The final model
correctly classified 28 out of 30 flowers in the test set. Petal length and
petal width were especially useful for separating the three species.
