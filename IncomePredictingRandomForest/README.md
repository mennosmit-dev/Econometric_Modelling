# Income Level Prediction with Random Forests

This project implements a structured machine learning pipeline for predicting 
income levels from tabular data using Random Forest models.

The workflow emphasizes robust evaluation through nested cross-validation, 
feature selection, and hyperparameter optimization.

---

## 🧠 Overview

Key components of the pipeline:

- Data preprocessing and feature handling
- Nested cross-validation for unbiased model selection
- Feature importance–based dimensionality reduction
- Random Forest model training and evaluation
- Prediction generation on unseen test data

The project focuses on reproducible experimentation and interpretable 
tree-based modeling for structured datasets.

---

## 📂 Project Structure

- `Predicting_Income_Level.ipynb` / `.py`  
  Main workflow:
  - Data loading and preprocessing
  - Nested cross-validation
  - Hyperparameter tuning
  - Feature selection
  - Final model training and prediction

- `train.csv` — Training dataset  
- `test.csv` — Test dataset  
- `codebook.csv` — Feature descriptions

### Generated Outputs

- `oos_accuracies.csv` — Cross-validation results  
- `mean_accuracy_hyperparameters.png` — Hyperparameter performance visualization  
- `features_prediction.png` — Feature importance plot  
- `predictions.txt` — Final model predictions

---

## ⚙️ Requirements

Python 3.7+

Libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm
```

## 🚀 Usage
1. Place datasets (train.csv, test.csv, codebook.csv) in the project directory.
2. Run:
```bash
python Predicting_Income_Level.py
```

The script will:

perform nested cross-validation

train the final Random Forest model

generate predictions and evaluation plots

## 📊 Methodology Notes

Nested cross-validation prevents information leakage during tuning.

Random Forest models are used without feature scaling.

Feature selection is based on importance thresholds derived from CV results.

## 📌 Context

This project complements my broader work in applied machine learning and
econometric modeling, focusing on interpretable tree-based methods and
robust model evaluation pipelines.
