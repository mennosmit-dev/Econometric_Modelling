# Stepwise Penalized Regression Models for GDP Forecasting

This project implements a multi-step econometric forecasting framework for GDP prediction 
using penalized regression techniques and forecast evaluation metrics.

The pipeline combines feature selection, regularized regression, custom egalitarian weighting, 
and statistical comparison via the Diebold–Mariano test to assess predictive performance.

---

## 🧠 Overview

Key components of the workflow:

- Stepwise modeling pipeline
- Penalized regression methods:
  - Lasso
  - Ridge
  - ElasticNet
- Egalitarian coefficient adjustment
- RMSE-based model weighting
- K-Fold cross-validation for hyperparameter tuning
- Diebold–Mariano tests for forecast comparison

The objective is to evaluate whether structured regularization and egalitarian weighting 
improve multi-step GDP forecasting accuracy.

---

## 📂 Methodology

### 🔹 Step 1 — Feature Selection

- Lasso regression used to identify informative predictors.
- Cross-validation applied to determine optimal regularization strength.

### 🔹 Step 2 — Penalized Forecast Models

Models trained on selected features:

- Standard penalized regressions
- Egalitarian-adjusted variants
- RMSE-weighted combinations

### 🔹 Step 3 — Model Evaluation

- Root Mean Squared Error (RMSE) used for performance comparison.
- Diebold–Mariano tests applied to assess statistical differences 
  in forecast accuracy between model variants.

---

## ⚙️ Requirements

Python 3.7+

Core libraries:

- numpy
- pandas
- scikit-learn
- matplotlib

Optional:

- dm_test (Diebold–Mariano implementation)
- Jupyter Notebook or Google Colab

---

## 🚀 Usage

1. Provide a dataset (e.g., `H1_gdp.csv`).
2. Run the script in a Python environment.
3. Outputs include:
   - Optimal regularization parameters
   - Selected features
   - RMSE comparison across models
   - Diebold–Mariano statistical test results

---

## 🔧 Key Functions

- `cross_validate_lasso()` — Lasso hyperparameter tuning  
- `cross_validate_ridge()` — Ridge hyperparameter tuning  
- `cross_validate_en()` — ElasticNet hyperparameter tuning  
- `egalitarian_transform()` — coefficient balancing adjustment  

---

## 📚 References

- Friedman, Hastie, Tibshirani (2010) — Regularization Paths for GLMs  
- Diebold & Mariano (1995) — Comparing Predictive Accuracy

---

## 📌 Context

This project forms part of a broader econometric modeling workflow 
focusing on structured forecast combination, regularization methods, 
and quantitative macroeconomic modeling.
