# Mixed-Frequency Forecasting with Bridge and MIDAS Models

Implementation of mixed-frequency forecasting methods using:

- Bridge regression
- MIDAS (Mixed Data Sampling) with Beta polynomial weights (MIDAS-b)
- MIDAS with Exponential Almon weights (MIDAS-e)

The project focuses on forecasting quarterly variables using higher-frequency
monthly indicators, including hyperparameter tuning, prediction, and
out-of-sample evaluation.

---

## 🧠 Overview

Key components of the workflow:

- Alignment of monthly and quarterly time series
- Construction of lag-augmented factor datasets
- Hyperparameter grid search across mixed-frequency models
- Out-of-sample forecasting and Mean Squared Prediction Error (MSPE) evaluation
- Automated storage of predictions and performance metrics

Models implemented:

- **Bridge Regression**
- **MIDAS-b** (Beta polynomial weighting)
- **MIDAS-e** (Exponential Almon weighting)

---

## ⚙️ Methodology

### Data Preparation
- Synchronization of mixed-frequency datasets
- Lag construction for monthly indicators
- Handling of missing monthly observations within quarterly periods

### Model Estimation
Hyperparameters tuned via grid search:

- Bridge regression: γ, η, polynomial order p
- MIDAS-b: Beta polynomial parameters
- MIDAS-e: Exponential Almon parameters

### Evaluation
- Multi-horizon forecasting
- Mean Squared Prediction Error (MSPE)
- Storage of optimal parameters and prediction outputs

---

## 🗂️ Outputs

The script generates:

- Prediction files per model and horizon
- MSPE evaluation metrics
- Optimal hyperparameter selections

---

## 🔧 Tech Stack

Python • NumPy • Pandas • SciPy  
Time-Series Econometrics • Mixed-Frequency Modeling • Forecast Evaluation

---

## 📌 Context

This project demonstrates mixed-frequency econometric forecasting techniques
commonly used in macroeconomic nowcasting and financial time-series modeling.
It complements my broader work in:

- dynamic factor models
- state-space estimation
- portfolio and trading models
- reinforcement learning for financial decision-making
