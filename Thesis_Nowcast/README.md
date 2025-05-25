**For all the details feel free to have a look at the pdf in the map.**

# 📊 Forecasting with Bridge and MIDAS Models 🚀

This repository contains Python code implementing forecasting methods using **Bridge regression** and **MIDAS** (Mixed Data Sampling) approaches. The code performs hyperparameter tuning, prediction, and error calculation across different models, storing results and saving them to CSV files for further analysis.

---

## 🔍 Overview

The script follows these main steps:

- 🗂️ Load and prepare monthly and quarterly time series data.
- ⚙️ Perform hyperparameter grid search to identify optimal parameters for three regression models:
  - 🌉 Bridge regression
  - 🧩 MIDAS-b (Beta polynomial weights)
  - 🔄 MIDAS-e (Exponential Almon polynomial weights)
- 📈 Use lag-augmented factor data to train models and make predictions.
- 📉 Calculate and store Mean Squared Prediction Errors (MSPE).
- 🧹 Impute missing data in monthly datasets for unknown months in quarters.
- 💾 Save all results including errors and predictions to CSV files.

---

## 🛠️ Code Structure

### Data Preparation
- 🔄 Align monthly and quarterly data series.
- ⏳ Compute lagged factors and construct training and testing datasets for each forecast horizon.

### Model Training and Hyperparameter Tuning
- 🌉 **Bridge regression**: Tune `gamma`, `eta`, and polynomial order `p`.
- 🧩 **MIDAS-b**: Tune Beta polynomial hyperparameters.
- 🔄 **MIDAS-e**: Tune Exponential Almon polynomial hyperparameters.

### Prediction and Error Calculation
- 🎯 Make forecasts for each horizon and store errors.
- 📊 Calculate MSPE for each model and horizon.
- 🗃️ Store predictions and errors in CSV files for review.

---

## 🚀 Usage

1. ✅ Ensure required packages are installed (e.g., `numpy`, `pandas`, `scipy`).
2. 📥 Load your monthly and quarterly data into the expected formats.
3. ⚙️ Adjust hyperparameter grids if needed.
4. ▶️ Run the script to perform hyperparameter tuning and forecasting.
5. 📂 Check output CSV files for errors and predictions.

---

## 💡 Code Snippet Example

```
# Example: Hyperparameter grid search for Bridge regression
for gamma in gamma_values:
    for eta in eta_values:
        for p in p_values:
            # Train model, predict, and compute MSPE
            ...
```

---

## 📈 Results

Results include:

- 🏆 MSPE values for each model and horizon.
- 🔧 Optimal hyperparameters selected by lowest MSPE.
- 📊 Predictions stored for evaluation.

---
