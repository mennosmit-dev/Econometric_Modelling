# Portfolio Risk Analysis & Value-at-Risk Estimation

Python implementation of portfolio risk analysis and Value-at-Risk (VaR) estimation 
using both classical and advanced statistical modeling techniques.

The project compares multiple risk modeling approaches, ranging from 
variance–covariance methods to Extreme Value Theory and copula-based simulations.

---

## 🧠 Overview

Implemented VaR methodologies:

- Variance–Covariance (Parametric VaR)
- Historical Simulation
- Normal Mixture Model (EM Algorithm)
- Extreme Value Theory (Hill Estimator)
- GARCH(1,1) Volatility Modeling
- Clayton Copula Simulation

Core features:

- Portfolio loss modeling from financial time series
- Tail-risk estimation
- Dependency structure modeling via copulas
- Visualization of risk metrics

---

## ⚙️ Workflow

1. Load financial loss/return data from Excel.
2. Compute portfolio statistics (mean, variance, quantiles).
3. Estimate VaR using multiple statistical frameworks.
4. Compare tail-risk estimates across methodologies.

---

## 📂 Project Structure

- `portfolio_risk_analysis.py`  
  Main script implementing data loading, statistical modeling, 
  and VaR estimation workflows.

---

## 🚀 Usage

Update the dataset path inside the script:

```python
df = pd.read_excel(
    r"C:\path\to\Data_QRM.xlsx",
    sheet_name="Data",
    skiprows=1,
```
Run:
```
python portfolio_risk_analysis.py
```
🔧 Tech Stack

Python • NumPy • Pandas • SciPy • ARCH • Matplotlib • Copulas

## 📐 Methodology Summary
### Variance–Covariance

Parametric VaR assuming normally distributed portfolio returns.

### Historical Simulation

Non-parametric VaR based on empirical return quantiles.

### Normal Mixture Model

Two-component Gaussian mixture fitted using EM to capture regime behavior.

### Extreme Value Theory (EVT)

Hill estimator applied to model tail risk and extreme losses.

### GARCH(1,1)

Time-varying volatility estimation for conditional risk forecasting.

### Clayton Copula

Dependence modeling via copula simulation for multivariate tail behavior.

## 📌 Context

This project complements my broader work in quantitative finance,
portfolio optimization, and reinforcement learning for asset allocation,
with a focus on advanced risk modeling and tail-risk analysis.

    usecols=[0, 1]
)
