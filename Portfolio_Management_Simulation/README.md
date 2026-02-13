# Portfolio Optimization & Performance Metrics

Python implementation of factor-based portfolio optimization using 
mean–variance principles and classical asset pricing theory.

The script evaluates multiple portfolio construction strategies and compares 
their risk-return characteristics using analytical optimization and matrix-based calculations.

---

## 🧠 Overview

Implemented portfolio strategies:

- Mean–Variance (MV) Portfolio
- Certainty Equivalent (CE) Portfolio
- Global Minimum Variance (GMV) Portfolio
- Equal Weight (EW) Portfolio

For each strategy, the following metrics are computed:

- Expected return (`mu`)
- Volatility (`vol`)
- Sharpe ratio (`sh`)
- Certainty equivalent (`ce`)

The implementation uses a single-factor model to construct the covariance matrix 
and incorporates idiosyncratic risk components.

---

## 📐 Model Setup

Key elements:

- Factor model with loadings (`beta`)
- Expected factor return (`mu_f`)
- Factor volatility (`sigma_f`)
- Idiosyncratic error covariance
- Investor risk aversion parameter

The covariance matrix combines systematic factor risk with asset-specific variance.

---

## ⚙️ Implementation Details

Workflow:

1. Construct expected returns from the factor model.
2. Build covariance matrix from factor exposure and residual risk.
3. Compute analytical portfolio weights for each strategy.
4. Evaluate performance metrics and print results.

---

## 🚀 Usage

```bash
python portfolio_optimization.py
```
Requires Python 3 and NumPy.

## 🔧 Tech Stack

Python • NumPy • Quantitative Finance • Portfolio Theory

## 📌 Context

This project complements my broader work in quantitative modeling,
reinforcement learning for portfolio management, and econometric forecasting,
demonstrating classical optimization methods in asset allocation.


