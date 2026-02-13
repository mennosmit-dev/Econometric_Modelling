# Kalman Filtering & EM Estimation for ARMA and Dynamic Factor Models

Implementation of state-space modeling techniques including:

- Kalman filtering and smoothing for ARMA(1,1)
- Maximum Likelihood Estimation (MLE) of state-space parameters
- Expectation-Maximization (EM) algorithm for Dynamic Factor Models (DFMs)
- Latent factor extraction from multivariate time series

This project focuses on likelihood-based estimation, latent state inference, and 
time-series modeling using state-space representations.

---

## 🧠 Overview

### 1️⃣ ARMA(1,1) via State-Space Representation

- Reformulated ARMA(1,1) as a state-space model
- Implemented Kalman filter for likelihood evaluation
- Estimated parameters via numerical optimization (MLE)
- Applied smoothing to recover latent states
- Analyzed residual behavior and model fit

Core outputs:

- Estimated parameters (φ, θ, Q, R)
- Filtered and smoothed states
- Log-likelihood evaluation
- Diagnostic visualizations

---

### 2️⃣ Dynamic Factor Model (EM Algorithm)

Implemented a one-factor Dynamic Factor Model for multivariate time series:

- E-step: Kalman smoothing to estimate latent factors
- M-step: Closed-form parameter updates
- Log-likelihood tracking for convergence monitoring

Estimated components:

- Factor loadings (Λ)
- Idiosyncratic variance (Σ)
- Factor AR coefficient (φ)
- Factor innovation variance (σ²η)

---

## ⚙️ Implementation Structure

- `kalman_arma.py`  
  State-space representation, Kalman filter, log-likelihood computation, and MLE optimization.

- `dynamic_factor_em.py`  
  EM algorithm for latent factor estimation using Kalman smoothing.

- `plotting.py`  
  Visualization utilities for states, residuals, and extracted factors.

---

## 🔬 Methods Applied

- State-space modeling
- Kalman filtering & smoothing
- Maximum likelihood estimation
- EM algorithm
- Latent factor extraction
- Numerical optimization (SLSQP)

---

## 🔧 Tech Stack

Python • NumPy • SciPy • Pandas • Matplotlib  
Time-Series Econometrics • State-Space Models • Latent Variable Modeling

---

## 📌 Context

This project complements my broader work in:

- econometric forecasting
- mixed-frequency modeling
- reinforcement learning for portfolio management
- systematic trading strategies

It demonstrates advanced likelihood-based time-series modeling and 
latent state estimation techniques frequently used in macroeconometrics 
and quantitative finance.
