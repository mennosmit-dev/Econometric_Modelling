# 🚀 Kalman Filter & EM Algorithm for ARMA(1,1) and Dynamic Factor Models 📈

This project implements:

- 🔍 Kalman Filtering for univariate ARMA(1,1) models  
- 📊 Maximum Likelihood Estimation (MLE) of ARMA parameters  
- 🤖 EM Algorithm for Dynamic Factor Models (DFMs) on multivariate data  
- 📉 Visualizations of state estimates, residuals, and latent factors  

---

## ⚙️ Installation

HAI!
pip install numpy scipy matplotlib pandas
```

---

## 🛠️ Usage

### ARMA(1,1) with Kalman Filter

- 🧮 Estimate ARMA parameters via MLE  
- 📈 Filter and smooth states  
- 📊 Plot observations, state estimates, residuals  

Example:

```
result = minimize(negative_log_likelihood_LL_ARMA, initial_guess, args=(y,), bounds=bounds, method='SLSQP', tol=1e-8)
phi_ML, theta_ML, Q_ML, R_ML = result.x
print(f"phi_ML: {phi_ML}, theta_ML: {theta_ML}, Q_ML: {Q_ML}, R_ML: {R_ML}")
```

---

## 🔄 EM Algorithm for Dynamic Factor Model

- 🕵️‍♂️ Extract latent common factors from multivariate time series  
- 🔁 Iteratively perform E-step (Kalman smoothing) and M-step (parameter update)  
- 📈 Track log-likelihood convergence  

Example:

```
Lambda, Sigma, phi, sigma_eta2, f_smoothed, f_filt, logL_history, f_filt_mean = EM_DynamicFactorModel(y, max_iter=1000)
print("Estimated Lambda:", Lambda)
print("Estimated Sigma:", Sigma)
print("Estimated phi:", phi)
print("Estimated sigma_eta2:", sigma_eta2)
```

---

## 📊 Visualization

Plot Kalman filter results and factors:

```
plot_kalman_results(T, y, predicted_xi, xi, eta_prev)
plot_factors(y, f_filt, f_smoothed)
```

---

## 🗂️ File Overview

- `kalman_arma.py` — Kalman filter, log-likelihood, and parameter optimization  
- `dynamic_factor_em.py` — EM algorithm and Kalman smoother for factor estimation  
- `plotting.py` — Functions to visualize states, residuals, and factors  

---
