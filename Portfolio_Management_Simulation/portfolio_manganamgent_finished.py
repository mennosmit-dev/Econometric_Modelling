# Import libraries
# Calculates the expected return of the portfolio
# Calculates variance of the portfolio
# Calculates certainty equivalent
# Calculates GMV (Global Minimum Variance) weights

import numpy as np

# Initialize parameters
N = 10
beta = np.linspace(0.5, 1.5, N).reshape(-1, 1)
mu_f = 0.1
sigma_f = 0.2
error_covariance_matrix = np.diag([(0.3 - 0.1) ** 2] * N)
Sigma_R = np.dot(beta, beta.T) * (sigma_f ** 2) + error_covariance_matrix
risk_aversion = 2
ones = np.ones(Sigma_R.shape[0])

# Mean-Variance (MV) portfolio
w_mv = (
    np.linalg.inv(Sigma_R) @ beta * mu_f /
    (ones.T @ np.linalg.inv(Sigma_R) @ beta * mu_f)
)
mu_mv = np.dot(w_mv.T, beta * mu_f)
var_mv = np.dot(w_mv.T, np.dot(Sigma_R, w_mv))
sh_mv = mu_mv / np.sqrt(var_mv)
ce_mv = w_mv.T @ beta * mu_f - (1 / 2) * risk_aversion * w_mv.T @ Sigma_R @ w_mv

print("mu_mv:", mu_mv)
print("vol_mv:", np.sqrt(var_mv))
print("sh_mv:", sh_mv)
print("ce_mv:", ce_mv)

# Certainty Equivalent (CE) portfolio
w_ce = (
    (1 / risk_aversion) * np.linalg.inv(Sigma_R) @ beta * mu_f
    + (1 - (1 / risk_aversion) * ones.T @ np.linalg.inv(Sigma_R) @ beta * mu_f) * w_mv
)
mu_ce = np.dot(w_ce.T, beta * mu_f)
var_ce = np.dot(w_ce.T, np.dot(Sigma_R, w_ce))
sh_ce = mu_ce / np.sqrt(var_ce)
ce_ce = w_ce.T @ beta * mu_f - (1 / 2) * risk_aversion * w_ce.T @ Sigma_R @ w_ce

print("mu_ce:", mu_ce)
print("vol_ce:", np.sqrt(var_ce))
print("sh_ce:", sh_ce)
print("ce_ce:", ce_ce)

# Global Minimum Variance (GMV) portfolio
w_gmv = (
    np.linalg.inv(Sigma_R) @ ones /
    (ones.T @ np.linalg.inv(Sigma_R) @ ones)
)
mu_gmv = np.dot(w_gmv.T, beta * mu_f)
var_gmv = np.dot(w_gmv.T, np.dot(Sigma_R, w_gmv))
sh_gmv = mu_gmv / np.sqrt(var_gmv)
ce_gmv = w_gmv.T @ beta * mu_f - (1 / 2) * risk_aversion * w_gmv.T @ Sigma_R @ w_gmv

print("mu_gmv:", mu_gmv)
print("vol_gmv:", np.sqrt(var_gmv))
print("sh_gmv:", sh_gmv)
print("ce_gmv:", ce_gmv)

# Equal Weight (EW) portfolio
w_ew = np.full(N, 1 / N).reshape(-1, 1)
mu_ew = np.dot(w_ew.T, beta * mu_f)
var_ew = np.dot(w_ew.T, np.dot(Sigma_R, w_ew))
sh_ew = mu_ew / np.sqrt(var_ew)
ce_ew = w_ew.T @ beta * mu_f - (1 / 2) * risk_aversion * w_ew.T @ Sigma_R @ w_ew

print("mu_ew:", mu_ew)
print("vol_ew:", np.sqrt(var_ew))
print("sh_ew:", sh_ew)
print("ce_ew:", ce_ew)
