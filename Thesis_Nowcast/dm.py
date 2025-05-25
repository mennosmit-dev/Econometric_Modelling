"""
This module calculates the Diebold-Mariano (DM) test p-value for comparing forecast errors.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Load error data from CSV files
error_PCA_Bridge = pd.read_csv("/content/PCA_error_Bridge.csv", header=None)
error_SPC_Bridge = pd.read_csv("/content/SPC_error_Bridge.csv", header=None)
error_sigmoid_Bridge = pd.read_csv("/content/KPCsigmoid_error_Bridge.csv", header=None)
error_RBF_Bridge = pd.read_csv("/content/KPCrbf_error_Bridge.csv", header=None)

error_PCA_MIDAS_b = pd.read_csv("/content/PCA_error_MIDAS-b.csv", header=None)
error_SPC_MIDAS_b = pd.read_csv("/content/SPC_error_MIDAS-b.csv", header=None)
error_sigmoid_MIDAS_b = pd.read_csv("/content/KPCsigmoid_error_MIDAS-b.csv", header=None)
error_RBF_MIDAS_b = pd.read_csv("/content/KPCrbf_error_MIDAS-b.csv", header=None)

error_PCA_MIDAS_e = pd.read_csv("/content/PCA_error_MIDAS-e.csv", header=None)
error_SPC_MIDAS_e = pd.read_csv("/content/SPC_error_MIDAS-e.csv", header=None)
error_sigmoid_MIDAS_e = pd.read_csv("/content/KPCsigmoid_error_MIDAS-e.csv", header=None)
error_RBF_MIDAS_e = pd.read_csv("/content/KPCrbf_error_MIDAS-e.csv", header=None)


def dmtest(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> float:
    """
    Compute the Diebold-Mariano test statistic for comparing forecast accuracy.

    Parameters:
        e1 (np.ndarray): Forecast errors from model 1.
        e2 (np.ndarray): Forecast errors from model 2.
        h (int): Forecast horizon.

    Returns:
        float: DM test statistic.
    """
    d = e1 - e2  # Loss differential
    mean_d = np.mean(d)
    n = len(d)

    # Calculate the variance of loss differential with autocovariance adjustment
    gamma = np.array([np.cov(d[:-lag], d[lag:])[0, 1] if lag != 0 else np.var(d, ddof=1)
                      for lag in range(h)])
    var_d = gamma[0] + 2 * np.sum(gamma[1:])

    dm_stat = mean_d / np.sqrt(var_d / n)
    return dm_stat


# Select errors to compare (e.g., full period errors)
e1 = error_PCA_Bridge.values.flatten()
e2 = error_SPC_Bridge.values.flatten()

# Calculate DM test statistic and corresponding two-sided p-value
DM_statistic = dmtest(e1, e2, h=1)
p_value = 2 * (1 - norm.cdf(abs(DM_statistic)))

print(f"Diebold-Mariano test p-value: {p_value:.6f}")
