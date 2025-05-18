# Imports
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, jarque_bera
from arch import arch_model
import matplotlib.pyplot as plt
from copulas.bivariate import Clayton

# Opening the data (ensure the path is correct)
df = pd.read_excel(
    r"C:\Users\Menno Smit\Downloads\Data_QRM.xlsx",
    sheet_name="Data",
    skiprows=1,
    usecols=[0, 1]
)

# Check the number of columns in the DataFrame (optional)
num_columns = len(df.columns)
print(f"The DataFrame has {num_columns} columns.")

# Select the first two columns and rename
if num_columns >= 2:
    df = df.iloc[:, :2]
    df.columns = ['Stock 1', 'Stock 2']
else:
    print("The DataFrame has less than two columns. Skipping column name assignment.")

stock1 = df.iloc[:, 0].to_numpy()
stock2 = df.iloc[:, 1].to_numpy()

print(stock1)
print(stock2)

"""METHOD 1: VARIANCE-COVARIANCE APPROACH"""

loss_returns = np.array(df)
num_obs = loss_returns.shape[0]

cov_matrix = np.cov(loss_returns, rowvar=False)

print(f"Mean Stock 1: {np.mean(loss_returns[:, 0]):.4g}")
print(f"Var Stock 1: {cov_matrix[0, 0]:.4g}")
print(f"Mean Stock 2: {np.mean(loss_returns[:, 1]):.4g}")
print(f"Var Stock 2: {cov_matrix[1, 1]:.4g}")
print(f"Covariance: {cov_matrix[0, 1]:.4g}")

mean_pf = np.mean(0.4 * loss_returns[:, 0] + 0.6 * loss_returns[:, 1])
var_pf = np.dot(np.dot(np.array([0.4, 0.6]), cov_matrix), np.array([0.4, 0.6]))
print(f"Mean Portfolio: {mean_pf:.4g}")
print(f"Var Portfolio: {var_pf:.4g}")
print(f"99% Quantile: {norm.ppf(0.99, mean_pf, np.sqrt(var_pf)):.4g}")

# Deriving first and second sample moment
data = np.vstack((stock1, stock2)).T
weights = np.array([0.4, 0.6])

portfolio_returns = np.dot(data, weights)
centered_portfolio_returns = portfolio_returns - np.mean(portfolio_returns)

mean_vector = np.mean(data, axis=0)
centered_data = data - mean_vector
cov_matrix = (1 / (len(stock1) - 1)) * np.dot(centered_data.T, centered_data)

mu1 = np.mean(stock1)
mu2 = np.mean(stock2)

# VaR and ES based on sample moments (variance-covariance) - one day 99% VaR
portfolio_mean = np.dot(weights, mean_vector)
portfolio_mean1 = np.mean(portfolio_returns)
portfolio_var = np.dot(np.dot(weights.T, cov_matrix), weights)
portfolio_var1 = (1 / (len(portfolio_returns) - 1)) * np.dot(centered_portfolio_returns.T, centered_portfolio_returns)
portfolio_sd = np.sqrt(portfolio_var)
portfolio_sd1 = np.sqrt(portfolio_var1)

z_score = norm.ppf(0.01)
VaR_99_varcov = -1 * (portfolio_mean + z_score * portfolio_sd)
VaR_99_varcov1 = -1 * (portfolio_mean + z_score * portfolio_sd1)

print(VaR_99_varcov, VaR_99_varcov1)

"""METHOD 2: HISTORICAL SIMULATION"""

pf_return = 0.4 * loss_returns[:, 0] + 0.6 * loss_returns[:, 1]
print(f"99% Quantile: {np.quantile(pf_return, 0.99):.4g}")

stock1_ordered = stock1[np.argsort(stock1)]
stock2_ordered = stock2[np.argsort(stock2)]

edf_values = np.zeros((len(stock1), len(stock2)))
for i in range(len(stock1)):
    for j in range(len(stock2)):
        count = np.sum((stock1 <= stock1[i]) & (stock2 <= stock2[j]))
        edf_values[i, j] = count / (len(stock1) * len(stock2))

closest_indices = np.unravel_index(np.argmin(np.abs(edf_values - 0.01)), edf_values.shape)
var_99_stock1 = stock1[closest_indices[0]]
var_99_stock2 = stock2[closest_indices[1]]

VaR99_HS = weights[0] * var_99_stock1 + weights[1] * var_99_stock2
print(VaR99_HS)

portfolio_returns = np.dot(weights, data.T)
portfolio_ordered = portfolio_returns[np.argsort(portfolio_returns)]

edf_values_portfolio = np.zeros(len(portfolio_returns))
for i in range(len(portfolio_returns)):
    count = np.sum(portfolio_returns <= portfolio_returns[i])
    edf_values_portfolio[i] = count / len(portfolio_returns)

closest_index = np.argmin(np.abs(edf_values_portfolio - 0.01))
VaR_99_portf_HS = -1 * portfolio_returns[closest_index]

print(VaR_99_portf_HS)

"""METHOD 3: NORMAL MIXTURE MODEL"""

# Jarque-Bera test on portfolio returns
n = len(portfolio_returns)
mu = np.mean(portfolio_returns)
sd = np.sqrt(1 / n * sum(centered_portfolio_returns**2))
b = 1 / n * sum((centered_portfolio_returns / sd) ** 3)  # skewness
k = 1 / n * sum((centered_portfolio_returns / sd) ** 4)  # kurtosis
T = n / 6 * (b**2 + (1 / 4) * (k - 3) ** 2)

print(mu, sd, b, k, T)
p_value = 1 - chi2.cdf(T, df=2)
print(p_value)

# EM Algorithm for Normal Mixture Model

def em_algorithm(portfolio_returns, iteration_limit=1000, tol=1e-6):
    """
    EM algorithm to estimate parameters for a two-component normal mixture.

    Returns:
        tuple: (lambda, mu1, sigma1, mu2, sigma2)
    """
    n = len(portfolio_returns)
    lamda = 0.3
    mu1 = np.mean(portfolio_returns) - 0.1
    sigma1 = max(np.std(portfolio_returns), 1e-5)
    mu2 = np.mean(portfolio_returns) + 0.1
    sigma2 = max(np.std(portfolio_returns), 1e-5)

    L_prev = -np.inf
    L_curr = 0
    iteration = 0

    while abs(L_curr - L_prev) > tol and iteration < iteration_limit:
        L_prev = L_curr
        iteration += 1

        # E-step
        denom = (
            lamda * norm.pdf(portfolio_returns, mu1, sigma1)
            + (1 - lamda) * norm.pdf(portfolio_returns, mu2, sigma2)
        )
        denom = np.where(denom == 0, 1e-10, denom)  # avoid div by zero
        p1 = (lamda * norm.pdf(portfolio_returns, mu1, sigma1)) / denom
        p2 = 1 - p1

        # M-step
        lamda = np.mean(p1)
        mu1 = np.sum(p1 * portfolio_returns) / (np.sum(p1) + 1e-8)
        mu2 = np.sum(p2 * portfolio_returns) / (np.sum(p2) + 1e-8)
        sigma1 = np.sqrt(np.sum(p1 * (portfolio_returns - mu1) ** 2) / (np.sum(p1) + 1e-8))
        sigma2 = np.sqrt(np.sum(p2 * (portfolio_returns - mu2) ** 2) / (np.sum(p2) + 1e-8))

        # Log-likelihood
        log_term1 = p1 * (np.log(lamda + 1e-12) + np.log(norm.pdf(portfolio_returns, mu1, sigma1) + 1e-12))
        log_term2 = p2 * (np.log(1 - lamda + 1e-12) + np.log(norm.pdf(portfolio_returns, mu2, sigma2) + 1e-12))

        # Replace -inf log with a large negative number
        log_term1 = np.where(np.isneginf(log_term1), -100, log_term1)
        log_term2 = np.where(np.isneginf(log_term2), -100, log_term2)

        L_curr = np.sum(log_term1 + log_term2)

    return lamda, mu1, sigma1, mu2, sigma2


lamda, mu1, sigma1, mu2, sigma2 = em_algorithm(portfolio_returns)
print(f"EM parameters: {lamda}, {mu1}, {sigma}")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, t as student_t, rankdata
from copulas.bivariate import ClaytonCopula
from arch import arch_model
import matplotlib.pyplot as plt


# =======================
# METHOD 4: EVT APPROACH
# =======================

def evt_hill_estimator(loss_returns, k_f=100, confidence_level=0.99):
    """
    Estimate VaR using EVT Hill estimator approach.

    Parameters:
        loss_returns (np.ndarray): 2D array with loss returns, shape (n, 2).
        k_f (int): Threshold parameter for Hill estimator.
        confidence_level (float): Confidence level for VaR.

    Returns:
        float: Estimated VaR.
    """
    mixture = -0.4 * loss_returns[:, 0] + 0.6 * loss_returns[:, 1]
    sorted_mixture = np.sort(mixture)
    num_obs = len(sorted_mixture)

    hill_est = np.zeros(300)

    for k in range(1, 301):
        h = 0
        for i in range(1, k + 1):
            numerator = sorted_mixture[-i]
            denominator = sorted_mixture[-k]
            # Avoid division by zero
            if denominator == 0:
                raise ZeroDivisionError("Zero encountered in denominator during Hill estimation.")
            h += np.log(numerator / denominator)

        if h > 0:
            hill_est[k - 1] = (h / k) ** -1

    tail_index = hill_est[k_f - 1]

    var_q4 = sorted_mixture[-k_f] * (k_f / (num_obs * (1 - confidence_level))) ** (1 / tail_index)

    return var_q4


# =======================
# METHOD 5: GARCH MODEL
# =======================

def garch_var(portfolio_returns, confidence_level=0.99):
    """
    Estimate VaR using a GARCH(1,1) model.

    Parameters:
        portfolio_returns (np.ndarray): 1D array of portfolio returns.
        confidence_level (float): Confidence level for VaR.

    Returns:
        float: Estimated VaR.
    """
    model = arch_model(portfolio_returns, vol='GARCH', p=1, q=1, mean='Constant')
    results = model.fit(disp='off')

    mu = results.params['mu']
    omega = results.params['omega']
    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']

    # Calculate unconditional volatility
    if (1 - alpha - beta) <= 0:
        raise ValueError("Unconditional variance not defined (1 - alpha - beta <= 0).")

    sigma_unc = omega / (1 - alpha - beta)

    sigma = np.zeros_like(portfolio_returns)
    sigma[0] = sigma_unc

    for i in range(1, len(portfolio_returns)):
        sigma[i] = omega + alpha * portfolio_returns[i - 1] ** 2 + beta * sigma[i - 1]

    z_scores = portfolio_returns / np.sqrt(sigma)

    var_99_garch = np.percentile(z_scores[1:], (1 - confidence_level) * 100) * np.sqrt(sigma[-1]) + mu

    # Since VaR is a loss, take the negative
    return -var_99_garch


# =======================
# METHOD 6: CLAYTON COPULA
# =======================

def clayton_copula_var(stock1, stock2, weights, num_simulations=1000, confidence_level=0.99):
    """
    Estimate VaR using Clayton copula fitted to the stock losses.

    Parameters:
        stock1 (np.ndarray): 1D array of losses for stock 1.
        stock2 (np.ndarray): 1D array of losses for stock 2.
        weights (list or np.ndarray): Portfolio weights [w1, w2].
        num_simulations (int): Number of copula samples to simulate.
        confidence_level (float): Confidence level for VaR.

    Returns:
        float: Estimated VaR.
    """
    n = len(stock1)

    rank1 = pd.Series(stock1).rank()
    rank2 = pd.Series(stock2).rank()

    concordant_pairs = 0
    discordant_pairs = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] < rank2[j]) or \
               (rank1[i] > rank1[j] and rank2[i] > rank2[j]):
                concordant_pairs += 1
            elif (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or \
                 (rank1[i] > rank1[j] and rank2[i] < rank2[j]):
                discordant_pairs += 1

    kendalls_tau = (concordant_pairs - discordant_pairs) / (0.5 * n * (n - 1))
    theta_est = 2 / (1 - kendalls_tau) - 2

    # Fit t-distributions
    fit1_params = student_t.fit(stock1)
    fit2_params = student_t.fit(stock2)

    df1, mu1, sigma1 = fit1_params[0], fit1_params[1], fit1_params[2]
    df2, mu2, sigma2 = fit2_params[0], fit2_params[1], fit2_params[2]

    clayton_copula = ClaytonCopula(theta=theta_est)

    var_sum = 0
    for _ in range(num_simulations):
        copula_samples = clayton_copula.sample(100000)
        x_copula = student_t.ppf(1 - copula_samples[:, 0], df1) * sigma1 + mu1
        y_copula = student_t.ppf(1 - copula_samples[:, 1], df2) * sigma2 + mu2

        portfolio_losses = weights[0] * x_copula + weights[1] * y_copula
        var_sum += np.quantile(portfolio_losses, confidence_level)

    va_r_copula = var_sum / num_simulations

    return va_r_copula


# =======================
# EXAMPLE USAGE
# =======================

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate example loss returns data
    num_obs = 2000
    loss_returns = np.random.randn(num_obs, 2)

    # EVT
    var_evt = evt_hill_estimator(loss_returns)
    print("VaR Q4 (EVT Hill estimator):", var_evt)

    # Portfolio returns example (sum of weighted losses, just for demo)
    weights = [0.4, 0.6]
    portfolio_returns = weights[0] * loss_returns[:, 0] + weights[1] * loss_returns[:, 1]

    # GARCH
    var_garch = garch_var(portfolio_returns)
    print("VaR 99% (GARCH):", var_garch)

    # Clayton copula
    var_copula = clayton_copula_var(loss_returns[:, 0], loss_returns[:, 1], weights)
    print("VaR 99% (Clayton copula):", var_copula)




                                        
