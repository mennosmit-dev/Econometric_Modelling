import pandas as pd
import numpy as np
import scipy
import statistics
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Import data
data = pd.read_excel('/Users/architjavkhedkar/Downloads/assignment/data.xlsx')

# Display data (optional, typically not in final scripts)
print(data)

# Filter out the 'DATE' column
data_filt = data.drop('DATE', axis=1)
print(data_filt)

# Training Data: first 189 data points
data_train = data_filt.iloc[:189]

# Data for question 1, RPI column
data_q1 = data_train['RPI']
print(data_q1)

# Selecting y2 as numpy array of values for modeling
y2 = data_q1.values
print(y2)

# --- Q1.a ---

# Initialize parameters
T = 189
p1 = 0.8
mu1 = y2.mean()
mu2 = y2.mean()
sigma1 = 0.5 * y2.std()
sigma2 = 1.5 * y2.std()

# Store initial parameter vector
parameter_vectors1 = [p1, mu1, mu2, sigma1, sigma2]


def log_likelihood(parameters, y):
    """
    Calculate the negative log-likelihood for a finite mixture model of two normal distributions.

    Parameters
    ----------
    parameters : list or np.ndarray
        List containing [p1, mu1, mu2, sigma1, sigma2]
        where p1 is the mixing probability of regime 1.
    y : np.ndarray
        Observed data points.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    p1 = parameters[0]
    p2 = 1 - p1
    mu1 = parameters[1]
    mu2 = parameters[2]
    sigma1 = parameters[3]
    sigma2 = parameters[4]

    # Mixture likelihood for each observation
    likelihood = p1 * norm.pdf(y, mu1, sigma1) + p2 * norm.pdf(y, mu2, sigma2)

    # Log likelihood per observation
    log_likelihood_vals = np.log(likelihood)

    # Sum over all observations
    ll_value = np.sum(log_likelihood_vals)

    # Return negative log likelihood for minimization
    return -ll_value


def optimise_and_print(parameter_vector, ll_func):
    """
    Optimize the log-likelihood function and print the estimated parameters.

    Parameters
    ----------
    parameter_vector : list
        Initial guess for parameters.
    ll_func : function
        Log-likelihood function to minimize.

    Returns
    -------
    tuple
        Estimated parameters (p1, p2, mu1, mu2, sigma1, sigma2).
    """
    bounds = [(0, 1), (-100, 100), (-100, 100), (0, 100), (0, 100)]
    res = minimize(
        ll_func,
        parameter_vector,
        args=(y2,),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-6
    )

    estimated_params = res.x

    p1_est = estimated_params[0]
    p2_est = 1 - p1_est
    mu1_est = estimated_params[1]
    mu2_est = estimated_params[2]
    sigma1_est = estimated_params[3]
    sigma2_est = estimated_params[4]

    print(f"p1_est = {p1_est}")
    print(f"p2_est = {p2_est}")
    print(f"mu1_est = {mu1_est}")
    print(f"mu2_est = {mu2_est}")
    print(f"sigma1_est = {sigma1_est}")
    print(f"sigma2_est = {sigma2_est}")

    optimized_log_likelihood = -res.fun
    print(f"Log Likelihood value: {optimized_log_likelihood}")

    return p1_est, p2_est, mu1_est, mu2_est, sigma1_est, sigma2_est


print('Part A Estimates and Log Likelihood Values')
optimise_and_print(parameter_vectors1, log_likelihood)

# Probability of being in state 2 if the value of the series y2,t equals 0
y_obs = 0
p1_est, p2_est, mu1_est, mu2_est, sigma1_est, sigma2_est = optimise_and_print(parameter_vectors1, log_likelihood)

probability_recession = (
    p2_est * norm.pdf(y_obs, mu2_est, sigma2_est)
    / (p2_est * norm.pdf(y_obs, mu2_est, sigma2_est) + p1_est * norm.pdf(y_obs, mu1_est, sigma1_est))
)
print(f"Probability of being in Recession if y2,t = 0: {probability_recession}")

# --- Q1.b ---

# Given transition probabilities
p11 = 0.8
p22 = 0.8

# Initialize parameter vector for Hamilton filter
initial_vectors = [p11, p22, mu1, mu2, sigma1, sigma2]


def hamilton_filter(p11, p22, mu1, mu2, sigma1, sigma2, y):
    """
    Run Hamilton filter for a two-state Markov switching model.

    Parameters
    ----------
    p11 : float
        Transition probability of staying in state 1.
    p22 : float
        Transition probability of staying in state 2.
    mu1, mu2 : float
        Means of states 1 and 2.
    sigma1, sigma2 : float
        Standard deviations of states 1 and 2.
    y : np.ndarray
        Observed data.

    Returns
    -------
    tuple
        filtered_xi: filtered state probabilities (2 x T array)
        predicted_xi: predicted state probabilities (2 x T array)
    """
    P = np.array([[p11, 1 - p22],
                  [1 - p11, p22]])

    T = len(y)
    predicted_xi = np.zeros((2, T))
    filtered_xi = np.zeros((2, T))
    likelihood = np.zeros((2, T))

    # Initial state prediction (start fully in state 1)
    predicted_xi[:, 0] = [1, 0]

    for t in range(T):
        likelihood[:, t] = [norm.pdf(y[t], mu1, sigma1), norm.pdf(y[t], mu2, sigma2)]
        numerator = predicted_xi[:, t] * likelihood[:, t]
        denominator = np.dot(predicted_xi[:, t], likelihood[:, t])
        filtered_xi[:, t] = numerator / denominator

        if t < T - 1:
            predicted_xi[:, t + 1] = P.dot(filtered_xi[:, t])

    return filtered_xi, predicted_xi


def log_likelihood_hamilton(parameters, y):
    """
    Calculate the negative log-likelihood using the Hamilton filter.

    Parameters
    ----------
    parameters : list or np.ndarray
        [p11, p22, mu1, mu2, sigma1, sigma2]
    y : np.ndarray
        Observed data.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    p11, p22, mu1, mu2, sigma1, sigma2 = parameters

    _, predicted_xi = hamilton_filter(p11, p22, mu1, mu2, sigma1, sigma2, y)

    ll = np.log(predicted_xi[0, :] * norm.pdf(y, mu1, sigma1)
                + predicted_xi[1, :] * norm.pdf(y, mu2, sigma2))
    total_ll = -np.sum(ll)

    return total_ll


def optimise_and_print_hamilton(parameter_vector, ll_func):
    """
    Optimize Hamilton filter log-likelihood and print estimated parameters.

    Parameters
    ----------
    parameter_vector : list
        Initial guess for parameters.
    ll_func : function
        Log-likelihood function.

    Returns
    -------
    tuple
        Estimated parameters, long-term means, and log-likelihood.
    """
    bounds = [(0, 1), (0, 1), (-100, 100), (-100, 100), (0, 100), (0, 100)]

    res = scipy.optimize.minimize(
        ll_func,
        parameter_vector,
        args=(y2,),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-6
    )

    est = res.x
    p11_est, p22_est = est[0], est[1]
    mu1_est, mu2_est = est[2], est[3]
    sigma1_est, sigma2_est = est[4], est[5]

    print(f"p11_est = {p11_est}")
    print(f"p22_est = {p22_est}")
    print(f"mu1_est = {mu1_est}")
    print(f"mu2_est = {mu2_est}")
    print(f"sigma1_est = {sigma1_est}")
    print(f"sigma2_est = {sigma2_est}")

    long_term_mean1 = (1 - p22_est) / (2 - p11_est - p22_est)
    long_term_mean2 = (1 - p11_est) / (2 - p11_est - p22_est)

    print(f"Long Term Mean 1 = {long_term_mean1}")
    print(f"Long Term Mean 2 = {long_term_mean2}")

    optimized_log_likelihood = -res.fun
    print(f"Log Likelihood value: {optimized_log_likelihood}")

    return (p11_est, p22_est, mu1_est, mu2_est,
            sigma1_est, sigma2_est, long_term_mean1,
            long_term_mean2, optimized_log_likelihood)


import numpy as np
from scipy.stats import norm
import scipy.optimize


def hamilton_filter_state2(p11, p22, mu1, mu2, sigma1, sigma2, y):
    """
    Hamilton filter assuming initial state is state 2.

    Parameters
    ----------
    p11 : float
        Transition probability of staying in state 1.
    p22 : float
        Transition probability of staying in state 2.
    mu1 : float
        Mean of state 1.
    mu2 : float
        Mean of state 2.
    sigma1 : float
        Standard deviation of state 1.
    sigma2 : float
        Standard deviation of state 2.
    y : array-like
        Observations.

    Returns
    -------
    filtered_xi : ndarray, shape (2, len(y))
        Filtered probabilities of each state at each time.
    predicted_xi : ndarray, shape (2, len(y))
        Predicted probabilities of each state at each time.
    """
    P = np.array([[p11, 1 - p22],
                  [1 - p11, p22]])

    T = len(y)
    predicted_xi = np.zeros((2, T))
    filtered_xi = np.zeros((2, T))
    likelihood = np.zeros((2, T))

    # Initial state: fully in state 2
    predicted_xi[:, 0] = [0, 1]

    for t in range(T):
        likelihood[:, t] = [
            norm.pdf(y[t], mu1, sigma1),
            norm.pdf(y[t], mu2, sigma2)
        ]
        numerator = predicted_xi[:, t] * likelihood[:, t]
        denominator = np.dot(predicted_xi[:, t], likelihood[:, t])
        filtered_xi[:, t] = numerator / denominator

        if t < T - 1:
            predicted_xi[:, t + 1] = P.dot(filtered_xi[:, t])

    return filtered_xi, predicted_xi


def log_likelihood_hamilton_state2(parameters, y):
    """
    Compute negative log-likelihood for Hamilton filter assuming initial state 2.

    Parameters
    ----------
    parameters : array-like
        Model parameters: [p11, p22, mu1, mu2, sigma1, sigma2]
    y : array-like
        Observations.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    p11, p22, mu1, mu2, sigma1, sigma2 = parameters

    _, predicted_xi = hamilton_filter_state2(p11, p22, mu1, mu2, sigma1, sigma2, y)

    ll = np.log(
        predicted_xi[0, :] * norm.pdf(y, mu1, sigma1)
        + predicted_xi[1, :] * norm.pdf(y, mu2, sigma2)
    )
    total_ll = -np.sum(ll)

    return total_ll


def optimise_and_print_hamilton_state2(parameter_vector, ll_func, y):
    """
    Optimize Hamilton filter parameters assuming initial state 2 and print results.

    Parameters
    ----------
    parameter_vector : array-like
        Initial guess for parameters.
    ll_func : callable
        Log-likelihood function.
    y : array-like
        Observations.

    Returns
    -------
    tuple
        Estimated parameters and log-likelihood.
    """
    bounds = [(0, 1), (0, 1), (-100, 100), (-100, 100), (0, 100), (0, 100)]

    res = scipy.optimize.minimize(
        ll_func,
        parameter_vector,
        args=(y,),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-6
    )

    est = res.x
    p11_est, p22_est, mu1_est, mu2_est = est[0], est[1], est[2], est[3]
    sigma1_est, sigma2_est = est[4], est[5]

    print(f"p11_est = {p11_est}")
    print(f"p22_est = {p22_est}")
    print(f"mu1_est = {mu1_est}")
    print(f"mu2_est = {mu2_est}")
    print(f"sigma1_est = {sigma1_est}")
    print(f"sigma2_est = {sigma2_est}")

    long_term_mean1 = (1 - p22_est) / (2 - p11_est - p22_est)
    long_term_mean2 = (1 - p11_est) / (2 - p11_est - p22_est)

    print(f"Long Term Mean 1 = {long_term_mean1}")
    print(f"Long Term Mean 2 = {long_term_mean2}")

    optimized_log_likelihood = -res.fun
    print(f"Log Likelihood value: {optimized_log_likelihood}")

    return (
        p11_est, p22_est, mu1_est, mu2_est,
        sigma1_est, sigma2_est, long_term_mean1,
        long_term_mean2, optimized_log_likelihood
    )


def hamilton_filter_long_run(p11, p22, mu1, mu2, sigma1, sigma2, y):
    """
    Hamilton filter with long-run mean initialization.

    Parameters
    ----------
    p11, p22, mu1, mu2, sigma1, sigma2 : floats
        Model parameters.
    y : array-like
        Observations.

    Returns
    -------
    filtered_xi : ndarray
        Filtered probabilities.
    predicted_xi : ndarray
        Predicted probabilities.
    """
    P = np.array([[p11, 1 - p22], [1 - p11, p22]])
    T = len(y)
    predicted_xi = np.zeros((2, T))
    filtered_xi = np.zeros((2, T))
    likelihood = np.zeros((2, T))

    # Initialize with stationary distribution (long-run mean)
    predicted_xi[:, 0] = [(1 - p22) / (2 - p11 - p22), (1 - p11) / (2 - p11 - p22)]

    for i in range(T):
        likelihood[:, i] = [
            norm.pdf(y[i], mu1, sigma1),
            norm.pdf(y[i], mu2, sigma2)
        ]
        filtered_xi[:, i] = (predicted_xi[:, i] * likelihood[:, i]) / np.dot(predicted_xi[:, i], likelihood[:, i])
        if i < T - 1:
            predicted_xi[:, i + 1] = P.dot(filtered_xi[:, i])

    return filtered_xi, predicted_xi


def log_likelihood_hamilton_long_run(parameters, y):
    """
    Negative log-likelihood using Hamilton filter with long-run initialization.

    Parameters
    ----------
    parameters : array-like
        Model parameters.
    y : array-like
        Observations.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    p11, p22, mu1, mu2, sigma1, sigma2 = parameters
    _, predicted_xi = hamilton_filter_long_run(p11, p22, mu1, mu2, sigma1, sigma2, y)
    ll = np.log(
        predicted_xi[0, :] * norm.pdf(y, mu1, sigma1) +
        predicted_xi[1, :] * norm.pdf(y, mu2, sigma2)
    )
    return -np.sum(ll)


def optimise_and_print_hamilton_long_run(parameter_vector, ll_func, y):
    """
    Optimize Hamilton filter with long-run initial state and print results.

    Parameters
    ----------
    parameter_vector : array-like
        Initial parameters guess.
    ll_func : callable
        Negative log-likelihood function.
    y : array-like
        Observations.

    Returns
    -------
    tuple
        Estimated parameters and log-likelihood.
    """
    bounds = [(0, 1), (0, 1), (-100, 100), (-100, 100), (0, 100), (0, 100)]

    res = scipy.optimize.minimize(
        ll_func,
        parameter_vector,
        args=(y,),
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-6
    )

    est = res.x
    p11_est, p22_est, mu1_est, mu2_est = est[0], est[1], est[2], est[3]
    sigma1_est, sigma2_est = est[4], est[5]

    print(f"p11_est = {p11_est}")
    print(f"p22_est = {p22_est}")
    print(f"mu1_est = {mu1_est}")
    print(f"mu2_est = {mu2_est}")
    print(f"sigma1_est = {sigma1_est}")
    print(f"sigma2_est = {sigma2_est}")

    long_term_mean1 = (1 - p22_est) / (2 - p11_est - p22_est)
    long_term_mean2 = (1 - p11_est) / (2 - p11_est - p22_est)

    print(f"Long Term Mean 1 = {long_term_mean1}")
    print(f"Long Term Mean 2 = {long_term_mean2}")

    optimized_log_likelihood = -res.fun
    print(f"Log Likelihood value: {optimized_log_likelihood}")

    return (
        p11_est, p22_est, mu1_est, mu2_est,
        sigma1_est, sigma2_est, long_term_mean1,
        long_term_mean2, optimized_log_likelihood
    )


def hamilton_filter_1d(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y):
    """
    Hamilton filter with user-defined initial state probabilities.

    Parameters
    ----------
    p11 : float
        Transition probability staying in state 1.
    p22 : float
        Transition probability staying in state 2.
    mu1 : float
        Mean of state 1.
    mu2 : float
        Mean of state 2.
    sigma1 : float
        Std dev of state 1.
    sigma2 : float
        Std dev of state 2.
    xi0_in : array-like, shape (2,)
        Initial predicted state probabilities.
    y : array-like
        Observations.

    Returns
    -------
    filtered_xi : ndarray
        Filtered state probabilities.
    predicted_xi : ndarray
        Predicted state probabilities.
    """
    P = np.array([[p11, 1 - p22], [1 - p11, p22]])
    T = len(y)
    predicted_xi = np.zeros((2, T))
    filtered_xi = np.zeros((2, T))
    likelihood = np.zeros((2, T))

    predicted_xi[:, 0] = xi0_in

    for i in range(T):
        likelihood[:, i] = [
            max(norm.pdf(y[i], mu1, sigma1), 1e-10),  # Prevent underflow
            max(norm.pdf(y[i], mu2, sigma2), 1e-10)
        ]
        denominator = np.dot(predicted_xi[:, i], likelihood[:, i]) + 1e-6
        filtered_xi[:, i] = (predicted_xi[:, i] * likelihood[:, i]) / denominator
        filtered_xi[:, i] /= np.sum(filtered_xi[:, i])

        if i < T - 1:
            predicted_xi[:, i + 1] = np.dot(P, filtered_xi[:, i])
            predicted_xi[:, i + 1] /= np.sum(predicted_xi[:, i + 1])

    return filtered_xi, predicted_xi


import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

def hamilton_smoother(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y):
    """
    Runs the Hamilton Smoother for a two-state Markov switching model.

    Parameters:
        p11, p22: transition probabilities.
        mu1, mu2: means of the regimes.
        sigma1, sigma2: standard deviations of the regimes.
        xi0_in: initial state probabilities.
        y: observed time series.

    Returns:
        smoothed_xi: smoothed state probabilities.
        xi0_out: updated initial probabilities.
        Pstar: joint smoothed state transition probabilities.
    """
    T = len(y)
    P = np.array([[p11, 1 - p22],
                  [1 - p11, p22]])

    filtered_xi, predicted_xi = hamilton_filter_1d(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y)
    smoothed_xi = np.zeros((2, T))
    smoothed_xi[:, T - 1] = filtered_xi[:, T - 1]

    for i in range(1, T):
        t = T - 1 - i
        smoothing_factor = (smoothed_xi[:, t + 1] / (predicted_xi[:, t + 1] + 1e-10))
        smoothed_xi[:, t] = filtered_xi[:, t] * (P.T @ smoothing_factor)
        smoothed_xi[:, t] /= np.sum(smoothed_xi[:, t])

    xi0_out = xi0_in * (P.T @ (smoothed_xi[:, 0] / (predicted_xi[:, 0] + 1e-10)))
    xi0_out /= np.sum(xi0_out)

    Pstar = np.empty((2, 2, T))
    Pstar[:, :, 0] = P * (np.outer(smoothed_xi[:, 0], xi0_in) /
                          np.dot(predicted_xi[:, 0], np.ones(2)))

    for t in range(1, T):
        for i in range(2):
            for j in range(2):
                Pstar[i, j, t] = (P[i, j] * filtered_xi[j, t - 1] * smoothed_xi[i, t]) / (predicted_xi[i, t] + 1e-10)
        Pstar[:, :, t] /= np.sum(Pstar[:, :, t])

    return smoothed_xi, xi0_out, Pstar


def EM_step(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y):
    """
    Performs one EM step for the Markov-switching model.
    """
    T = len(y)
    smoothed_xi, xi0_out, Pstar = hamilton_smoother(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y)

    p11star, p22star, p1star, p2star = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

    for t in range(T):
        p11star[t] = Pstar[0, 0, t]
        p1star[t] = np.sum(Pstar[0, :, t])
        p22star[t] = Pstar[1, 1, t]
        p2star[t] = np.sum(Pstar[1, :, t])

    p11_out = np.sum(p11star) / (xi0_out[0] + np.sum(p1star[:-1]))
    p22_out = np.sum(p22star) / (xi0_out[1] + np.sum(p2star[:-1]))
    mu1_out = np.sum(p1star * y) / np.sum(p1star)
    mu2_out = np.sum(p2star * y) / np.sum(p2star)
    sigma1_out = np.sqrt(np.sum(p1star * (y - mu1) ** 2) / np.sum(p1star))
    sigma2_out = np.sqrt(np.sum(p2star * (y - mu2) ** 2) / np.sum(p2star))

    return p11_out, p22_out, mu1_out, mu2_out, sigma1_out, sigma2_out, xi0_out


def log_likelihood_hamilton_1d(parameters, xi0, y):
    """
    Computes the log-likelihood of the Hamilton filter.
    """
    p11, p22, mu1, mu2, sigma1, sigma2 = parameters
    _, predicted_xi = hamilton_filter_1d(p11, p22, mu1, mu2, sigma1, sigma2, xi0, y)
    ll = np.log(predicted_xi[0, :] * norm.pdf(y, mu1, sigma1) + 
                predicted_xi[1, :] * norm.pdf(y, mu2, sigma2))
    return np.sum(ll)


def negative_log_likelihood_LL(parameter_vector, y):
    """
    Computes the negative log-likelihood using a Kalman Filter.
    """
    phi, Q, R = parameter_vector
    _, _, predicted_xi, predicted_P = KF_LL(phi, Q, R, y, 0, Q / (1 - phi ** 2))

    LL = 0
    for t in range(len(y)):
        sigma = predicted_P[t] + R
        mu = predicted_xi[t]
        LL += -0.5 * np.log(2 * np.pi * sigma) - 0.5 * ((y[t] - mu) ** 2 / sigma)

    return -LL


def KF_LL(phi, Q, R, y, mu_init, P_init):
    """
    Kalman filter for AR(1) process estimation.
    """
    T = len(y)
    predicted_xi = np.zeros(T)
    predicted_P = np.zeros(T)
    xi = np.zeros(T)
    P = np.zeros(T)

    predicted_xi[0] = mu_init
    predicted_P[0] = P_init

    xi[0] = predicted_xi[0] + (predicted_P[0] / (predicted_P[0] + R)) * (y[0] - predicted_xi[0])
    P[0] = predicted_P[0] - (predicted_P[0] / (predicted_P[0] + R)) * predicted_P[0]

    for t in range(1, T):
        predicted_xi[t] = phi * xi[t - 1]
        predicted_P[t] = phi ** 2 * P[t - 1] + Q
        xi[t] = predicted_xi[t] + (predicted_P[t] / (predicted_P[t] + R)) * (y[t] - predicted_xi[t])
        P[t] = predicted_P[t] - (predicted_P[t] / (predicted_P[t] + R)) * predicted_P[t]

    return xi, P, predicted_xi, predicted_P


# Load series INDPRO (column 0)
y = MainData_demeaned.iloc[:, 0]

# Initial guesses: phi, Q (sigma_theta^2), R (sigma_error^2)
initial_guess = [0.5, 15, 25]
bounds = [(-1 + 1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None)]

# Optimization options
options = {
    'maxiter': 1000,
    'disp': True,
}

# Run optimization for MLE
result = minimize(
    negative_log_likelihood_LL,
    initial_guess,
    args=(y,),
    bounds=bounds,
    options=options,
    method='SLSQP',
    tol=1e-8
)

# Extract MLE estimates
phi_ML, Q_ML, R_ML = result.x
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("ML_LogL:", -ML_LogL)  # Negative log-likelihood is minimized
import matplotlib.pyplot as plt

# Run Kalman Filter with estimated parameters
xi, P, predicted_xi, predicted_P = KF_LL(
    phi_ML,
    Q_ML,
    R_ML,
    y,
    mu_init=0,
    P_init=Q_ML / (1 - phi_ML**2)
)

T = len(y)
plt.figure(figsize=(10, 6))

plt.scatter(range(1, T + 1), y, color='k', label='Observations')
plt.plot(range(1, T + 1), predicted_xi, 'r-', label='Predicted State', linewidth=2)
plt.plot(range(1, T + 1), xi, 'k-', label='Updated State (Kalman)', linewidth=1)

plt.xlabel('Time (t)')
plt.ylabel('State Estimate')
plt.title('Kalman Filter State Estimates - AR(1)')
plt.legend()
plt.grid(True)
plt.show()
def KF_LL_ARMA(phi, theta, Q, R, y, mu_init, P_init):
    """
    Kalman Filter for ARMA(1,1) state-space representation.
    """
    T = len(y)
    predicted_xi = np.zeros(T)
    predicted_P = np.zeros(T)
    xi = np.zeros(T)
    P = np.zeros(T)
    eta = np.zeros(T)

    predicted_xi[0] = mu_init
    predicted_P[0] = P_init

    xi[0] = predicted_xi[0] + (predicted_P[0] / (predicted_P[0] + R)) * (y[0] - predicted_xi[0])
    P[0] = predicted_P[0] - (predicted_P[0] / (predicted_P[0] + R)) * predicted_P[0]
    eta[0] = xi[0] - phi * mu_init

    for t in range(1, T):
        predicted_xi[t] = phi * xi[t - 1] + theta * eta[t - 1]
        predicted_P[t] = phi**2 * P[t - 1] + (1 + theta**2) * Q

        xi[t] = predicted_xi[t] + (predicted_P[t] / (predicted_P[t] + R)) * (y[t] - predicted_xi[t])
        P[t] = predicted_P[t] - (predicted_P[t] / (predicted_P[t] + R)) * predicted_P[t]

        eta[t] = xi[t] - phi * xi[t - 1] - theta * eta[t - 1]

    return eta, xi, P, predicted_xi, predicted_P
def negative_log_likelihood_LL_ARMA(params, y):
    """
    Negative log-likelihood function for ARMA(1,1) model.
    """
    phi, theta, Q, R = params
    mu_init = 0
    P_init = Q * (theta**2 + 1) / (1 - phi**2)

    _, _, _, predicted_xi, predicted_P = KF_LL_ARMA(phi, theta, Q, R, y, mu_init, P_init)

    LL = 0
    for t in range(len(y)):
        sigma2 = max(predicted_P[t] + R, 1e-6)
        mu = predicted_xi[t]
        LL += -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((y[t] - mu) ** 2 / sigma2)

    return -LL
y = MainData_demeaned.iloc[:, 0]  # INDPRO

initial_guess = [0.5, 0.5, 30, 15]  # phi, theta, Q, R
bounds = [
    (-1 + 1e-6, 1 - 1e-6),
    (-1 + 1e-6, 1 - 1e-6),
    (1e-6, None),
    (1e-6, None)
]

result = minimize(
    negative_log_likelihood_LL_ARMA,
    initial_guess,
    args=(y,),
    bounds=bounds,
    method='SLSQP',
    tol=1e-8
)

phi_ML, theta_ML, Q_ML, R_ML = result.x
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("theta_ML:", theta_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("ML_LogL:", -ML_LogL)
eta_prev, xi, P, predicted_xi, predicted_P = KF_LL_ARMA(
    phi_ML, theta_ML, Q_ML, R_ML,
    y, 0, Q_ML * (theta_ML**2 + 1) / (1 - phi_ML**2)
)

T = len(y)
plt.figure(figsize=(12, 8))

# Plot state estimates
plt.subplot(2, 1, 1)
plt.scatter(range(1, T + 1), y, color='k', label='Observations', s=10)
plt.plot(range(1, T + 1), predicted_xi, 'r-', label='Predicted State', linewidth=2)
plt.plot(range(1, T + 1), xi, 'b-', label='Updated State', linewidth=1)
plt.title('ARMA(1,1) Kalman Filter Estimates')
plt.legend()
plt.grid(True)

# Plot residuals
plt.subplot(2, 1, 2)
plt.plot(range(1, T + 1), eta_prev, 'g-', label='Residuals (eta)', linewidth=2)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.title('MA(1) Residuals (Eta)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



eta_prev, xi, P, predicted_xi, predicted_P = KF_LL_ARMA(phi_ML, theta_ML, Q_ML, R_ML, y, 0, Q_ML*(theta_ML**2 + 1) / (1 - phi_ML**2))

# To repeat this analysis for RPI and PAYEMS, change the line:

# Visualization of results for the ARMA(1,1) model for RPI
# y = MainData_demeaned.iloc[:, 1]  # For RPI
# or
# y = MainData_demeaned.iloc[:, 2]  # For PAYEMS

T = len(y)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def negative_log_likelihood_LL_ARMA(params, y):
    """
    Negative log-likelihood function for ARMA(1,1) model using the Kalman filter.

    Parameters:
        params (list): Model parameters [phi, theta, Q, R].
        y (np.ndarray): Observed data.

    Returns:
        float: Negative log-likelihood.
    """
    phi, theta, Q, R = params
    T = len(y)

    xi = np.zeros(T)
    P = np.zeros(T)
    predicted_xi = np.zeros(T)
    eta_prev = np.zeros(T)

    xi[0] = 0
    P[0] = 10e6

    log_likelihood = 0

    for t in range(T):
        if t > 0:
            predicted_xi[t] = phi * xi[t - 1] + theta * eta_prev[t - 1]
            P[t] = phi ** 2 * P[t - 1] + Q

        v = y[t] - predicted_xi[t]
        S = P[t] + R
        K = P[t] / S

        xi[t] = predicted_xi[t] + K * v
        P[t] = (1 - K) * P[t]
        eta_prev[t] = v

        log_likelihood += -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)

    return -log_likelihood


def EM_DynamicFactorModel(y, max_iter=1000):
    """
    Estimate parameters in a dynamic factor model using the EM algorithm.

    Parameters:
        y (np.ndarray): Observed multivariate data (T x n).
        max_iter (int): Maximum number of EM iterations.

    Returns:
        Tuple containing estimated Lambda, Sigma, phi, sigma_eta2, smoothed factors, 
        filtered factors, log-likelihood history, and mean filtered factor.
    """
    T, n = y.shape
    Lambda = np.ones((n, 1))
    Sigma = 0.2 * np.cov(np.diff(y, axis=0), rowvar=False)
    phi = 0.85
    sigma_eta2 = 1.0

    logL_history = []
    f_filt_mean = []

    for _ in range(max_iter):
        f_smoothed, f_filt, P_smoothed, logL, P_cross = KalmanSmoother(y, Lambda, Sigma, phi, sigma_eta2)
        logL_history.append(logL)
        f_filt_mean.append(np.mean(f_filt))

        Lambda = (np.linalg.inv(f_smoothed.T @ f_smoothed) @ (f_smoothed.T @ y)).T
        y_pred = (Lambda @ f_smoothed.T).T
        residuals = y - y_pred
        Sigma = np.cov(residuals, rowvar=False)

        f_t = f_smoothed[:-1]
        f_t_plus1 = f_smoothed[1:]
        phi = np.sum(f_t * f_t_plus1) / np.sum(f_t ** 2)
        sigma_eta2 = np.mean((f_t_plus1 - phi * f_t) ** 2)

    return Lambda, Sigma, phi, sigma_eta2, f_smoothed, f_filt, logL_history, f_filt_mean


def KalmanSmoother(y, Lambda, Sigma, phi, sigma_eta2):
    """
    Run Kalman filter and smoother for dynamic factor model.

    Parameters:
        y (np.ndarray): Observed multivariate data.
        Lambda (np.ndarray): Loading matrix.
        Sigma (np.ndarray): Noise covariance matrix.
        phi (float): AR(1) coefficient.
        sigma_eta2 (float): Process noise variance.

    Returns:
        Smoothed states, filtered states, smoothed covariances, log-likelihood, and cross-covariances.
    """
    T, n = y.shape
    f_pred = np.zeros((T, 1))
    P_pred = np.zeros((T, 1, 1))
    f_filt = np.zeros((T, 1))
    P_filt = np.zeros((T, 1, 1))
    logL = 0

    f_pred[0] = 0
    P_pred[0] = 10e6

    for t in range(T):
        if t > 0:
            f_pred[t] = phi * f_filt[t - 1]
            P_pred[t, 0, 0] = phi ** 2 * P_filt[t - 1, 0, 0] + sigma_eta2
            P_pred[t] = (P_pred[t] + P_pred[t].T) / 2

        v_t = y[t, :].reshape(-1, 1) - Lambda @ f_pred[t]
        S_t = Lambda @ (P_pred[t] @ Lambda.T) + Sigma
        S_t = (S_t + S_t.T) / 2
        K_t = (P_pred[t] * Lambda.T) @ np.linalg.inv(S_t)

        f_filt[t] = f_pred[t] + (K_t @ v_t)
        P_filt[t] = P_pred[t] - (K_t @ Lambda @ P_pred[t])

        logL += -0.5 * (np.log(np.linalg.det(S_t * 2 * np.pi)) + v_t.T @ np.linalg.inv(S_t) @ v_t)

    f_smoothed = np.copy(f_filt)
    P_smoothed = np.copy(P_filt)
    P_cross = np.zeros((T - 1, 1, 1))

    for t in reversed(range(T - 1)):
        J_t = P_filt[t] * phi @ np.linalg.inv(P_pred[t + 1])
        f_smoothed[t] = f_filt[t] + J_t @ (f_smoothed[t + 1] - f_pred[t + 1])
        P_smoothed[t] = P_filt[t] - J_t @ (P_pred[t + 1] - P_smoothed[t + 1]) @ J_t.T
        P_smoothed[t] = (P_smoothed[t] + P_smoothed[t].T) / 2

        if P_pred[t + 1].ndim < 2:
            P_pred[t + 1] = P_pred[t + 1].reshape(1, 1)
        P_cross[t] = P_smoothed[t + 1] @ np.linalg.inv(P_pred[t + 1]) @ (phi * P_filt[t])

    return f_smoothed, f_filt, P_smoothed, logL, P_cross


def plot_kalman_results(T, y, predicted_xi, xi, eta_prev):
    """
    Plot Kalman Filter state estimates and residuals for an ARMA(1,1) model.

    Parameters:
        T (int): Number of time steps.
        y (np.ndarray): Observations.
        predicted_xi (np.ndarray): Predicted state estimates.
        xi (np.ndarray): Updated state estimates from Kalman filter.
        eta_prev (np.ndarray): Residuals (MA component).
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(range(1, T + 1), y, color='k', label='Observations', s=10)
    plt.plot(range(1, T + 1), predicted_xi, 'r-', label='Predicted State Estimate', linewidth=2)
    plt.plot(range(1, T + 1), xi, 'b-', label='Kalman Filter (Updated State)', linewidth=1)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title('ARMA(1,1) Model: Kalman Filter Estimates', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(1, T + 1), eta_prev, 'g-', label='Residuals (MA Component)', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Residuals (eta)', fontsize=12)
    plt.title('Eta', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_factors(y, f_filt, f_smoothed):
    """
    Plot observed data series alongside filtered and smoothed factors.

    Parameters:
        y (np.ndarray): Multivariate time series data (T x n).
        f_filt (np.ndarray): Filtered factor estimates.
        f_smoothed (np.ndarray): Smoothed factor estimates.
    """
    T, n = y.shape
    plt.figure(figsize=(12, 8))

    for i in range(n):
        plt.plot(range(T), y[:, i], label=f"Data Series {i + 1}", alpha=0.6)

    plt.plot(range(T), f_filt.squeeze(), label="Filtered Factor", color="black", linestyle="--", linewidth=2)
    plt.plot(range(T), f_smoothed.squeeze(), label="Smoothed Factor", color="red", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Comparison of Data Series with Filtered and Smoothed Common Factors")
    plt.legend()
    plt.grid(True)
    plt.show()

