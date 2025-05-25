"""
This module contains all functions needed for the main program to run.
"""

import os
import time
import random
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import least_squares
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from numpy.random import seed
from rpy2.robjects import r, pandas2ri, FloatVector, DataFrame
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

from midas.weights import (
    polynomial_weights,
    WeightMethod,
    BetaWeights,
    ExpAlmonWeights,
)  # midaspy
from midas.fit import jacobian_wx  # midaspy

pandas2ri.activate()  # activates R


def lagaugment(a, p):
    """
    Extends the original matrix with its lags.

    Parameters
    ----------
    a : np.ndarray
        The original matrix.
    p : int
        The number of lags the matrix is expanded by.

    Returns
    -------
    tuple or np.ndarray
        If p > 0, returns (a, b) where b is a matrix of lags,
        otherwise returns a unchanged.
    """
    try:
        a = a.reshape(a.shape[0], 1)
    except (ValueError, IndexError):
        pass
    if p == 0:
        return a
    else:
        b = np.full((a.shape[0], p * a.shape[1]), np.nan)
        for pp in range(p):
            b[pp + 1 :, a.shape[1] * pp : (pp + 1) * a.shape[1]] = a[: -(pp + 1), :]
        return a, b


def dmtest(e1, e2, h):
    """
    Calculates the Diebold-Mariano (DM) test statistic.

    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors from model 1.
    e2 : np.ndarray
        Forecast errors from model 2.
    h : int
        The forecast horizon, equals 1 when nowcasting GDP.

    Returns
    -------
    float
        The Diebold-Mariano test statistic.
    """
    e1 = e1.reshape(e1.shape[0], 1)
    e2 = e2.reshape(e2.shape[0], 1)
    T = e1.shape[0]  # number of observations

    d = e1**2 - e2**2  # loss differential
    mu = np.mean(d)
    gamma0 = np.var(d) * T / (T - 1)  # autocovariance at lag 0

    if h > 1:
        gamma = np.zeros(h - 1)
        for i in range(1, h):
            s_cov = np.cov(np.vstack((d[i:T].T, d[0 : T - i].T)))
            gamma[i - 1] = s_cov[0, 1]
        var_d = gamma0 + 2 * np.sum(gamma)
    else:
        var_d = gamma0

    dm_stat = mu / np.sqrt((1 / T) * var_d)  # DM statistic ~ N(0,1)
    return dm_stat


def get_pcs(data_training, mode, hyper, k):
    """
    Returns principal components associated with the data.

    Parameters
    ----------
    data_training : pd.DataFrame
        The monthly regressor data.
    mode : list or str
        Contains the type of principal component to be returned.
    hyper : float
        Hyperparameter value for the principal component.
    k : int
        Number of principal components to return.

    Returns
    -------
    np.ndarray
        Matrix consisting of the principal components.
    """
    # Standardize the training data
    data_training_standardized = stats.zscore(data_training, axis=0, ddof=1)

    # Determine kernel type if applicable
    if mode[0] == 'K':
        kernel = mode[3:]  # Extract kernel type
    else:
        kernel = False

    # PCA
    if mode[0] == 'P':
        pca = PCA(n_components=k)
        fhat = pca.fit_transform(data_training_standardized)

    # SPC (Squared Principal Components)
    elif mode[0] == 'S':
        wts = np.hstack((data_training_standardized, data_training_standardized**2))
        wts = StandardScaler().fit_transform(wts)
        pca = PCA(n_components=k)
        fhat = pca.fit_transform(wts)

    # KPCA (Kernel PCA)
    elif mode[0] == 'K':
        if kernel in ['sigmoid', 'rbf']:
            transformer = KernelPCA(n_components=k, kernel=kernel, gamma=hyper)
        else:
            transformer = KernelPCA(n_components=k, kernel=kernel, degree=2)
        fhat = transformer.fit_transform(data_training_standardized)

    else:
        fhat = None

    return fhat


def prepare_current_data(m, first_month_in_data, use_vintage_data):
    """
    Extracts and processes the current data.

    Parameters
    ----------
    m : int
        The current vintage data requested.
    first_month_in_data : int
        The first month in the training data.
    use_vintage_data : bool
        Whether or not vintage data is used.

    Returns
    -------
    None
        Processed data is saved to csv files for safety.
    """
    q = (m - 1) // 3  # current quarter
    year = 1960 + (m - 1) // 12  # current year
    month = (m - 1) % 12 + 1  # current vintage month
    monthly_str = f"{year}-{month:02d}m.csv"

    # Clear existing modified data files
    for fname in [
        "/content/Monthly_Data_Modified.csv",
        "/content/Quarterly_Data_Modified.csv",
        "/content/Transformed_Monthly_Data.csv",
        "/content/Transformed_Quarterly_Data.csv",
    ]:
        with open(fname, "w"):
            pass

    if use_vintage_data:
        print(monthly_str)
        monthly_file_path = f"/content/{monthly_str}"
        quarterly_file_path = "/content/Quarterly_Vintages.xlsx"

        monthly = pd.read_csv(monthly_file_path, header=None)
        quarterly = pd.read_excel(quarterly_file_path, header=None)

        # Extract current quarterly vintage column with adjustment
        col_quarterly = m - 69
        quarterly = quarterly.iloc[:, col_quarterly]

        # Trim quarterly data up to last valid index
        end_index = quarterly.last_valid_index()
        quarterly = quarterly.iloc[: end_index + 1]

        # Save adjusted data
        quarterly = quarterly.to_frame()
        monthly.to_csv("/content/Monthly_Data_Modified.csv", header=None, index=False)
        quarterly.to_csv("/content/Quarterly_Data_Modified.csv", header=None, index=False)

        # Transform and impute
        transformation_function(
            "/content/Monthly_Data_Modified.csv",
            "/content/Quarterly_Data_Modified.csv",
            first_month_in_data,
            m,
        )
        Setup_Imputation()
        Impute()

    else:
        # Load last monthly vintage and quarterly vintages
        monthly_file_path = "/content/2024-04m.csv"
        quarterly_file_path = "/content/Quarterly_Vintages.xlsx"

        monthly = pd.read_csv(monthly_file_path, header=None)
        quarterly = pd.read_excel(quarterly_file_path, header=None)

        # Load vintage reference files
        monthly_reference = pd.read_csv(f"/content/{monthly_str}", header=None)
        quarterly_reference = pd.read_excel(quarterly_file_path, header=None)
        print(monthly_str)

        col_quarterly = m - 69
        quarterly = quarterly.iloc[:, col_quarterly]
        end_index = quarterly.last_valid_index()
        quarterly = quarterly.iloc[: end_index + 1]

        # Align lengths with reference vintages
        monthly_diff = len(monthly) - len(monthly_reference)
        quarterly_diff = len(quarterly) - len(quarterly_reference)
        if monthly_diff > 0:
            monthly = monthly.iloc[:-monthly_diff]
        if quarterly_diff > 0:
            quarterly = quarterly.iloc[:-quarterly_diff]

        # Variable groups for lag treatment
        cols_zero_or_one = [116, 117, 118, 119]
        cols_one = [58, 62, 63, 72]
        cols_one_or_two = [1, 2, 3, 75, 124, 125]
        cols_zero_to_eight = [20, 21, 76]

        # Apply zeroing of last elements randomly as per variable group
        for col in cols_zero_or_one:
            num_to_zero = random.randint(0, 1)
            monthly.iloc[-num_to_zero:, col] = 0

        for col in cols_one:
            monthly.iloc[-1, col] = 0

        for col in cols_one_or_two:
            num_to_zero = random.randint(1, 2)
            monthly.iloc[-num_to_zero:, col] = 0

        for col in cols_zero_to_eight:
            num_to_zero = random.randint(0, 8)
            monthly.iloc[-num_to_zero:, col] = 0

        # Save adjusted data
        quarterly = quarterly.to_frame()
        monthly.to_csv("/content/Monthly_Data_Modified.csv", header=None, index=False)
        quarterly.to_csv("/content/Quarterly_Data_Modified.csv", header=None, index=False)

        # Transform and impute
        transformation_function(
            "/content/Monthly_Data_Modified.csv",
            "/content/Quarterly_Data_Modified.csv",
            first_month_in_data,
            m,
        )
        Setup_Imputation()
        Impute()

    return None


def transformation_function(
    new_monthly_file_path, new_quarterly_file_path, first_month_in_data, m
):
    """
    Transforms the monthly and quarterly data.

    Parameters
    ----------
    new_monthly_file_path : str
        Path to monthly data CSV file.
    new_quarterly_file_path : str
        Path to quarterly data CSV file.
    first_month_in_data : int
        The first month in the training data.
    m : int
        The current month.

    Returns
    -------
    None
        The transformed data is saved to CSV files.
    """
    monthly = pd.read_csv(new_monthly_file_path, header=None)
    quarterly = pd.read_csv(new_quarterly_file_path, header=None)

    first_quarter_in_data = (first_month_in_data - 1) // 3
    q = (m - 1) // 3 + 1  # added one to correct quarter index

    rawdata1 = monthly.values[13 + first_month_in_data - 1 : 13 + m, 1:]
    trans_code1 = monthly.values[1, 1:].astype(int).reshape(-1, 1)

    rawdata2 = quarterly.values[53 + first_quarter_in_data - 1 :, 0]

    X = np.full(rawdata1.shape, np.nan)
    small = 1e-6  # Threshold for small values

    for i in range(rawdata1.shape[1]):
        x = rawdata1[:, i].astype(float)
        tcode = int(trans_code1[i][0])
        n = x.shape[0]

        if tcode == 1:
            # Level (no transformation)
            X[:, i] = x
        elif tcode == 2:
            # First difference
            X[1:n, i] = x[1:n] - x[0 : n - 1]
        elif tcode == 3:
            # Second difference
            X[2:n, i] = x[2:n] - 2 * x[1 : n - 1] + x[0 : n - 2]
        elif tcode == 4:
            # Natural log
            for j in range(n):
                if x[j] > small:
                    X[j, i] = np.log(x[j])
                else:
                    X[j, i] = np.nan
        elif tcode == 5:
            # First difference of natural log
            for j in range(1, n):
                if x[j] > small and x[j - 1] > small:
                    X[j, i] = np.log(x[j]) - np.log(x[j - 1])
                else:
                    X[j, i] = np.nan
        elif tcode == 6:
            # Second difference of natural log
            for j in range(2, n):
                if x[j] > small and x[j - 1] > small and x[j - 2] > small:
                    X[j, i] = np.log(x[j]) - 2 * np.log(x[j - 1]) + np.log(x[j - 2])
                else:
                    X[j, i] = np.nan
        elif tcode == 7:
            # First difference of percent change
            z = np.zeros(n)
            for j in range(1, n):
                if x[j - 1] > small:
                    z[j] = (x[j] - x[j - 1]) / x[j - 1]
            for j in range(2, n):
                X[j, i] = z[j] - z[j - 1]

    Y = np.full((rawdata2.shape[0], 1), np.nan)

    x = rawdata2.astype(float)
    n = x.shape[0]
    for j in range(1, n):
        if x[j] > small and x[j - 1] > small:
            Y[j, 0] = np.log(x[j]) - np.log(x[j - 1])
        else:
            Y[j, 0] = np.nan

    np.savetxt("/content/Monthly_Data_Modified.csv", X, delimiter=",")
    np.savetxt("/content/Quarterly_Data_Modified.csv", Y, delimiter=",")

    return None


import numpy as np
import pandas as pd
from pandas import DataFrame
from rpy2.robjects import r, FloatVector
from scipy.optimize import least_squares

# Assuming polynomial_weights, BetaWeights, jacobian_wx are defined elsewhere


def setup_imputation():
    """
    Sets up the data so it can be imputed later.

    Reads monthly and quarterly data CSV files, cleans the data by:
    - Dropping columns with all NaN values
    - Dropping the first row if it contains NaNs
    - Replacing zeros with NaNs
    - Performing forward-fill AR(1) imputation on the quarterly data
    
    The processed data is saved back to CSV files for later imputation.

    Parameters:
        None (Uses local CSV files as data source)

    Returns:
        None (Saves processed data to CSV files)
    """
    monthly = pd.read_csv("/content/Monthly_Data_Modified.csv", header=None)
    quarterly = pd.read_csv("/content/Quarterly_Data_Modified.csv", header=None)

    # Drop columns that contain only NaNs
    monthly = monthly.dropna(axis=1, how='all')

    # Drop the first row (likely containing NaNs after previous steps)
    monthly = monthly.drop(index=0)
    quarterly = quarterly.drop(index=0)

    # Replace zero values with NaN
    monthly.replace(0, np.nan, inplace=True)
    quarterly.replace(0, np.nan, inplace=True)

    # Convert quarterly DataFrame to Series for AR imputation
    quarterly = quarterly.iloc[:, 0]

    # Forward-fill NaN values in quarterly data (AR(1) imputation)
    if quarterly.isnull().any():
        quarterly.fillna(method='ffill', inplace=True)

    # Convert back to DataFrame
    quarterly = quarterly.to_frame()

    # Save processed data
    monthly.to_csv("/content/Monthly_Data_Modified.csv", header=None, index=False)
    quarterly.to_csv("/content/Transformed_Quarterly_Data.csv", header=None, index=False)


def impute():
    """
    Performs imputation on monthly data using an EM algorithm implemented in R.

    Reads the cleaned monthly data CSV, passes it to R,
    runs the EM algorithm from the 'fbi' R package,
    and saves the transformed data back to a CSV file.

    Parameters:
        None (Reads from and writes to local CSV files)

    Returns:
        None (Saves imputed data to CSV)
    """
    monthly = pd.read_csv("/content/Monthly_Data_Modified.csv", header=None)

    # Convert data to R format (list of FloatVectors)
    monthly_r_format = DataFrame({
        col: FloatVector(monthly[col].astype(float).values)
        for col in monthly.columns
    })

    # Assign data to R environment
    r.assign('Monthly', monthly_r_format)

    # R code to run EM algorithm and save results
    r_code = """
    if (!requireNamespace("devtools", quietly = TRUE)) {
        install.packages("devtools", repos='http://cran.us.r-project.org')
    }
    if (!requireNamespace("readr", quietly = TRUE)) {
        install.packages("readr", repos='http://cran.us.r-project.org')
    }
    if (!requireNamespace("stats", quietly = TRUE)) {
        install.packages("stats", repos='http://cran.us.r-project.org')
    }
    library(devtools)
    library(readr)
    library(stats)

    if (!requireNamespace("fbi", quietly = TRUE)) {
        devtools::install_github("cykbennie/fbi")
    }
    library(fbi)

    transform_dataset <- function(Monthly) {
        kmax <- 12
        if (!is.matrix(Monthly)) {
            Monthly <- as.matrix(Monthly)
        }
        transformed_monthly <- tp_apc(Monthly, kmax = kmax, center = TRUE, standardize = TRUE, re_estimate = TRUE)
        write.csv(as.data.frame(transformed_monthly$data), '/content/Transformed_Monthly_Data.csv', row.names = FALSE)
    }

    transform_dataset(Monthly)
    """

    # Execute R code
    r(r_code)

    # Reload transformed data and save again for reliability
    transformed_monthly = pd.read_csv("/content/Transformed_Monthly_Data.csv")
    np.savetxt("/content/Transformed_Monthly_Data.csv", transformed_monthly, delimiter=',')


def estimate_general(y, yl, X, polys, lambda_1):
    """
    Estimates a MIDAS model using Nonlinear Least Squares (NLS).

    Parameters:
        y (np.ndarray): Dependent variable (growth rate of GDP)
        yl (np.ndarray or None): First lag of dependent variable (growth rate of GDP)
        X (np.ndarray): Regressor dataset (higher frequency)
        polys (list): Polynomial weighting methods used for MIDAS
        lambda_1 (float): Ridge penalty coefficient to prevent overfitting

    Returns:
        opt_res (OptimizeResult): Result object with estimated parameters
    """
    # Extract weighting methods instances
    weight_methods = [polynomial_weights(poly) for poly in polys]

    # Number of regressors equals number of weighting methods
    num_regressors = len(weight_methods)

    # Apply initial polynomial weights to X
    xws = [
        weight_method.x_weighted(X[:, i * 3:(i + 1) * 3], weight_method.init_params())[0]
        for i, weight_method in enumerate(weight_methods)
    ]

    # Concatenate weighted regressors into 2D array
    xw_concat = np.concatenate([xw.reshape((len(xw), 1)) for xw in xws], axis=1)

    if yl is not None:
        # OLS initial parameters with lagged y
        c = np.linalg.lstsq(
            np.concatenate([np.ones((len(xw_concat), 1)), xw_concat, yl], axis=1),
            y,
            rcond=None,
        )[0]

        # Objective and Jacobian for NLS
        f = lambda v: ssr_generalized(v, X, y, yl, weight_methods, lambda_1)
        jac = lambda v: jacobian_generalized(v, X, y, yl, weight_methods)

        c_flat = c.T.flatten()

        # Initial params: intercept, regressors, poly params, lag params
        init_params = np.concatenate(
            [
                c_flat[0 : num_regressors + 1],
                *[wm.init_params() for wm in weight_methods],
                c_flat[num_regressors + 1 :],
            ]
        )
    else:
        # OLS initial parameters without lagged y
        c = np.linalg.lstsq(
            np.concatenate([np.ones((len(xw_concat), 1)), xw_concat], axis=1),
            y,
            rcond=None,
        )[0]

        f = lambda v: ssr_generalized(v, X, y, yl, weight_methods, lambda_1)
        jac = lambda v: jacobian_generalized(v, X, y, yl, weight_methods)

        c_flat = c.T.flatten()

        init_params = np.concatenate(
            [c_flat[0 : num_regressors + 1], *[wm.init_params() for wm in weight_methods]]
        )

    # Bounds if using BetaWeights to enforce positivity on theta1, theta2
    if isinstance(weight_methods[0], BetaWeights):
        lower_bounds = [-np.inf] * (1 + num_regressors)
        upper_bounds = [np.inf] * (1 + num_regressors)

        for _ in range(num_regressors):
            lower_bounds.extend([1e-6, 1e-6])
            upper_bounds.extend([np.inf, np.inf])

        if yl is not None:
            lower_bounds.extend([-np.inf] * len(yl[0]))
            upper_bounds.extend([np.inf] * len(yl[0]))

        opt_res = least_squares(
            f,
            init_params,
            jac,
            bounds=(lower_bounds, upper_bounds),
            xtol=1e-9,
            ftol=1e-9,
            max_nfev=5000,
            verbose=0,
        )
    else:
        opt_res = least_squares(
            f,
            init_params,
            jac,
            xtol=1e-9,
            ftol=1e-9,
            max_nfev=5000,
            verbose=0,
        )

    return opt_res


def forecast_general(Xfc, yfcl, res, polys):
    """
    Produces forecasts using the MIDAS model with estimated parameters.

    Parameters:
        Xfc (np.ndarray): Factor values for the current quarter (can be single or multiple forecasts)
        yfcl (np.ndarray or None): Previous quarter GDP growth rate(s)
        res (OptimizeResult): Estimated parameters from MIDAS model
        polys (list): Polynomial weighting methods used

    Returns:
        yf or yfs (float or np.ndarray): Forecasted value(s)
    """
    weight_methods = [polynomial_weights(poly) for poly in polys]

    a = res.x[0]  # Intercept
    num_regressors = len(weight_methods)
    bs = res.x[1 : num_regressors + 1]
    thetas = res.x[num_regressors + 1 : num_regressors + 1 + 2 * num_regressors]
    lambdas = res.x[num_regressors + 1 + 2 * num_regressors :]

    if Xfc.ndim == 1:  # Single forecast
        yf = a
        for i, weight_method in enumerate(weight_methods):
            theta1, theta2 = thetas[2 * i : 2 * i + 2]
            xw, _ = weight_method.x_weighted(Xfc[i * 3 : (i + 1) * 3], [theta1, theta2])
            yf += bs[i] * xw

        if yfcl is not None:
            for i in range(len(lambdas)):
                yf += lambdas[i] * yfcl[i]

        return yf

    elif Xfc.ndim == 2:  # Multiple forecasts
        nof = Xfc.shape[0]
        yfs = np.zeros((nof, 1))

        for j in range(nof):
            yf = a
            for i, weight_method in enumerate(weight_methods):
                theta1, theta2 = thetas[2 * i : 2 * i + 2]
                xw, _ = weight_method.x_weighted(Xfc[j, i * 3 : (i + 1) * 3], [theta1, theta2])
                yf += bs[i] * xw

            if yfcl is not None:
                for i in range(len(lambdas)):
                    yf += lambdas[i] * yfcl[j, i]

            yfs[j, 0] = yf

        return yfs


def new_x_weighted(self, x, params):
    """
    Transforms higher-frequency data into lower-frequency using polynomial weights.

    Parameters:
        self: Instance of Beta polynomial weights class
        x (np.ndarray): Higher-frequency regressor data
        params (list or np.ndarray): Current parameter values [theta1, theta2]

    Returns:
        result (float or np.ndarray): Transformed lower-frequency data
        weights (np.ndarray): Corresponding weights used
    """
    self.theta1, self.theta2 = params

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Calculate weights
    w = self.weights(x.shape[1])

    # Weighted sum
    result = np.dot(x, w)

    # Convert single-element array to scalar
    if result.size == 1:
        result = result.item()

    # Repeat weights to match number of rows if needed
    return result, np.tile(w.T, (x.shape[0], 1))


def ssr_generalized(a, X, y, yl, weight_methods, lambda_1):
    """
    Computes the sum of squared residuals with penalties for MIDAS estimation.

    Parameters:
        a (np.ndarray): Current parameter estimates (coefficients and polynomial params)
        X (np.ndarray): High-frequency data
        y (np.ndarray): Dependent variable
        yl (np.ndarray or None): Lagged dependent variable
        weight_methods (list): Polynomial weighting method instances
        lambda_1 (float): Ridge penalty coefficient

    Returns:
        objective_value (float): Sum of squared residuals + penalties
    """
    num_regressors = len(weight_methods)
    products = []
    theta_params = []

    for i in range(num_regressors):
        theta_start = 1 + i * 2 + num_regressors
        theta_end = theta_start + 2
        theta_params.extend(a[theta_start:theta_end])

        xw, _ = weight_methods[i].x_weighted(X[:, i * 3 : (i + 1) * 3], a[theta_start:theta_end])
        products.append(a[1 + i] * xw)

    error = y - a[0] - sum(products)

    if yl is not None:
        error -= a[num_regressors] * yl

    SSR = sum(error**2)

    convergence_param = 0.01  # Small penalty on theta params for convergence

    objective_value = (
        SSR
        + lambda_1 * np.sum(np.array(a[1 : num_regressors + 1]) ** 2)
        + convergence_param * sum(np.array(theta_params) ** 2)
    )

    return objective_value


def jacobian_generalized(a, X, y, yl, weight_methods):
    """
    Computes the Jacobian matrix for the MIDAS model given current parameter estimates.

    Parameters:
        a (np.ndarray): Current parameters (coefficients + polynomial params)
        X (np.ndarray): High-frequency data
        y (np.ndarray): Dependent variable
        yl (np.ndarray or None): Lagged dependent variable
        weight_methods (list): Polynomial weighting method instances

    Returns:
        jacobian (np.ndarray): Jacobian matrix of residuals
    """
    num_regressors = len(weight_methods)
    jwx_all = []
    weighted_regressors = []

    for i in range(num_regressors):
        theta_start = 1 + i * 2 + num_regressors
        theta_end = theta_start + 2
        theta_i = a[theta_start:theta_end]

        # Weighted regressors
        xw, _ = weight_methods[i].x_weighted(X[:, i * 3 : (i + 1) * 3], theta_i)
        weighted_regressors.append(xw)

        # Jacobian of weighted regressors w.r.t theta
        jwx = jacobian_wx(X[:, i * 3 : (i + 1) * 3], theta_i)
        jwx_all.append(jwx)

    # Error vector
    error = y - a[0] - sum(a[1 + i] * weighted_regressors[i] for i in range(num_regressors))

    if yl is not None:
        error -= a[num_regressors] * yl

    n = len(error)
    num_params = len(a)
    jac = np.zeros((n, num_params))

    # Partial derivative w.r.t intercept
    jac[:, 0] = -1

    # Partial derivatives w.r.t slope coefficients
    for i in range(num_regressors):
        jac[:, 1 + i] = -weighted_regressors[i]

    # Partial derivatives w.r.t polynomial parameters
    for i in range(num_regressors):
        idx = 1 + num_regressors + 2 * i
        jac[:, idx : idx + 2] = -a[1 + i] * jwx_all[i]

    # Partial derivatives w.r.t lag parameters (if any)
    if yl is not None:
        jac[:, num_regressors] = -yl

    return jac
