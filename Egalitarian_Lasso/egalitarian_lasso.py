"""
Income Prediction Models with Lasso, Ridge, ElasticNet, RMSE Weighting, and Diebold-Mariano Test

This script:
- Loads a GDP dataset
- Applies Lasso for feature selection
- Creates subsets of features based on selected coefficients
- Calculates weights and weighted predictions using RMSE
- Tunes hyperparameters with cross-validation for Lasso, Ridge, and ElasticNet
- Fits various step 2 models using transformed targets
- Calculates RMSE for all models
- Performs Diebold-Mariano tests to compare model forecast accuracy

Note: Requires 'dm_test.py' with the 'dm_test' function available.
"""

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from dm_test import dm_test
import matplotlib.pyplot as plt
from google.colab import files


def cross_validate_lasso(X, y, lambdas, n_splits=5):
    """
    Cross-validate Lasso regression over a range of lambdas.
    Returns the lambda with the lowest average MSE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Lasso(alpha=alpha, max_iter=1000000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))

        average_mse_scores.append(np.mean(mse_scores))

    min_mse_index = np.argmin(average_mse_scores)
    return lambdas[min_mse_index]


def cross_validate_ridge(X, y, lambdas, n_splits=5):
    """
    Cross-validate Ridge regression over a range of lambdas.
    Returns the lambda with the lowest average MSE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Ridge(alpha=alpha, max_iter=1000000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))

        average_mse_scores.append(np.mean(mse_scores))

    min_mse_index = np.argmin(average_mse_scores)
    return lambdas[min_mse_index]


def cross_validate_elastic_net(X, y, lambdas, n_splits=5):
    """
    Cross-validate ElasticNet regression over a range of lambdas.
    Returns the lambda with the lowest average MSE.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    average_mse_scores = []

    for alpha in lambdas:
        mse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = ElasticNet(alpha=alpha, max_iter=1000000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))

        average_mse_scores.append(np.mean(mse_scores))

    min_mse_index = np.argmin(average_mse_scores)
    return lambdas[min_mse_index]


def main():
    # Upload file (Google Colab)
    uploaded = files.upload()

    # Load dataset
    data = pd.read_csv('/content/H1_gdp.csv')

    target = data['Actual'].values
    features = data.loc[:, 'ID_1':'ID_95'].values
    T, K = features.shape

    # Step 1: Lasso for feature selection
    lambdas = np.logspace(-5, 2, 100)
    optimal_lambda = cross_validate_lasso(features, target, lambdas)
    print(f"Optimal lambda for Lasso: {optimal_lambda:.5f}")

    lasso = Lasso(alpha=optimal_lambda, max_iter=1000000)
    lasso.fit(features, target)

    coef = lasso.coef_
    selected_indices = np.nonzero(coef)[0]
    print(f"Selected feature indices: {selected_indices}")

    # Subset X with selected indices - here hardcoded indices from original script replaced with selected_indices
    # For your example, you had indices_to_keep = [2, 9, 21]
    # Using selected indices instead:
    indices_to_keep = selected_indices
    X_subset = features[:, indices_to_keep]

    # Calculate average of selected subset features for transformation
    X_bar_subset = np.mean(X_subset, axis=1)
    Y_transform = target - X_bar_subset

    # RMSE weighting to create weights for the selected subset features
    c = 1  # Constant for weighting, can be tuned

    rmse = np.zeros(len(indices_to_keep))
    for i in range(len(indices_to_keep)):
        rmse[i] = math.sqrt(np.mean((X_subset[:, i] - target) ** 2))

    denominator = np.sum(np.exp(c / rmse))
    weights = np.exp(c / rmse) / denominator

    # Weighted sum of features
    weighted_features = np.zeros(T)
    for i in range(len(indices_to_keep)):
        weighted_features += weights[i] * X_subset[:, i]

    Y_weighted_transform = target - weighted_features

    # Prepare lambdas for step 2 tuning
    lambdas_step2 = lambdas  # reuse same range or define new

    # Step 2: Tune ElasticNet on subset features and transformed target
    optimal_lambda_step2 = cross_validate_elastic_net(X_subset, target, lambdas_step2)
    print(f"Optimal lambda for ElasticNet step 2: {optimal_lambda_step2:.5f}")

    # Step 2 Models: ElasticNet, Ridge, Lasso on transformed targets
    # Using Y_transform = target - X_bar_subset
    # Using Y_weighted_transform for RMSE weighted models

    # ElasticNet with transformed target
    model_en = ElasticNet(alpha=optimal_lambda_step2, max_iter=1000000)
    model_en.fit(X_subset, Y_transform)
    coef_en = model_en.coef_
    coef_en_adjusted = coef_en + 1 / len(indices_to_keep)
    prediction_en = X_subset @ coef_en_adjusted
    rmse_en = mean_squared_error(target, prediction_en, squared=False)
    print(f"RMSE ElasticNet step 2 (transformed): {rmse_en:.5f}")

    # Ridge with transformed target
    model_ridge = Ridge(alpha=optimal_lambda_step2, max_iter=1000000)
    model_ridge.fit(X_subset, Y_transform)
    coef_ridge = model_ridge.coef_
    coef_ridge_adjusted = coef_ridge + 1 / len(indices_to_keep)
    prediction_ridge = X_subset @ coef_ridge_adjusted
    rmse_ridge = mean_squared_error(target, prediction_ridge, squared=False)
    print(f"RMSE Ridge step 2 (transformed): {rmse_ridge:.5f}")

    # Lasso with transformed target
    model_lasso = Lasso(alpha=optimal_lambda_step2, max_iter=1000000)
    model_lasso.fit(X_subset, Y_transform)
    coef_lasso = model_lasso.coef_
    coef_lasso_adjusted = coef_lasso + 1 / len(indices_to_keep)
    prediction_lasso = X_subset @ coef_lasso_adjusted
    rmse_lasso = mean_squared_error(target, prediction_lasso, squared=False)
    print(f"RMSE Lasso step 2 (transformed): {rmse_lasso:.5f}")

    # RMSE weighted models - step 2 on weighted transform target
    # ElasticNet RMSE weighted
    model_en_rmse = ElasticNet(alpha=optimal_lambda_step2, max_iter=1000000)
    model_en_rmse.fit(X_subset, Y_weighted_transform)
    coef_en_rmse = model_en_rmse.coef_
    prediction_en_rmse = X_subset @ coef_en_rmse
    rmse_en_rmse = mean_squared_error(target, prediction_en_rmse, squared=False)
    print(f"RMSE ElasticNet step 2 (RMSE weighted): {rmse_en_rmse:.5f}")

    # Ridge RMSE weighted
    model_ridge_rmse = Ridge(alpha=optimal_lambda_step2, max_iter=1000000)
    model_ridge_rmse.fit(X_subset, Y_weighted_transform)
    coef_ridge_rmse = model_ridge_rmse.coef_
    prediction_ridge_rmse = X_subset @ coef_ridge_rmse
    rmse_ridge_rmse = mean_squared_error(target, prediction_ridge_rmse, squared=False)
    print(f"RMSE Ridge step 2 (RMSE weighted): {rmse_ridge_rmse:.5f}")

    # Lasso RMSE weighted
    model_lasso_rmse = Lasso(alpha=optimal_lambda_step2, max_iter=1000000)
    model_lasso_rmse.fit(X_subset, Y_weighted_transform)
    coef_lasso_rmse = model_lasso_rmse.coef_
    prediction_lasso_rmse = X_subset @ coef_lasso_rmse
    rmse_lasso_rmse = mean_squared_error(target, prediction_lasso_rmse, squared=False)
    print(f"RMSE Lasso step 2 (RMSE weighted): {rmse_lasso_rmse:.5f}")

    # Simple average baseline RMSE
    rmse_simple_average = mean_squared_error(target, X_bar_subset, squared=False)
    print(f"RMSE simple average baseline: {rmse_simple_average:.5f}")

    # Diebold-Mariano tests comparing predictions
    # Example: compare ElasticNet RMSE weighted vs simple average baseline
    h = 1  # forecast horizon for DM test

    # Assuming dm_test(Y, pred1, pred2, h, crit) where crit is string (e.g. 'MSE' or 'MAE')
    dm_result_en_vs_avg = dm_test(target, prediction_en_rmse, X_bar_subset, h=h, crit='MSE')
    print(f"DM test result ElasticNet RMSE vs Simple Average: {dm_result_en_vs_avg}")

    # Compare other pairs as needed:
    dm_result_lasso_vs_avg = dm_test(target, prediction_lasso_rmse, X_bar_subset, h=h, crit='MSE')
    print(f"DM test result Lasso RMSE vs Simple Average: {dm_result_lasso_vs_avg}")

    dm_result_ridge_vs_avg = dm_test(target, prediction_ridge_rmse, X_bar_subset, h=h, crit='MSE')
    print(f"DM test result Ridge RMSE vs Simple Average: {dm_result_ridge_vs_avg}")


if __name__ == "__main__":
    main()
