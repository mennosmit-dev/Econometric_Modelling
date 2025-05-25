"""
This module contains the main program for nowcasting GDP using an AR(1) model.
"""

import numpy as np
import pandas as pd

# Initial Data Settings
FIRST_MONTH_IN_DATA = 241  # First month in training data (1980-01)
START_MONTH = 601          # First out-of-sample vintage monthly data (2010-01)
END_MONTH = 772            # Last out-of-sample vintage monthly data (2024-04)
OOS_OBS = END_MONTH - START_MONTH  # Number of out-of-sample observations
USE_VINTAGE_DATA = True    # Whether to use vintage data or only last vintage

# Prepare the last transformed quarterly vintage data to calculate errors later
prepare_current_data(END_MONTH, FIRST_MONTH_IN_DATA, USE_VINTAGE_DATA)
last_vintage_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

# Initialize current month and error storage array
current_month = START_MONTH
ar1_errors = np.zeros((OOS_OBS, 1))


def get_last_nonzero_element(df: pd.DataFrame) -> pd.Series:
    """
    Finds the last non-zero (non-NaN) element in a DataFrame column-wise.

    Parameters:
        df (pd.DataFrame): DataFrame from which to find the last nonzero element.

    Returns:
        pd.Series: The last nonzero row (as a Series).
    """
    for i in range(1, len(df) + 1):
        row = df.iloc[-i, :].dropna()
        if not row.empty:
            return row
    return pd.Series(dtype=float)


# Loop over all out-of-sample months and estimate AR(1) errors
for t in range(OOS_OBS):
    current_month = START_MONTH + t

    # Prepare data for the current month
    prepare_current_data(current_month, FIRST_MONTH_IN_DATA, USE_VINTAGE_DATA)
    current_data_quarterly = pd.read_csv("/content/Transformed_Quarterly_Data.csv", header=None)

    # Calculate quarters to go for actual GDP value to predict
    quarters_to_go = last_vintage_data_quarterly.shape[0] - current_data_quarterly.shape[0]

    # Access the actual GDP data to predict
    actual_data = last_vintage_data_quarterly.iloc[last_vintage_data_quarterly.shape[0] - quarters_to_go]

    # Get the last nonzero element for AR(1) prediction
    last_element = get_last_nonzero_element(current_data_quarterly)

    # Calculate AR(1) prediction error for this month
    ar1_errors[t] = actual_data - last_element

# Store AR(1) prediction errors to CSV file
ar1_errors_df = pd.DataFrame(ar1_errors)
ar1_errors_df.to_csv('/content/AR(1)_errors.csv', index=False, header=False)
