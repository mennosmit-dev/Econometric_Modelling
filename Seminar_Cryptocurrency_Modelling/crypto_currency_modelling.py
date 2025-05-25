"""
This file is used to trade BTC and ETH futures based on econometric equations 
that should theoretically hold.

For the details, please refer to the accompanying PDF in which everything 
is discussed in detail.
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import warnings
import statsmodels.api as sm
from scipy.stats import t

warnings.filterwarnings("ignore")


def reformat_date_column(df, column_name):
    """Reformat date from 'ETH_16JUN23' to 'ETH_230616' style."""
    def reformat_value(value):
        try:
            prefix, date_part = value.split('_', 1)
            date_obj = datetime.strptime(date_part, '%d%b%y')
            return f"{prefix}_{date_obj.strftime('%y%m%d')}"
        except (ValueError, IndexError):
            return value

    df[column_name] = df[column_name].apply(reformat_value)
    return df


def reformat_date_column2(df, column_name):
    """Replace dashes with underscores for date-formatted strings."""
    def reformat_value(value):
        if isinstance(value, str) and value.count('-') == 2:
            return value.replace('-', '_')
        return value

    df[column_name] = df[column_name].apply(reformat_value)
    return df


def calculate_tenor(df, column1, column2):
    """Calculate the time difference between contract expiration and timestamp."""
    cet = pytz.timezone('CET')

    def parse_column1(value):
        try:
            date_str = value[-6:]
            date_obj = datetime.strptime(date_str, '%y%m%d')
            return cet.localize(datetime.combine(date_obj.date(), datetime.min.time()) + timedelta(hours=9))
        except ValueError:
            return None

    def parse_column2(value):
        try:
            dt = pd.to_datetime(value, errors='coerce')
            if dt.tzinfo is None:
                return dt.tz_localize('UTC').tz_convert(cet)
            return dt.tz_convert(cet)
        except Exception:
            return None

    df['column1_datetime'] = df[column1].apply(parse_column1)
    df['column2_datetime'] = df[column2].apply(parse_column2)
    df['tenor'] = df['column1_datetime'] - df['column2_datetime']
    df.drop(columns=['column1_datetime', 'column2_datetime'], inplace=True)
    return df


def precompute_spot_prices_last(reference_df, start_time, end_time):
    """Precompute hourly last spot prices from reference data."""
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
    spot_prices = {}

    for i in range(len(timestamps) - 1):
        start_interval = timestamps[i]
        end_interval = timestamps[i + 1]

        interval_values = reference_df[
            (reference_df['dateTime_round_s'] >= start_interval) &
            (reference_df['dateTime_round_s'] < end_interval)
        ]

        if not interval_values.empty:
            last_point = interval_values.iloc[-1]
            spot_prices[end_interval] = last_point['mid_Px']
        else:
            spot_prices[end_interval] = np.nan

    return spot_prices


def precompute_spot_prices_average(reference_df, start_time, end_time):
    """Precompute hourly average spot prices from reference data."""
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
    spot_prices = {}

    for i in range(len(timestamps) - 1):
        start_interval = timestamps[i]
        end_interval = timestamps[i + 1]
        midpoint = (start_interval + (end_interval - start_interval) / 2).replace(second=0)

        interval_values = reference_df[
            (reference_df['dateTime_round_s'] >= start_interval) &
            (reference_df['dateTime_round_s'] < end_interval)
        ]

        spot_prices[midpoint] = interval_values['mid_Px'].mean() if not interval_values.empty else np.nan

    return spot_prices


# Load and preprocess files
df1 = pd.read_parquet("C:/Users/EfeKa/OneDrive/Bureaublad/Seminar/merged_batch_1.parquet")
df2 = pd.read_parquet("C:/Users/EfeKa/OneDrive/Bureaublad/Seminar/merged_batch_2.parquet")

if list(df1.columns) != list(df2.columns):
    raise ValueError("Column names or order do not match between the files.")

df = pd.concat([df1, df2], ignore_index=True)
print('files merged')

df['symbol'] = df['symbol'].str.replace('-', '_')
df['spread bps'] = 10000 * (df['ask_price'] - df['bid_price']) / df['mid_Px']
df['liquidity prox'] = df['ask_price'] * df['ask_amount'] + df['bid_price'] * df['bid_amount']

btc_df = df[df['symbol'].str.contains('BTC')]
eth_df = df[df['symbol'].str.contains('ETH')]
print('files split into btc/eth')

btc_binance = btc_df[btc_df['exchange'] == 'binance']
btc_okex = btc_df[btc_df['exchange'] == 'okex']
btc_deribit = reformat_date_column(btc_df[btc_df['exchange'] == 'deribit'], 'symbol')

eth_binance = eth_df[eth_df['exchange'] == 'binance']
eth_okex = eth_df[eth_df['exchange'] == 'okex']
eth_deribit = reformat_date_column(eth_df[eth_df['exchange'] == 'deribit'], 'symbol')

exchange_names = {
    'btc_binance': btc_binance,
    'btc_okex': btc_okex,
    'btc_deribit': btc_deribit,
    'eth_binance': eth_binance,
    'eth_okex': eth_okex,
    'eth_deribit': eth_deribit
}

print('files split into exchanges')

all_files = []

for name, file in exchange_names.items():
    unique_elements = file['symbol'].unique()
    for element in unique_elements:
        new_df_name = f"{name}_{element}"
        all_files.append((new_df_name, file[file['symbol'] == element]))
        globals()[new_df_name] = file[file['symbol'] == element]

print('files split into instruments')

start_time = pd.Timestamp('2023-06-13 00:00:00')
end_time = pd.Timestamp('2023-07-12 00:00:00')

btc_spot_prices_mid = precompute_spot_prices_average(btc_binance_BTCUSDT, start_time, end_time)
eth_spot_prices_mid = precompute_spot_prices_average(eth_binance_ETHUSDT, start_time, end_time)

btc_spot_series_mid = pd.Series(btc_spot_prices_mid)
eth_spot_series_mid = pd.Series(eth_spot_prices_mid)

btc_spot_prices_last = precompute_spot_prices_last(btc_binance_BTCUSDT, start_time, end_time)
eth_spot_prices_last = precompute_spot_prices_last(eth_binance_ETHUSDT, start_time, end_time)

btc_spot_series_last = pd.Series(btc_spot_prices_last)
eth_spot_series_last = pd.Series(eth_spot_prices_last)


interval = '1H'
all_combined_files = []


def aggregate_by_interval(df, interval='1H'):
    """Aggregate trading data by hourly intervals (averages)."""
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df['interval_start'] = df['dateTime'].dt.floor(interval) + pd.to_timedelta(1, unit='h')

    aggregated = df.groupby('interval_start').agg({
        'ask_amount': 'sum',
        'bid_amount': 'sum',
        'ask_price': 'mean',
        'bid_price': 'mean',
        'mid_Px': 'mean',
        'spread bps': 'mean',
        'liquidity prox': 'mean'
    }).reset_index()

    aggregated.rename(columns={'interval_start': 'hour_end'}, inplace=True)
    aggregated = aggregated.rename(columns={
        'ask_amount': 'ask_amount_avg',
        'bid_amount': 'bid_amount_avg',
        'ask_price': 'ask_price_avg',
        'bid_price': 'bid_price_avg',
        'mid_Px': 'mid_Px_avg',
        'spread bps': 'spread_bps_avg',
        'liquidity prox': 'liquidity_prox_avg'
    })

    return aggregated


def aggregate_last_available_by_hour(df):
    """Aggregate trading data by last available point per hour."""
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df = df.sort_values(by='dateTime')
    df['interval_start'] = df['dateTime'].dt.floor(interval) + pd.to_timedelta(1, unit='h')

    last_points = df.groupby('interval_start').last().reset_index()
    last_points.rename(columns={'interval_start': 'hour_end'}, inplace=True)
    last_points = last_points.rename(columns={
        'ask_amount': 'ask_amount_end',
        'bid_amount': 'bid_amount_end',
        'ask_price': 'ask_price_end',
        'bid_price': 'bid_price_end',
        'mid_Px': 'mid_Px_end',
        'spread bps': 'spread_bps_end',
        'liquidity prox': 'liquidity_prox_end'
    })

    return last_points


for f in all_files:
    file_name, df = f
    last_points_df = aggregate_last_available_by_hour(df)
    aggregated_df = aggregate_by_interval(df, interval)
    combined_df = last_points_df.merge(aggregated_df, on='hour_end', suffixes=('_end', '_avg'))
    combined_df['symbol'] = df['symbol'].values[0]
    combined_df['exchange'] = df['exchange'].values[0]
    all_combined_files.append((f"last_point_{file_name}", combined_df))

    if not (file_name.endswith("USDT") or "SWAP" in file_name or "PERP" in file_name):
        calculate_tenor(combined_df, 'symbol', 'hour_end')


def estimate_regression(
    data_last,
    method='OLS',
    use_indicator=True,
    use_spread=True,
    use_liquidity=True,
    use_midpx_avg=False,
    t_value=2 / 3
):
    """
    Estimate regression coefficients for future premium vs tenor.

    Parameters
    ----------
    data_last : np.ndarray
        Array of shape (n_samples, features) where columns include tenor, future premium,
        spread, liquidity, and mid-price average as applicable.
    method : str, optional
        Regression method to use ('OLS', 'WLS', 'GLS', 'RLS'). Default is 'OLS'.
    use_indicator : bool, optional
        Whether to include an indicator variable based on tenor cutoff `t_value`.
    use_spread : bool, optional
        Whether to include spread as a regressor.
    use_liquidity : bool, optional
        Whether to include liquidity as a regressor.
    use_midpx_avg : bool, optional
        Whether to include mid-price average as a regressor.
    t_value : float, optional
        Threshold for the indicator variable, default is 2/3.

    Returns
    -------
    dict
        Regression results including slope, intercept, optional coefficients, residuals,
        and design matrix.
    """
    data_last = np.asarray(data_last, dtype=np.float64)
    x = data_last[:, 0]  # Tenor
    y = data_last[:, 1]  # Future premium
    n = len(x)

    # Design matrix with intercept and tenor
    X = np.column_stack((np.ones(n), x))

    # Add indicator variable if specified
    if use_indicator:
        if t_value is None:
            raise ValueError("t_value must be provided when use_indicator=True")
        indicator = (x <= t_value).astype(np.float64) * x
        X = np.column_stack([X, indicator])

    if use_midpx_avg:
        mid_px_avg = data_last[:, -1]
        X = np.column_stack([X, mid_px_avg])

    if use_liquidity:
        liquidity = data_last[:, -2]
        X = np.column_stack([X, liquidity])

    # Add spread variable if specified
    if use_spread and data_last.shape[1] > 2:
        spread = data_last[:, -3]
        X = np.column_stack([X, spread])

    # Choose regression method
    if method == 'OLS':
        model = sm.OLS(y, X).fit()
    elif method == 'WLS':
        ols_model = sm.OLS(y, X).fit()
        residuals_ols = ols_model.resid
        weights = 1 / (np.abs(residuals_ols) + 1e-6)
        model = sm.WLS(y, X, weights=weights).fit()
    elif method == 'GLS':
        ols_model = sm.OLS(y, X).fit()
        residuals_ols = ols_model.resid
        weights = 1 / (np.abs(residuals_ols) + 1e-6)  # Approximate heteroskedasticity
        sigma = np.diag(weights)
        model = sm.GLS(y, X, sigma=sigma).fit()
    elif method == 'RLS':
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    else:
        raise ValueError("Invalid method. Choose 'OLS', 'WLS', 'RLS', or 'GLS'")

    # Extract results
    beta = model.params
    residuals = model.resid

    results = {'slope': beta[1]}
    results['intercept'] = beta[0]
    index = 2  # Start from beta[2]

    if use_indicator:
        results['indicator'] = beta[index]
        index += 1

    if use_spread:
        results['spread'] = beta[index]
        index += 1

    if use_liquidity:
        results['liquidity'] = beta[index]
        index += 1

    if use_midpx_avg:
        results['Mid Price'] = beta[index]

    results['residuals'] = residuals
    results['design_matrix'] = X

    return results


# Initialize lists for storing results
time_series_data_D = []
detailed_rows_D = []

# Reset and rename spot series for ETH and BTC
ethspot = eth_spot_series_last.reset_index()
ethspot.columns = ['time', 'spot']

btcspot = btc_spot_series_last.reset_index()
btcspot.columns = ['time', 'spot']

start_time = pd.Timestamp('2023-06-13 01:00:00')
end_time = pd.Timestamp('2023-07-11 01:00:00')

# Iterate over time in hourly steps
current_time = start_time
while current_time <= end_time:
    data_pairs = []
    for idx, (df_name, df_x) in enumerate(all_combined_files):

        if df_name.endswith("USDT") or "SWAP" in df_name or "PERP" in df_name:
            continue

        df_x['hour_end'] = pd.to_datetime(df_x['hour_end'])
        row = df_x[df_x['hour_end'] == current_time]

        if 'BTC' in df_name:
            spot_series, color = btcspot, 'orange'
        elif 'ETH' in df_name:
            spot_series, color = ethspot, 'blue'
        else:
            continue

        if row.empty:
            continue

        tenor_value = row['tenor'].iloc[0]
        px_mid_value = row['mid_Px_end'].iloc[0]
        log_px_mid_value = np.log(px_mid_value)
        spot = float(spot_series[spot_series['time'] == pd.Timestamp(current_time)]['spot'])
        future_premium = np.log(px_mid_value / spot)
        ask_price = row['ask_price_end'].iloc[0]
        bid_price = row['bid_price_end'].iloc[0]
        liquidity = row['liquidity_prox_avg'].iloc[0]
        avg_midprice = row['mid_Px_avg'].iloc[0]
        spread = row['spread_bps_avg'].iloc[0]

        if pd.isna(tenor_value):
            tenor_value = 0
            continue

        if isinstance(tenor_value, pd.Timedelta):
            if tenor_value < pd.Timedelta(days=1):
                continue

        if pd.isna(tenor_value) or pd.isna(future_premium):
            continue

        if isinstance(tenor_value, pd.Timedelta):
            tenor_value = tenor_value.total_seconds() / (365.25 * 24 * 3600)

        data_pairs.append([
            tenor_value,
            future_premium,
            color,
            df_name,
            px_mid_value,
            ask_price,
            bid_price,
            spread,
            liquidity,
            avg_midprice
        ])

    matrix = np.array(data_pairs, dtype=object)

    if len(data_pairs) > 14:
        tenors = matrix[:, 0].astype(np.float64)
        future_premium = matrix[:, 1].astype(np.float64)
        colors = matrix[:, 2]
        filenames = matrix[:, 3]
        mid_prices = matrix[:, 4].astype(np.float64)
        ask_price = matrix[:, 5].astype(np.float64)
        bid_price = matrix[:, 6].astype(np.float64)
        spread = matrix[:, 7].astype(np.float64)
        liquidity = matrix[:, 8].astype(np.float64)
        avg_midprice = matrix[:, 9].astype(np.float64)

        valid_data = [i for i in range(len(tenors)) if pd.notna(tenors[i]) and pd.notna(mid_prices[i])]

        tenors_valid = tenors[valid_data]
        premium_valid = future_premium[valid_data]
        colors_valid = colors[valid_data]
        filenames_valid = filenames[valid_data]
        mid_prices_valid = mid_prices[valid_data]
        ask_price = ask_price[valid_data]
        bid_price = bid_price[valid_data]
        spread = spread[valid_data]
        liquidity = liquidity[valid_data]
        avg_midprice = avg_midprice[valid_data]

        valid_matrix = np.column_stack(
            (tenors_valid, future_premium, spread, liquidity, avg_midprice)
        ).astype(np.float64)

        coeffs_and_pvalues = estimate_regression(valid_matrix)

        slope = coeffs_and_pvalues['slope']
        intercept = coeffs_and_pvalues['intercept']
        residuals = coeffs_and_pvalues['residuals']

        X = coeffs_and_pvalues['design_matrix']

        # Commented out plotting code (optional)
        '''
        plt.figure(figsize=(8, 6))
        for i in range(len(tenors_valid)):
            plt.scatter(tenors_valid[i], premium_valid[i], color=colors_valid[i], label="")
        # Plot fitted regression line
        x_fit = np.linspace(min(tenors_valid), max(tenors_valid), 100)
        y_fit = sum(beta[i] * x_fit**(degree - i) for i in range(degree)) + intercept
        plt.plot(x_fit, y_fit, color='black', linestyle='dashed', label='Fitted Line')
        plt.xlabel("Tenor")
        plt.ylabel("Future Premium (log)")
        plt.title(f"Regression Fit for {current_time}")
        plt.legend()
        plt.show()
        '''

        for i in range(len(tenors_valid)):
            detailed_rows_D.append({
                "timestamp": current_time,
                "df_filename": filenames_valid[i],
                "Tenor": tenors_valid[i],
                "Mid price": mid_prices_valid[i],
                "log(premium)": future_premium[i],
                "slope": slope,
                "residual": residuals[i],
                "Bid price": bid_price[i],
                "Ask price": ask_price[i],
                "Liquidity": liquidity[i]
            })

        time_series_data_D.append({"timestamp": current_time, **coeffs_and_pvalues})

    current_time += pd.Timedelta(seconds=3600)

detailed_df_D = pd.DataFrame(detailed_rows_D)
time_series_df_D = pd.DataFrame(time_series_data_D)


# PnL Benchmark Directional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2):
    """
    Calculate PnL from the given DataFrame with benchmark directional strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trading data.
    start_capital : int, optional
        Initial capital for PnL calculation. Default is 1,000,000.
    plot : bool, optional
        Whether to plot the PnL graph. Default is True.
    n : int, optional
        Parameter affecting calculation. Usage depends on context.

    Returns
    -------
    tuple of pd.DataFrame
        Returns two DataFrames representing detailed PnL and summary stats.
    """
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    required_columns = {'df_filename', 'timestamp', 'Tenor'}
    if not required_columns.issubset(df.columns):
        logging.error(
            "Missing required columns in DataFrame. "
            "Ensure 'df_filename', 'timestamp', and 'Tenor' exist."
        )
        return pd.DataFrame(), pd.DataFrame()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging


def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2):
    """
    Calculate realized PnL based on residuals in the DataFrame and track positions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing columns like 'timestamp', 'df_filename', 'residual',
        'Ask price', 'Bid price', 'Tenor', etc.
    start_capital : float, optional
        Starting capital to allocate for trades (default is 1,000,000).
    plot : bool, optional
        Whether to plot the realized PnL over time (default is True).
    n : int, optional
        Number of long and short positions to open each period (default is 2).

    Returns
    -------
    trade_df : pd.DataFrame
        DataFrame logging each executed trade.
    pnl_df : pd.DataFrame
        DataFrame containing timestamps and realized PnL values.
    position_tracker : pd.DataFrame
        DataFrame tracking current open positions with exposures and prices.
    """

    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_cols = {'df_filename', 'timestamp', 'Tenor'}
    if not required_cols.issubset(df.columns):
        logging.error(f"Missing required columns in DataFrame: {required_cols - set(df.columns)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for current_time in timestamps:
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['residual']
        to_close = []

        # Reset unrealized PnL
        unrealized_pnl = 0

        # Calculate current exposure sums
        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions if residual or tenor conditions are met
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            close_long = details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)
            close_short = details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)

            if close_long or close_short:
                pnl = ((current_price - weighted_avg_price) * quantity if details['position'] == 'long'
                       else (weighted_avg_price - current_price) * quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions
        for product in to_close:
            balances.pop(product, None)
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() +
                                                 residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions (most underpriced)
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = exposure / row['Ask price']

            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue

            total_long_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'long',
                'quantity': quantity,
                'entry_price': row['Ask price'],
                'dollar_exposure': exposure,
                'weighted_avg_price': row['Ask price']
            }
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) +
                                 (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'],
                                     ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = (
                    new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure)
            else:
                new_row = pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity,
                                         row['Ask price'], quantity]],
                                       columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        # Handle short positions (most overpriced)
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = -exposure / row['Bid price']

            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue

            total_short_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'short',
                'quantity': quantity,
                'entry_price': row['Bid price'],
                'dollar_exposure': -exposure,
                'weighted_avg_price': row['Bid price']
            }
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) +
                                 (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'],
                                     ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = (
                    new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure)
            else:
                new_row = pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity,
                                         row['Bid price'], quantity]],
                                       columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        pnl_tracker.append([current_time, realized_pnl])

    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return trade_df, pnl_df, position_tracker


import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_pnL_ew(df, start_capital=1_000_000, plot=True, n=2, threshold=200 / 10000):
    """
    Calculate realized PnL using an Equal Weight (EW) strategy based on adjusted residuals.

    Positions are opened and closed depending on residuals and tenor.
    Exposure is equally distributed across selected products rather than weighted by residuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'df_filename', 'timestamp', 'Tenor', 'adjusted_residual',
        'Ask price', and 'Bid price'.
    start_capital : float, optional
        Initial capital for allocation (default is 1,000,000).
    plot : bool, optional
        Whether to plot the realized PnL over time (default is True).
    n : int, optional
        Number of top and bottom residuals to select (default is 2).
    threshold : float, optional
        Minimum absolute adjusted residual for considering trades (default is 0.02).

    Returns
    -------
    trade_df : pd.DataFrame
        DataFrame of executed trades with columns ['timestamp', 'product', 'amount_available', 'position', 'price'].
    pnl_df : pd.DataFrame
        DataFrame of realized PnL over time with columns ['timestamp', 'PnL'].
    position_tracker : pd.DataFrame
        Current open positions with columns ['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'].
    """
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_cols = ['df_filename', 'timestamp', 'Tenor', 'adjusted_residual', 'Ask price', 'Bid price']
    if not all(col in df.columns for col in required_cols):
        logging.error(
            f"Missing required columns in DataFrame. "
            f"Ensure these columns exist: {required_cols}."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(
        columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity']
    )
    timestamps = sorted(df['timestamp'].unique())

    for current_time in timestamps:
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Calculate current exposures
        total_long_exposure = sum(
            details['dollar_exposure']
            for details in balances.values()
            if details['position'] == 'long'
        )
        total_short_exposure = sum(
            abs(details['dollar_exposure'])
            for details in balances.values()
            if details['position'] == 'short'
        )

        # Close positions if conditions met
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = (
                product_slice['Bid price'].iloc[0]
                if details['position'] == 'short'
                else product_slice['Ask price'].iloc[0]
            )
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1 / 365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1 / 365)):
                pnl = (
                    (current_price - weighted_avg_price) * quantity
                    if details['position'] == 'long'
                    else (weighted_avg_price - current_price) * quantity
                )
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions
        for product in to_close:
            balances.pop(product, None)
            position_tracker = position_tracker[position_tracker['product'] != product]

        # Select top n over- and under-priced products
        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        # Equal weighting: divide capital equally among longs and shorts
        long_exposure_per_asset = start_capital / n
        short_exposure_per_asset = start_capital / n

        # Handle long positions (most underpriced)
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = long_exposure_per_asset
            quantity = exposure / row['Ask price']

            existing_long = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long.empty:
                total_long_exposure += existing_long['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue

            total_long_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'long',
                'quantity': quantity,
                'entry_price': row['Ask price'],
                'dollar_exposure': exposure,
                'weighted_avg_price': row['Ask price'],
            }
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            if not existing_long.empty:
                new_position = existing_long['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing_long['net_position'].values[0] * existing_long['weighted_avg_price'].values[0]) +
                    (quantity * row['Ask price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing_long['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                new_row = pd.DataFrame([[
                    row['df_filename'], exposure, quantity, row['Ask price'], quantity
                ]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        # Handle short positions (most overpriced)
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = short_exposure_per_asset
            quantity = -exposure / row['Bid price']

            existing_short = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short.empty:
                total_short_exposure -= existing_short['dollar_exposure'].sum()

            if abs(total_short_exposure + exposure) > 1_000_000:
                continue

            total_short_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'short',
                'quantity': quantity,
                'entry_price': row['Bid price'],
                'dollar_exposure': -exposure,
                'weighted_avg_price': row['Bid price'],
            }
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            if not existing_short.empty:
                new_position = existing_short['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing_short['net_position'].values[0] * existing_short['weighted_avg_price'].values[0]) +
                    (quantity * row['Bid price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing_short['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                new_row = pd.DataFrame([[
                    row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity
                ]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        pnl_tracker.append([current_time, realized_pnl])

    trade_df = pd.DataFrame(
        trade_records, columns=["timestamp", "product", "amount_available", "position", "price"]
    )
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time (Equal Weight)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return trade_df, pnl_df, position_tracker


    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging


def calculate_pnl(df, start_capital=1_000_000, plot=True, n=2, threshold=0):
    """
    Calculate realized PnL and track trades and positions over time based on adjusted residuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trading data. Must include columns:
        ['timestamp', 'df_filename', 'adjusted_residual', 'Bid price', 'Ask price', 'Tenor'].
    start_capital : float, optional
        Initial capital for trading exposure (default is 1,000,000).
    plot : bool, optional
        Whether to plot the realized PnL over time (default is True).
    n : int, optional
        Number of top and bottom residuals to consider for opening positions (default is 2).
    threshold : float, optional
        Minimum absolute residual threshold to consider trades (default is 0).

    Returns
    -------
    trade_df : pd.DataFrame
        DataFrame with trade records including timestamp, product, amount, position, and price.
    pnl_df : pd.DataFrame
        DataFrame with timestamps and cumulative realized PnL.
    position_tracker : pd.DataFrame
        DataFrame tracking current open positions with product details.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty. Exiting calculation.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Ensure required columns exist
    required_cols = {'timestamp', 'df_filename', 'adjusted_residual', 'Bid price', 'Ask price', 'Tenor'}
    if not required_cols.issubset(df.columns):
        logging.error(f"Missing required columns in DataFrame: {required_cols - set(df.columns)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(
        columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity']
    )
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset unrealized PnL at this timestamp
        unrealized_pnl = 0

        total_long_exposure = sum(
            details['dollar_exposure']
            for details in balances.values()
            if details['position'] == 'long'
        )
        total_short_exposure = sum(
            abs(details['dollar_exposure'])
            for details in balances.values()
            if details['position'] == 'short'
        )

        # Close positions when conditions met
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = (
                product_slice['Bid price'].iloc[0]
                if details['position'] == 'short'
                else product_slice['Ask price'].iloc[0]
            )
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            close_long = details['position'] == 'long' and (current_residual > 0 or tenor < 1 / 365)
            close_short = details['position'] == 'short' and (current_residual < 0 or tenor < 1 / 365)

            if close_long or close_short:
                pnl = (
                    (current_price - weighted_avg_price) * quantity
                    if details['position'] == 'long'
                    else (weighted_avg_price - current_price) * quantity
                )
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append(
                    [current_time, product, -quantity, round(quantity, 0), current_price]
                )

        # Remove closed positions from balances and position tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n_limit = min(n, len(residuals) // 2)
        if n_limit == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n_limit:]
        most_underpriced_indices = residuals_sorted_indices[:n_limit]

        total_residual_magnitude = np.sum(
            np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist())
        )
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions (most underpriced)
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_limit)
            quantity = exposure / row['Ask price']  # quantity based on exposure and price

            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip if exposure limits exceeded

            total_long_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'long',
                'quantity': quantity,
                'entry_price': row['Ask price'],
                'dollar_exposure': exposure,
                'weighted_avg_price': row['Ask price'],
            }
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update or add position tracker entry
            if not existing_long_position.empty:
                existing = existing_long_position.iloc[0]
                new_position = existing['net_position'] + quantity
                new_avg_price = (
                    (existing['net_position'] * existing['weighted_avg_price'] + quantity * row['Ask price']) / new_position
                )
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = (
                    new_position, new_avg_price, existing['quantity'] + quantity, new_dollar_exposure
                )
            else:
                new_row = pd.DataFrame(
                    [[row['df_filename'], exposure, quantity, row['Ask price'], quantity]],
                    columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'],
                )
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        # Handle short positions (most overpriced)
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_limit)
            quantity = -exposure / row['Bid price']

            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            if abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip if exposure limits exceeded

            total_short_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'short',
                'quantity': quantity,
                'entry_price': row['Bid price'],
                'dollar_exposure': -exposure,
                'weighted_avg_price': row['Bid price'],
            }
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update or add position tracker entry
            if not existing_short_position.empty:
                existing = existing_short_position.iloc[0]
                new_position = existing['net_position'] + quantity
                new_avg_price = (
                    (existing['net_position'] * existing['weighted_avg_price'] + quantity * row['Bid price']) / new_position
                )
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = (
                    new_position, new_avg_price, existing['quantity'] + quantity, new_dollar_exposure
                )
            else:
                new_row = pd.DataFrame(
                    [[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]],
                    columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'],
                )
                position_tracker = pd.concat([position_tracker, new_row], ignore_index=True)

        pnl_tracker.append([current_time, realized_pnl])

    # Prepare output DataFrames
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return trade_df, pnl_df, position_tracker


    import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_pnl(
    df,
    start_capital=1_000_000,
    plot=True,
    n=2,
    threshold=200 / 10000,
):
    """
    Calculate realized PnL and track positions based on residuals.

    This function iterates over timestamps in the input DataFrame `df`,
    manages long and short positions by residuals, calculates realized PnL,
    and optionally plots the PnL over time.

    Args:
        df (pd.DataFrame): Input DataFrame containing trade data. Must include columns:
            'df_filename', 'timestamp', 'Tenor', 'adjusted_residual', 'Ask price', 'Bid price'.
        start_capital (float, optional): Initial capital to allocate per side. Defaults to 1_000_000.
        plot (bool, optional): If True, plot realized PnL over time. Defaults to True.
        n (int, optional): Number of positions to open on each side (long/short). Defaults to 2.
        threshold (float, optional): Minimum residual magnitude to consider a trade. Defaults to 0.02.

    Returns:
        tuple:
            trade_df (pd.DataFrame): Records of trades executed.
            pnl_df (pd.DataFrame): Realized PnL over time.
            position_tracker (pd.DataFrame): Current open positions with exposures.
    """
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_cols = {'df_filename', 'timestamp', 'Tenor', 'adjusted_res
    required_cols = {'df_filename', 'timestamp', 'Tenor', 'adjusted_residual', 'Ask price', 'Bid price'}
    if not required_cols.issubset(df.columns):
        logging.error(
            "Missing required columns in DataFrame. "
            f"Ensure columns: {required_cols} exist."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(
        columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity']
    )
    timestamps = sorted(df['timestamp'].unique())

    for current_time in timestamps:
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset unrealized PnL (not used but can be extended)
        unrealized_pnl = 0

        total_long_exposure = sum(
            details['dollar_exposure']
            for details in balances.values()
            if details['position'] == 'long'
        )
        total_short_exposure = sum(
            abs(details['dollar_exposure'])
            for details in balances.values()
            if details['position'] == 'short'
        )

        # Close positions when conditions are met
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = (
                product_slice['Bid price'].iloc[0]
                if details['position'] == 'short'
                else product_slice['Ask price'].iloc[0]
            )
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (
                (details['position'] == 'long' and (current_residual > 0 or tenor < 1 / 365))
                or (details['position'] == 'short' and (current_residual < 0 or tenor < 1 / 365))
            ):
                pnl = (
                    (current_price - weighted_avg_price) * quantity
                    if details['position'] == 'long'
                    else (weighted_avg_price - current_price) * quantity
                )
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append(
                    [current_time, product, -quantity, round(quantity, 0), current_price]
                )

        # Remove closed positions from balances and position tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n_positions = min(n, len(residuals) // 2)
        if n_positions == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n_positions:]
        most_underpriced_indices = residuals_sorted_indices[:n_positions]

        total_residual_magnitude = np.sum(
            np.abs(
                residuals.iloc[most_overpriced_indices].tolist()
                + residuals.iloc[most_underpriced_indices].tolist()
            )
        )
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_positions)
            quantity = exposure / row['Ask price']

            existing_long_position = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue

            total_long_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'long',
                'quantity': quantity,
                'entry_price': row['Ask price'],
                'dollar_exposure': exposure,
                'weighted_avg_price': row['Ask price'],
            }
            trade_records.append(
                [current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']]
            )

            existing = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing['net_position'].values[0] * existing['weighted_avg_price'].values[0])
                    + (quantity * row['Ask price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat(
                    [
                        position_tracker,
                        pd.DataFrame(
                            [[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]],
                            columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'],
                        ),
                    ],
                    ignore_index=True,
                )

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_positions)
            quantity = -exposure / row['Bid price']

            existing_short_position = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            if abs(total_short_exposure + exposure) > 1_000_000:
                continue

            total_short_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'short',
                'quantity': quantity,
                'entry_price': row['Bid price'],
                'dollar_exposure': -exposure,
                'weighted_avg_price': row['Bid price'],
            }
            trade_records.append(
                [current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']]
            )

            existing = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing['net_position'].values[0] * existing['weighted_avg_price'].values[0])
                    + (quantity * row['Bid price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat(
                    [
                        position_tracker,
                        pd.DataFrame(
                            [[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]],
                            columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'],
                        ),
                    ],
                    ignore_index=True,
                )

        # Record realized PnL for the current timestamp
        pnl_tracker.append([current_time, realized_pnl])

    # Create DataFrames for trades and PnL
    trade_df = pd.DataFrame(
        trade_records, columns=["timestamp", "product", "amount_available", "position", "price"]
    )
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return trade_df, pnl_df, position_tracker

def process_trading_strategy(
    df, n, start_capital, threshold, plot=False
):
    """
    Process trading strategy based on residuals and market data.

    This function simulates a trading strategy that opens and closes long and short
    positions based on adjusted residuals, maintaining exposure limits and tracking PnL.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing market data and residuals. Must contain columns:
        ['df_filename', 'timestamp', 'Tenor', 'adjusted_residual', 'Ask price', 'Bid price'].
    n : int
        Number of positions to open on each side (long and short).
    start_capital : float
        Initial capital to allocate for trading.
    threshold : float
        Minimum absolute residual value to consider for trading.
    plot : bool, optional
        If True, plots realized PnL over time. Default is False.

    Returns
    -------
    trade_df : pandas.DataFrame
        DataFrame logging all trades executed with columns:
        ['timestamp', 'product', 'amount_available', 'position', 'price'].
    pnl_df : pandas.DataFrame
        DataFrame containing realized PnL over time with columns ['timestamp', 'PnL'].
    position_tracker : pandas.DataFrame
        Current open positions with columns:
        ['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'].

    Notes
    -----
    Positions are closed when residuals revert or when Tenor is less than 1 day.
    Exposure is managed to not exceed 90% of start capital for longs and 1,000,000 USD for shorts.
    """

    required_cols = {
        'df_filename', 'timestamp', 'Tenor', 'adjusted_residual', 'Ask price', 'Bid price'
    }
    if not required_cols.issubset(df.columns):
        logging.error(
            "Missing required columns in DataFrame. "
            f"Ensure columns: {required_cols} exist."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(
        columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity']
    )
    timestamps = sorted(df['timestamp'].unique())

    for current_time in timestamps:
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset unrealized PnL (not used but can be extended)
        unrealized_pnl = 0

        total_long_exposure = sum(
            details['dollar_exposure']
            for details in balances.values()
            if details['position'] == 'long'
        )
        total_short_exposure = sum(
            abs(details['dollar_exposure'])
            for details in balances.values()
            if details['position'] == 'short'
        )

        # Close positions when conditions are met
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = (
                product_slice['Bid price'].iloc[0]
                if details['position'] == 'short'
                else product_slice['Ask price'].iloc[0]
            )
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (
                (details['position'] == 'long' and (current_residual > 0 or tenor < 1 / 365))
                or (details['position'] == 'short' and (current_residual < 0 or tenor < 1 / 365))
            ):
                pnl = (
                    (current_price - weighted_avg_price) * quantity
                    if details['position'] == 'long'
                    else (weighted_avg_price - current_price) * quantity
                )
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append(
                    [current_time, product, -quantity, round(quantity, 0), current_price]
                )

        # Remove closed positions from balances and position tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n_positions = min(n, len(residuals) // 2)
        if n_positions == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n_positions:]
        most_underpriced_indices = residuals_sorted_indices[:n_positions]

        total_residual_magnitude = np.sum(
            np.abs(
                residuals.iloc[most_overpriced_indices].tolist()
                + residuals.iloc[most_underpriced_indices].tolist()
            )
        )
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_positions)
            quantity = exposure / row['Ask price']

            existing_long_position = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue

            total_long_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'long',
                'quantity': quantity,
                'entry_price': row['Ask price'],
                'dollar_exposure': exposure,
                'weighted_avg_price': row['Ask price'],
            }
            trade_records.append(
                [current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']]
            )

            existing = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing['net_position'].values[0] * existing['weighted_avg_price'].values[0])
                    + (quantity * row['Ask price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat(
                    [
                        position_tracker,
                        pd.DataFrame(
                            [[
                                row['df_filename'], quantity * row['Ask price'], quantity,
                                row['Ask price'], quantity
                            ]],
                            columns=[
                                'product', 'dollar_exposure', 'net_position',
                                'weighted_avg_price', 'quantity'
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side / (2 * n_positions)
            quantity = -exposure / row['Bid price']

            existing_short_position = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            if abs(total_short_exposure + exposure) > 1_000_000:
                continue

            total_short_exposure += exposure
            balances[row['df_filename']] = {
                'position': 'short',
                'quantity': quantity,
                'entry_price': row['Bid price'],
                'dollar_exposure': -exposure,
                'weighted_avg_price': row['Bid price'],
            }
            trade_records.append(
                [current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']]
            )

            existing = position_tracker[
                position_tracker['product'] == row['df_filename']
            ]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = (
                    (existing['net_position'].values[0] * existing['weighted_avg_price'].values[0])
                    + (quantity * row['Bid price'])
                ) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[
                    position_tracker['product'] == row['df_filename'],
                    ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure'],
                ] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat(
                    [
                        position_tracker,
                        pd.DataFrame(
                            [[
                                row['df_filename'], quantity * row['Bid price'], quantity,
                                row['Bid price'], quantity
                            ]],
                            columns=[
                                'product', 'dollar_exposure', 'net_position',
                                'weighted_avg_price', 'quantity'
                            ],
                        ),
                    ],
                    ignore_index=True,
                )

        # Record realized PnL for the current timestamp
        pnl_tracker.append([current_time, realized_pnl])

    # Create DataFrames for trades and PnL
    trade_df = pd.DataFrame(
        trade_records, columns=["timestamp", "product", "amount_available", "position", "price"]
    )
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return trade_df, pnl_df, position_tracker
