"""
This file is used to trade BTC and ETH futures based on econometric equations 
that should theoretically hold.

For the details, please refer to the accompanying PDF in which everything 
is discussed in detail.
"""
# Download and reformat files
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import re
import time
import pandas as pd
from datetime import datetime


def reformat_date_column(df, column_name):
    # Process all rows in the specified column
    def reformat_value(value):
        try:
            # Split the prefix (e.g., ETH) and the date part (_16JUN23)
            prefix, date_part = value.split('_', 1)
            # Parse the date part
            date_obj = datetime.strptime(date_part, '%d%b%y')
            # Reformat as 'ETH_230616'
            return f"{prefix}_{date_obj.strftime('%y%m%d')}"
        except (ValueError, IndexError):
            # If parsing fails or format is unexpected, return the original value
            return value

    # Apply the reformatting function to the entire column
    df[column_name] = df[column_name].apply(reformat_value)
    return df

def reformat_date_column2(df, column_name):

    def reformat_value(value):
        if isinstance(value, str) and value.count('-') == 2:
            # Replace '-' with '_'
            return value.replace('-', '_')
        # If the value doesn't have the expected format, return it as is
        return value

    # Apply the reformatting function to the column
    df[column_name] = df[column_name].apply(reformat_value)
    return df

def calculate_tenor(df, column1, column2):

    # Define CET timezone
    cet = pytz.timezone('CET')

    def parse_column1(value):
        # Extract last 6 characters and parse as datetime
        try:
            date_str = value[-6:]  # Extract yymmdd
            date_obj = datetime.strptime(date_str, '%y%m%d')
            # Set time to 9:00 CET
            return cet.localize(datetime.combine(date_obj.date(), datetime.min.time()) + timedelta(hours=9))
        except ValueError:
            return None

    # Process the first column (convert to CET timezone-aware datetime)
    df['column1_datetime'] = df[column1].apply(parse_column1)

    # Convert the second column to datetime and localize to UTC (handling already localized values)
    def parse_column2(value):
        try:
            dt = pd.to_datetime(value, errors='coerce')
            if dt.tzinfo is None:  # If naive, localize to UTC and convert to CET
                return dt.tz_localize('UTC').tz_convert(cet)
            return dt.tz_convert(cet)  # If already tz-aware, convert to CET
        except Exception:
            return None

    df['column2_datetime'] = df[column2].apply(parse_column2)

    # Calculate tenor as the difference
    df['tenor'] = df['column1_datetime'] - df['column2_datetime']

    # Drop intermediate datetime columns if not needed
    df.drop(columns=['column1_datetime', 'column2_datetime'], inplace=True)

    return df

def precompute_spot_prices_last(reference_df, start_time, end_time):
    # Define hourly intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')

    spot_prices = {}

    for i in range(len(timestamps) - 1):
        start_interval = timestamps[i]
        end_interval = timestamps[i +1]

        # Filter values within the interval
        interval_values = reference_df[
            (reference_df['dateTime_round_s'] >= start_interval) &
            (reference_df['dateTime_round_s'] < end_interval)
        ]

        if not interval_values.empty:
            # Select the last available point within the interval
            last_point = interval_values.iloc[-1]
            # Assign the value to the beginning of the next hour
            spot_prices[end_interval] = last_point['mid_Px']
        else:
            # If no values are available, set NaN
            spot_prices[end_interval] = np.nan

    return spot_prices

def precompute_spot_prices_average(reference_df, start_time, end_time):
    # Define hourly intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
    spot_prices = {}

    for i in range(len(timestamps) - 1):
        start_interval = timestamps[i]
        end_interval = timestamps[i + 1]

        # Get midpoint for the interval
        midpoint = (start_interval + (end_interval - start_interval) / 2).replace(second=0)

        # Filter values within the interval
        interval_values = reference_df[
            (reference_df['dateTime_round_s'] >= start_interval) &
            (reference_df['dateTime_round_s'] < end_interval)
        ]

        # Calculate the average if there are values, otherwise set NaN
        spot_prices[midpoint] = interval_values['mid_Px'].mean() if not interval_values.empty else np.nan

    return spot_prices
# Load the Parquet files

df1 = pd.read_parquet("C:/Users/EfeKa/OneDrive/Bureaublad/Seminar/merged_batch_1.parquet")
df2 = pd.read_parquet("C:/Users/EfeKa/OneDrive/Bureaublad/Seminar/merged_batch_2.parquet")

if list(df1.columns) != list(df2.columns):
    raise ValueError("Column names or order do not match between the files.")

# Concatenate vertically
df = pd.concat([df1, df2], ignore_index=True)

print('files merged')
#df=pd.read_parquet('/Users/hugogroenewegen/Master/Seminar/part0.parquet 2')
df['symbol'] = df['symbol'].str.replace('-', '_')
df['spread bps']= 10000*(df['ask_price']-df['bid_price'])/df['mid_Px']
df['liquidity prox'] = df['ask_price'] * df['ask_amount'] + df['bid_price'] * df['bid_amount']


btc_df = df[df['symbol'].str.contains('BTC')]
eth_df = df[df['symbol'].str.contains('ETH')]

print('files split into btc/eth ')

btc_binance = btc_df[btc_df['exchange'] == 'binance']
btc_okex = btc_df[btc_df['exchange'] == 'okex']
btc_deribit = btc_df[btc_df['exchange'] == 'deribit']
btc_deribit=reformat_date_column(btc_deribit,'symbol')
eth_binance = eth_df[eth_df['exchange'] == 'binance']
eth_okex = eth_df[eth_df['exchange'] == 'okex']
eth_deribit = eth_df[eth_df['exchange'] == 'deribit']
eth_deribit=reformat_date_column(eth_deribit,'symbol')
# Define a dictionary to map DataFrame variables to names
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
count = 0

# Double loop to process each dataframe
for name, file in exchange_names.items():  # Iterate over the dictionary
    unique_elements = file['symbol'].unique()
    for element in unique_elements:
        new_df_name = f"{name}_{element}"  # Use the variable name from the dictionary
        all_files.append((new_df_name, file[file['symbol'] == element]))  # Store name and df as a tuple

        # Optionally create the DataFrame in globals (not required unless you need it globally)
        globals()[new_df_name] = file[file['symbol'] == element]
print('files split into instruments')
# Precompute spot prices for BTC and ETH


# Time range
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

#Aggregate; combined
import pandas as pd
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

interval = '1H'

all_combined_files = []

def aggregate_by_interval(df, interval='1H'):
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

    # Renaming columns to follow the "_avg" convention
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
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df = df.sort_values(by='dateTime')
    df['interval_start'] = df['dateTime'].dt.floor(interval) + pd.to_timedelta(1, unit='h')

    last_points = df.groupby('interval_start').last().reset_index()
    last_points.rename(columns={'interval_start': 'hour_end'}, inplace=True)

    # Renaming columns to follow the "_end" convention
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
    file_name = f[0]
    df = f[1]

    last_points_df = aggregate_last_available_by_hour(df)
    aggregated_df = aggregate_by_interval(df, interval)

    combined_df = last_points_df.merge(aggregated_df, on='hour_end', suffixes=('_end', '_avg'))
    combined_df['symbol'] = df['symbol'].values[0]
    combined_df['exchange'] = df['exchange'].values[0]

    all_combined_files.append((f"last_point_{file_name}", combined_df))

    if not (file_name.endswith("USDT") or "SWAP" in file_name or "PERP" in file_name):
        calculate_tenor(combined_df, 'symbol', 'hour_end')

#model with all features
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def estimate_regression(data_last, method='OLS', use_indicator=True, use_spread=True, use_liquidity=True, use_midpx_avg=False, t_value=2/3):

    data_last = np.asarray(data_last, dtype=np.float64)
    x = data_last[:, 0]  # Tenor
    y = data_last[:, 1]  # Future premium
    n = len(x)

    # Design matrix with intercept and polynomial terms

    X = np.column_stack((np.ones(n), x))

    # Add indicator variable if specified
    if use_indicator:
        if t_value is None:
            raise ValueError("t_value must be provided when use_indicator=True")
        indicator = (x <= t_value).astype(np.float64) * x
        X = np.column_stack([X, indicator])
    if use_midpx_avg:
        mid_px_avg = data_last[:,-1]
        X = np.column_stack([X, mid_px_avg])

    if use_liquidity:
        liquidity = data_last[:,-2]
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
        weights = 1 / (np.abs(residuals_ols) + 1e-6)  # Approximate heteroskedasticityx
        sigma = np.diag(weights)
        model = sm.GLS(y, X, sigma=sigma).fit()
    elif method == 'RLS':
        model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    else:
        raise ValueError("Invalid method. Choose 'OLS', 'WLS','RLS', or 'GLS'")

    # Extract results
    beta = model.params
    residuals = model.resid

    results = {
        'slope': beta[1]
    }
    results['intercept'] = beta[0]
    index = 2  # Start from beta[2] since beta[0] is intercept, beta[1] is slope

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
        results['Mid Price'] = beta[index]  # This will be the last one



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

# Iterate over time
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

        tenor_value, px_mid_value = row['tenor'].iloc[0], row['mid_Px_end'].iloc[0]
        log_px_mid_value = np.log(px_mid_value)
        spot = float(spot_series[spot_series['time'] == pd.Timestamp(current_time)]['spot'])
        future_premium = np.log(px_mid_value / spot)
        ask_price, bid_price = row['ask_price_end'].iloc[0], row['bid_price_end'].iloc[0]
        liquidity, avg_midprice,spread =row['liquidity_prox_avg'].iloc[0],row['mid_Px_avg'].iloc[0],row['spread_bps_avg'].iloc[0]


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

        data_pairs.append([tenor_value, future_premium, color, df_name, px_mid_value, ask_price, bid_price,spread,liquidity,avg_midprice])

    matrix = np.array(data_pairs, dtype=object)

    if len(data_pairs) >  14:
        tenors, future_premium, colors, filenames, mid_prices, ask_price, bid_price, spread ,liquidity, avg_midprice= (
            matrix[:, 0].astype(np.float64),
            matrix[:, 1].astype(np.float64),
            matrix[:, 2],
            matrix[:, 3],
            matrix[:, 4].astype(np.float64),
            matrix[:, 5].astype(np.float64),
            matrix[:, 6].astype(np.float64),
            matrix[:, 7].astype(np.float64),
            matrix[:, 8].astype(np.float64),
            matrix[:, 9].astype(np.float64)
        )

        valid_data = [i for i in range(len(tenors)) if pd.notna(tenors[i]) and pd.notna(mid_prices[i])]
        tenors_valid, premium_valid, colors_valid, filenames_valid, mid_prices_valid, ask_price, bid_price,spread ,liquidity, avg_midprice= (
            tenors[valid_data],
            future_premium[valid_data],
            colors[valid_data],
            filenames[valid_data],
            mid_prices[valid_data],
            ask_price[valid_data],
            bid_price[valid_data],
            spread[valid_data],
            liquidity[valid_data],
            avg_midprice[valid_data]

        )

        valid_matrix = np.column_stack((tenors_valid, future_premium,spread,liquidity,avg_midprice)).astype(np.float64)

        coeffs_and_pvalues = estimate_regression(valid_matrix)

        slope = coeffs_and_pvalues['slope']
        intercept = coeffs_and_pvalues['intercept']
        residuals = coeffs_and_pvalues['residuals']

        X = coeffs_and_pvalues['design_matrix']

        '''
        plt.figure(figsize=(8, 6))
        for i in range(len(tenors_valid)):
            plt.scatter(tenors_valid[i], premium_valid[i], color=colors_valid[i], label= "")

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

#PnL Benchmark Directional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker

pnl_results, pnl_time_series,pt = calculate_pnL(detailed_df_D, n=2)
pt

#PnL Benchmark EW
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]
            exposure = total_exposure_per_side/(2*n)
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]
            exposure = total_exposure_per_side/(2*n)
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])

    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker

pnl_results, pnl_time_series,pt = calculate_pnL(detailed_df_D, n=2)
pt

#PnL Modified Directional
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2, threshold=200/10000):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = (abs(row['adjusted_residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = (abs(row['adjusted_residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])

    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker
pnl_results, pnl_time_series,pt = calculate_pnL(detailed_df_D, n=2)
pt

#PnL Modified EW
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2, threshold=200/10000):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side/(2*n)
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side/(2*n)
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])

    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker
trade_df_D, pnl_time_series,pt = calculate_pnL(detailed_df_D)
pt

#Buy Sell plots
import matplotlib.pyplot as plt
import pandas as pd

def plot_instrument_trades(trade_df):
    if trade_df.empty:
        print("No trade data available to plot.")
        return

    trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
    unique_instruments = trade_df['instrument'].unique()

    for instrument in unique_instruments:
        instrument_data = trade_df[trade_df['instrument'] == instrument]

        plt.figure(figsize=(12, 6))
        plt.plot(instrument_data['timestamp'], instrument_data['mid_price'], label=instrument, linestyle='-', marker='')

        buy_signals = instrument_data[instrument_data['trade_action'] == 1]
        sell_signals = instrument_data[instrument_data['trade_action'] == -1]

        plt.scatter(buy_signals['timestamp'], buy_signals['mid_price'], marker='^', color='green', label='Buy')
        plt.scatter(sell_signals['timestamp'], sell_signals['mid_price'], marker='x', color='red', label='Sell')

        plt.xlabel("Time")
        plt.ylabel("Mid Price")
        plt.title(f"Instrument Mid Price with Buy/Sell Signals: {instrument}")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.show()

# Example usage
# plot_instrument_trades(trade_df)


plot_instrument_trades(trade_df_D)

#mean reversion test
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Initialize best_instrument as a DataFrame
best_instrument = pd.DataFrame(columns=['Instrument', 'AR1', 'Half-Life AR'])

def estimate_theta(ts, dt=1):
    Y = ts[:-1]
    X = ts[1:]

    theta_init = -np.log(np.corrcoef(X, Y)[0, 1]) / dt

    def neg_log_likelihood(theta):
        if theta <= 0:
            return np.inf

        residuals_shifted = ts[:-1]
        expected = residuals_shifted * np.exp(-theta * dt)
        variance = np.var(ts) * (1 - np.exp(-2 * theta * dt))

        return -np.sum(-0.5 * np.log(2 * np.pi * variance) - (X - expected) ** 2 / (2 * variance))

    result = minimize(neg_log_likelihood, [theta_init], method='L-BFGS-B')
    return result.x[0] if result.success else None

def compute_ar1_metrics(df, min_obs=100):
    global best_instrument  # Ensure the DataFrame is modified globally
    ar1_results = []
    non_stationary_count = 0
    total_series = 0

    for f in pd.unique(df['df_filename']):
        residuals = df[df['df_filename'] == f]['residual'].dropna()
        num_observations = len(residuals)

        if num_observations < min_obs:
            continue

        total_series += 1

        # AR(1) model fitting
        model = ARIMA(residuals, order=(1, 0, 0))
        result = model.fit()
        ar1_coeff = round(result.arparams[0], 3)

        # Perform ADF test
        adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals)
        adf_stat = round(adf_stat, 3)

        if p_value > 0.05:
            non_stationary_count += 1

        # Variance ratio
        differenced_residuals = np.diff(residuals)
        lags = 1
        variances = [np.var(differenced_residuals[:num_observations - lag]) / np.var(differenced_residuals[lag:]) for lag in range(1, lags + 1)]
        variance_ratio = np.mean(variances)

        theta = estimate_theta(residuals, dt=1)
        half_life_OU = round(np.log(2) / theta, 3)
        half_life_AR = round(-np.log(2) / np.log(ar1_coeff), 4)

        ar1_results.append((f, num_observations, theta, half_life_OU, p_value, ar1_coeff, half_life_AR, variance_ratio))

        # Add to best_instrument DataFrame if AR1 < 0.7
        if half_life_AR < 2:
            best_instrument = pd.concat([best_instrument, pd.DataFrame([[f, ar1_coeff, half_life_AR]], columns=best_instrument.columns)], ignore_index=True)

    # Convert to DataFrame
    ar1_df = pd.DataFrame(ar1_results, columns=['Instrument', 'Observations', 'OU', 'Half-Life OU', 'ADF Statistic', 'AR1 Coefficient', 'Half-Life AR', 'VRT'])

    # Compute averages for valid instruments
    valid_ar1_df = ar1_df[ar1_df['Observations'] >= min_obs]
    mean_ar1 = round(valid_ar1_df['AR1 Coefficient'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_half_life_OU = round(valid_ar1_df['Half-Life OU'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_half_life_AR = round(valid_ar1_df['Half-Life AR'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_VRT = round(valid_ar1_df['VRT'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_OU = round(valid_ar1_df['OU'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    non_stationary_ratio = round((non_stationary_count / total_series) * 100, 2) if total_series > 0 else "N/A"

    return mean_OU, mean_half_life_OU, non_stationary_ratio, mean_ar1, mean_half_life_AR, mean_VRT

# Run function
mean_OU, mean_half_life_OU, non_stationary_ratio, mean_ar1, mean_half_life_AR, mean_VRT = compute_ar1_metrics(detailed_df_D, 100)

# Display best_instrument DataFrame
best_detailed_df_D=detailed_df_D[detailed_df_D['df_filename'].isin(best_instrument['Instrument'])]
best_detailed_df_D

#PnL Best instruments directional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]
            exposure = (abs(row['residual']) / total_residual_magnitude) * total_exposure_per_side
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])
    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker

pnl_results, pnl_time_series,pt = calculate_pnL(best_detailed_df_D, n=2)
pt

#PnL Best Instruments EW
def calculate_pnL(df, start_capital=1_000_000, plot=True, n=2, threshold=200/10000):
    if df.empty:
        logging.warning("The input DataFrame is empty. Exiting function.")
        return pd.DataFrame(), pd.DataFrame()

    if 'df_filename' not in df.columns or 'timestamp' not in df.columns or 'Tenor' not in df.columns:
        logging.error("Missing required columns in DataFrame. Ensure 'df_filename', 'timestamp', and 'Tenor' exist.")
        return pd.DataFrame(), pd.DataFrame()

    trade_records = []
    pnl_tracker = []
    realized_pnl = 0
    balances = {}
    position_tracker = pd.DataFrame(columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])
    timestamps = sorted(df['timestamp'].unique())

    for i, current_time in enumerate(timestamps):
        slice_df = df[df['timestamp'] == current_time]
        if slice_df.empty:
            continue

        residuals = slice_df['adjusted_residual']
        to_close = []

        # Reset the unrealized PnL
        unrealized_pnl = 0

        total_long_exposure = sum(details['dollar_exposure'] for details in balances.values() if details['position'] == 'long')
        total_short_exposure = sum(abs(details['dollar_exposure']) for details in balances.values() if details['position'] == 'short')

        # Close positions when necessary
        for product, details in list(balances.items()):
            product_slice = slice_df[slice_df['df_filename'] == product]
            if product_slice.empty:
                continue

            current_residual = product_slice['adjusted_residual'].iloc[0]
            current_price = product_slice['Bid price'].iloc[0] if details['position'] == 'short' else product_slice['Ask price'].iloc[0]
            tenor = product_slice['Tenor'].iloc[0]
            weighted_avg_price = details['weighted_avg_price']
            quantity = details['quantity']

            if (details['position'] == 'long' and (current_residual > 0 or tenor < 1/365)) or \
               (details['position'] == 'short' and (current_residual < 0 or tenor < 1/365)):
                pnl = (current_price - weighted_avg_price) * (quantity) if details['position'] == 'long' else (weighted_avg_price - current_price) * (quantity)
                realized_pnl += pnl
                to_close.append(product)
                trade_records.append([current_time, product, -quantity, round(quantity, 0), current_price])

        # Remove closed positions from balances and position_tracker
        for product in to_close:
            del balances[product]
            position_tracker = position_tracker[position_tracker['product'] != product]

        n = min(n, len(residuals) // 2)
        if n == 0:
            continue

        residuals_sorted_indices = np.argsort(residuals)
        most_overpriced_indices = residuals_sorted_indices[-n:]
        most_underpriced_indices = residuals_sorted_indices[:n]

        total_residual_magnitude = np.sum(np.abs(residuals.iloc[most_overpriced_indices].tolist() + residuals.iloc[most_underpriced_indices].tolist()))
        if total_residual_magnitude == 0:
            continue

        total_exposure_per_side = start_capital

        # Handle long positions
        for idx in most_underpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side/(2*n)
            quantity = exposure / row['Ask price']  # quantity is now based on exposure/price

            # Check if the position already exists and if the total long exposure exceeds 1,000,000
            existing_long_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_long_position.empty:
                total_long_exposure += existing_long_position['dollar_exposure'].sum()

            if total_long_exposure + exposure > 0.9 * start_capital or exposure > 1_000_000:
                continue  # Skip trade if exposure exceeds 1,000,000 or 90% of the capital

            total_long_exposure += exposure
            balances[row['df_filename']] = {'position': 'long', 'quantity': quantity, 'entry_price': row['Ask price'], 'dollar_exposure': exposure, 'weighted_avg_price': row['Ask price']}
            trade_records.append([current_time, row['df_filename'], exposure, round(quantity, 0), row['Ask price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Ask price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Ask price'], quantity, row['Ask price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Handle short positions
        for idx in most_overpriced_indices:
            row = slice_df.iloc[idx]

            # Skip if adjusted residual is less than threshold
            if abs(row['adjusted_residual']) < threshold:
                continue

            exposure = total_exposure_per_side/(2*n)
            quantity = -exposure / row['Bid price']  # quantity is now based on exposure/price

            # Ensure that the short exposure does not exceed -1,000,000
            existing_short_position = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing_short_position.empty:
                total_short_exposure -= existing_short_position['dollar_exposure'].sum()

            # Check if the total short exposure exceeds the limit before updating
            if np.abs(total_short_exposure + exposure) > 1_000_000:
                continue  # Skip trade if short exposure exceeds -1,000,000

            total_short_exposure += exposure
            balances[row['df_filename']] = {'position': 'short', 'quantity': quantity, 'entry_price': row['Bid price'], 'dollar_exposure': -exposure, 'weighted_avg_price': row['Bid price']}
            trade_records.append([current_time, row['df_filename'], -exposure, round(quantity, 0), row['Bid price']])

            # Update position tracker with quantity, exposure, and weighted price
            existing = position_tracker[position_tracker['product'] == row['df_filename']]
            if not existing.empty:
                new_position = existing['net_position'].values[0] + quantity
                new_avg_price = ((existing['net_position'].values[0] * existing['weighted_avg_price'].values[0]) + (quantity * row['Bid price'])) / new_position
                new_dollar_exposure = new_position * new_avg_price
                position_tracker.loc[position_tracker['product'] == row['df_filename'], ['net_position', 'weighted_avg_price', 'quantity', 'dollar_exposure']] = new_position, new_avg_price, existing['quantity'].values[0] + quantity, new_dollar_exposure
            else:
                position_tracker = pd.concat([position_tracker, pd.DataFrame([[row['df_filename'], quantity * row['Bid price'], quantity, row['Bid price'], quantity]], columns=['product', 'dollar_exposure', 'net_position', 'weighted_avg_price', 'quantity'])], ignore_index=True)

        # Record only realized PnL
        pnl_tracker.append([current_time, realized_pnl])


    # Return DataFrames for trade, PnL, and position tracking
    trade_df = pd.DataFrame(trade_records, columns=["timestamp", "product", "amount_available", "position", "price"])
    pnl_df = pd.DataFrame(pnl_tracker, columns=["timestamp", "PnL"])

    pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
    if plot:
        # Plotting the PnL data
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_df['timestamp'], pnl_df['PnL'], label='Realized PnL')

        # Set the x-axis to show only every 7 days
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the dates

        # Rotate the x-axis labels for readability
        plt.xticks(rotation=45)

        # Adding labels and title
        plt.ylabel('Realized PnL')
        plt.title('Realized PnL Over Time')
        plt.legend()

        # Show the plot
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    return trade_df, pnl_df, position_tracker
trade_df_D, pnl_time_series,pt = calculate_pnL(best_detailed_df_D,n=2)
pt

#model with all features and grid searchh
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

def estimate_theta(ts, dt=1):

    Y = ts[:-1]
    X = ts[1:]

    theta_init = -np.log(np.corrcoef(X, Y)[0, 1]) / dt

    def neg_log_likelihood(theta):
        if theta <= 0:
            return np.inf

        residuals_shifted = ts[:-1]
        expected =residuals_shifted * np.exp(-theta * dt)
        variance = np.var(ts) * (1 - np.exp(-2 * theta * dt))

        return -np.sum(-0.5 * np.log(2 * np.pi * variance) - (X - expected) ** 2 / (2 * variance))

    result = minimize(neg_log_likelihood, [theta_init], method='L-BFGS-B')
    return result.x[0] if result.success else None

def compute_ar1_metrics(df, min_obs=100):
    ar1_results = []
    non_stationary_count = 0
    total_series = 0



    for f in pd.unique(df['df_filename']):
        residuals = df[df['df_filename'] == f]['residual'].dropna()
        num_observations = len(residuals)

        if num_observations < min_obs:
            continue

        total_series += 1


        # AR(1) model fitting
        model = ARIMA(residuals, order=(1, 0, 0))
        result = model.fit()
        ar1_coeff = round(result.arparams[0], 3)

        # Perform ADF test
        adf_stat, p_value, _, _, critical_values, _ = adfuller(residuals)
        adf_stat = round(adf_stat, 3)

        if p_value > 0.05:
            non_stationary_count += 1
        # Variance ratio
        differenced_residuals = np.diff(residuals)
        lags=1
        variances = [np.var(differenced_residuals[:num_observations-lag]) / np.var(differenced_residuals[lag:]) for lag in range(1, lags+1)]
        variance_ratio = np.mean(variances)
        theta= estimate_theta(residuals, dt=1)

        if theta is not None:
            half_life_OU = round(np.log(2) / theta, 3)
        else:
            continue
        half_life_AR = round(-np.log(2) / np.log(ar1_coeff))
        ar1_results.append((f, num_observations, theta, half_life_OU,p_value, ar1_coeff, half_life_AR, variance_ratio,))

    # Convert to DataFrame
    ar1_df = pd.DataFrame(ar1_results, columns=['Instrument', 'Observations', 'OU','Half-Life OU','ADF Statistic','AR1 Coefficient', 'Half-Life AR', 'VRT'])
    # Compute averages for valid instruments
    valid_ar1_df = ar1_df[ar1_df['Observations'] >= min_obs]
    mean_ar1 = round(valid_ar1_df['AR1 Coefficient'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_half_life_OU = round(valid_ar1_df['Half-Life OU'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_half_life_AR = round(valid_ar1_df['Half-Life AR'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_VRT = round(valid_ar1_df['VRT'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    mean_OU = round(valid_ar1_df['OU'].mean(), 4) if not valid_ar1_df.empty else "N/A"
    non_stationary_ratio = round((non_stationary_count / total_series) * 100, 2) if total_series > 0 else "N/A"


    return mean_OU, mean_half_life_OU, non_stationary_ratio, mean_ar1, mean_half_life_AR, mean_VRT

df = pd.DataFrame(columns=['Settings', 'OU','Half Time OU','ADF','AR1', 'Half Time AR','VRT'])

from itertools import product

# Define possible values for each parameter
methods = ['WLS', 'OLS', 'GLS', 'RLS']
boolean_options = [True, False]
boolean_midpx = [False]

# Generate all possible combinations of parameter values
grid_settings = [
    {
        'method': method,
        'use_indicator': ind,
        'use_spread': spread,
        'use_liquidity': liquidity,
        'use_midpx_avg': midpx_avg
    }
    for method, ind, spread, liquidity, midpx_avg in product(
        methods, boolean_options, boolean_options, boolean_options, boolean_midpx
    )
]
for grid in grid_settings:
# New function accepting grid as input
    def estimate_regression(grid, data_last, t_value=2/3):
        method = grid['method']
        use_indicator = grid['use_indicator']
        use_spread = grid['use_spread']
        use_liquidity = grid['use_liquidity']
        use_midpx_avg = grid['use_midpx_avg']

        data_last = np.asarray(data_last, dtype=np.float64)
        x = data_last[:, 0]  # Tenor
        y = data_last[:, 1]  # Future premium
        n = len(x)

        # Design matrix with intercept and polynomial terms

        X = np.column_stack((np.ones(n), x))

        # Add indicator variable if specified
        if use_indicator:
            if t_value is None:
                raise ValueError("t_value must be provided when use_indicator=True")
            indicator = (x <= t_value).astype(np.float64) * x
            X = np.column_stack([X, indicator])
        if use_midpx_avg:
            mid_px_avg = data_last[:,-1]
            X = np.column_stack([X, mid_px_avg])

        if use_liquidity:
            liquidity = data_last[:,-2]
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
            weights = 1 / (np.abs(residuals_ols) + 1e-6)  # Approximate heteroskedasticityx
            sigma = np.diag(weights)
            model = sm.GLS(y, X, sigma=sigma).fit()
        elif method == 'RLS':
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        else:
            raise ValueError("Invalid method. Choose 'OLS', 'WLS','RLS', or 'GLS'")

        # Extract results
        beta = model.params
        residuals = model.resid

        results = {
            'slope': beta[1]
        }
        results['intercept'] = beta[0]
        index = 2  # Start from beta[2] since beta[0] is intercept, beta[1] is slope

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
            results['Mid Price'] = beta[index]  # This will be the last one


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

    # Iterate over time
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

            tenor_value, px_mid_value = row['tenor'].iloc[0], row['mid_Px_end'].iloc[0]
            log_px_mid_value = np.log(px_mid_value)
            spot = float(spot_series[spot_series['time'] == pd.Timestamp(current_time)]['spot'])
            future_premium = np.log(px_mid_value / spot)
            ask_price, bid_price = row['ask_price_end'].iloc[0], row['bid_price_end'].iloc[0]
            liquidity, avg_midprice,spread =row['liquidity_prox_end'].iloc[0],row['mid_Px_avg'].iloc[0],row['spread_bps_end'].iloc[0]

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

            data_pairs.append([tenor_value, future_premium, color, df_name, px_mid_value, ask_price, bid_price,spread,liquidity,avg_midprice])

        matrix = np.array(data_pairs, dtype=object)

        if len(data_pairs) >  14:
            tenors, future_premium, colors, filenames, mid_prices, ask_price, bid_price, spread ,liquidity, avg_midprice= (
                matrix[:, 0].astype(np.float64),
                matrix[:, 1].astype(np.float64),
                matrix[:, 2],
                matrix[:, 3],
                matrix[:, 4].astype(np.float64),
                matrix[:, 5].astype(np.float64),
                matrix[:, 6].astype(np.float64),
                matrix[:, 7].astype(np.float64),
                matrix[:, 8].astype(np.float64),
                matrix[:, 9].astype(np.float64)
            )

            valid_data = [i for i in range(len(tenors)) if pd.notna(tenors[i]) and pd.notna(mid_prices[i])]
            tenors_valid, premium_valid, colors_valid, filenames_valid, mid_prices_valid, ask_price, bid_price,spread ,liquidity, avg_midprice= (
                tenors[valid_data],
                future_premium[valid_data],
                colors[valid_data],
                filenames[valid_data],
                mid_prices[valid_data],
                ask_price[valid_data],
                bid_price[valid_data],
                spread[valid_data],
                liquidity[valid_data],
                avg_midprice[valid_data]

            )

            valid_matrix = np.column_stack((tenors_valid, future_premium,spread,liquidity,avg_midprice)).astype(np.float64)

            coeffs_and_pvalues = estimate_regression(grid, valid_matrix)

            beta = coeffs_and_pvalues['slope']
            intercept = coeffs_and_pvalues['intercept']
            residuals = coeffs_and_pvalues['residuals']

            X = coeffs_and_pvalues['design_matrix']

            '''
            plt.figure(figsize=(8, 6))
            for i in range(len(tenors_valid)):
                plt.scatter(tenors_valid[i], premium_valid[i], color=colors_valid[i], label= "")

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
                    "slope": beta,
                    "residual": residuals[i],
                    "Bid price": bid_price[i],
                    "Ask price": ask_price[i],
                    "Liquidity": liquidity[i]
                })
            time_series_data_D.append({"timestamp": current_time, **coeffs_and_pvalues})
        current_time += pd.Timedelta(seconds=3600)

    detailed_df_D = pd.DataFrame(detailed_rows_D)
    time_series_df_D = pd.DataFrame(time_series_data_D)
    mean_OU, mean_half_life_OU, non_stationary_ratio, mean_ar1, mean_half_life_AR, mean_VRT=compute_ar1_metrics(detailed_df_D, min_obs=0.1*24*30)
    new_row = {'Settings': grid, 'OU': mean_OU, 'Half Life OU':mean_half_life_OU, 'ADF': non_stationary_ratio, 'Mean Reversion': mean_ar1, 'Half Life AR':mean_half_life_AR, 'VRT': mean_VRT }  # Replace with your actual values
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(new_row)

import pandas as pd
detailed_df_D = pd.read_excel("/Users/hugogroenewegen/Master/Seminar/Detailed_GLS.xlsx")


detailed_df_D['adjusted_residual'] = (1 + detailed_df_D['residual']) ** (1 / detailed_df_D['Tenor']) - 1
print(np.mean(np.abs(detailed_df_D['adjusted_residual'])))
detailed_df_D
