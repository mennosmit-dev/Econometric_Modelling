**This project was under supervision of Maverick Derrivatives. For details, feel free to download the pdf in the map.**


# Trading Strategy Simulation with Residual-Based Positions

## Overview

This repository contains a Python implementation of a trading strategy that opens and closes long and short positions based on adjusted residuals from market data. The strategy manages exposure limits, tracks realized profit and loss (PnL), and supports visualization of PnL over time.

The approach leverages data on residuals, bid/ask prices, and tenor to decide when to open or close positions, aiming to capitalize on pricing inefficiencies.

---

## Features

- Processes timestamped market data with adjusted residuals.
- Opens and closes positions dynamically based on residual signals and tenor.
- Tracks dollar exposure and net positions for multiple products.
- Calculates realized PnL continuously.
- Manages position sizing and exposure limits for risk control.
- Optional visualization of realized PnL over time.
- Outputs detailed trade logs, PnL history, and current positions.

---

## Data Requirements

The input DataFrame **must** contain the following columns:

- `df_filename` — Identifier for the product/instrument.
- `timestamp` — Timestamp of the market snapshot.
- `Tenor` — Time to maturity (in years).
- `adjusted_residual` — Residual value indicating mispricing.
- `Ask price` — Current ask price.
- `Bid price` — Current bid price.

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   (Make sure you have pandas, numpy, matplotlib, and logging installed.)
   ```

## Installation & Setup

Import and run the main strategy function (example):

import pandas as pd
from strategy_module import process_trading_strategy  # Adjust import as per your file structure

# Load your market data into a DataFrame 'df'
df = pd.read_csv("market_data.csv")

# Parameters
n_positions = 5
starting_capital = 1_000_000
residual_threshold = 0.01
plot_pnl = True

# Run strategy
trade_df, pnl_df, position_tracker = process_trading_strategy(
    df,
    n=n_positions,
    start_capital=starting_capital,
    threshold=residual_threshold,
    plot=plot_pnl
)

# Save outputs if needed
trade_df.to_csv("trades.csv", index=False)
pnl_df.to_csv("pnl.csv", index=False)
position_tracker.to_csv("positions.csv", index=False)

# Function Details
`process_trading_strategy(df, n, start_capital, threshold, plot=False)`
Simulates the trading strategy with the given input DataFrame and parameters.

## Parameters:
• df: Input DataFrame with required columns.

• n: Number of positions to open per side (long/short).

• start_capital: Initial capital allocation.

• threshold: Minimum residual magnitude to trigger trades.

• plot: If True, plots realized PnL over time.

## Returns:

• trade_df: DataFrame logging all trades executed.

• pnl_df: DataFrame of realized PnL across timestamps.

• position_tracker: DataFrame of current open positions.

# How It Works

• At each timestamp, the strategy evaluates residuals to determine overpriced and underpriced instruments.

• Positions are opened for the most underpriced (long) and overpriced (short) products.

• Positions are closed if residuals revert or the instrument nears maturity.

• Position sizes are limited to avoid exceeding risk and capital constraints.

• Realized PnL is tracked cumulatively and can be visualized.

# Contributing
Contributions are welcome! Please open issues or submit pull requests to improve features, optimize performance, or add support for additional instruments.

