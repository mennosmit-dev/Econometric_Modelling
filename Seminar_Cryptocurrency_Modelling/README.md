# Residual-Based Trading Strategy Simulation

Python implementation of a systematic trading strategy developed under supervision 
of **Maverick Derivatives**, focusing on residual-based mispricing signals, 
dynamic position management, and risk-controlled execution.

The strategy opens and closes long/short positions based on adjusted residuals, 
tracks realized PnL, and enforces exposure constraints across instruments.

---

## 🧠 Overview

Core components:

- Residual-driven signal generation
- Dynamic long/short position management
- Exposure and capital constraints
- Realized PnL tracking and visualization
- Trade logging and position monitoring

The objective is to exploit pricing inefficiencies while maintaining 
controlled risk exposure.

---

## 📂 Strategy Logic

At each timestamp:

1. Evaluate adjusted residuals across instruments.
2. Identify underpriced (long) and overpriced (short) opportunities.
3. Open positions subject to exposure and capital limits.
4. Close positions when residuals revert or maturity approaches.
5. Update realized PnL and portfolio state.

---

## 📊 Data Requirements

Input DataFrame must include:

- `df_filename` — Instrument identifier
- `timestamp` — Market snapshot time
- `Tenor` — Time to maturity
- `adjusted_residual` — Mispricing signal
- `Ask price` — Execution ask
- `Bid price` — Execution bid

---

## ⚙️ Usage

```python
from strategy_module import process_trading_strategy
import pandas as pd

df = pd.read_csv("market_data.csv")

trade_df, pnl_df, position_tracker = process_trading_strategy(
    df,
    n=5,
    start_capital=1_000_000,
    threshold=0.01,
    plot=True
)
```
Outputs:

trade_df — Executed trades

pnl_df — Realized PnL over time

position_tracker — Current open positions

## 🔧 Tech Stack

Python • Pandas • NumPy • Quantitative Trading • Risk Management

## 📌 Context

This project was developed during a quantitative research collaboration
with Maverick Derivatives and complements my broader work in:
- reinforcement learning for portfolio management
- econometric forecasting
- systematic trading strategy design

For technical details and results, see the accompanying project PDF.





    threshold=0.01,
    plot=True
)
