# Vasicek Cap Pricing – Monte Carlo Simulation

This repository contains a Python implementation for pricing an interest rate cap 
using a **one-factor Vasicek short-rate model** and Monte Carlo simulation.

The project explores mean-reverting interest rate dynamics and discounted payoff 
evaluation for fixed-income derivatives.

---

## 🧠 Overview

Core components:

- Short-rate simulation under the Vasicek model
- Monte Carlo pricing framework
- Yield curve calculation from simulated short rates
- Discounted payoff evaluation for cap contracts

The implementation demonstrates stochastic interest rate modeling and 
derivative pricing verification through repeated simulation.

---

## 📐 Model Setup
dr_t = κ(μ − r_t) dt + σ dW_t

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `r_0`     | Initial short rate |
| `kappa`   | Mean reversion speed |
| `mu`      | Long-term rate level |
| `sigma`   | Volatility |
| `delta`   | Time step size |
| `T`       | Contract maturity |
| `K`       | Cap strike |
| `R`       | Number of simulation paths |

---

## ⚙️ Implementation Details

Main workflow:

1. Simulate short-rate paths using discretized Vasicek dynamics.
2. Compute yields at reset dates.
3. Evaluate cap payoffs when the rate exceeds the strike.
4. Discount and average payoffs across simulation paths.

Helper function:

- `get_vasicek_yield()` — computes model-implied yield from short rates.

---

## 🚀 Usage

```bash
python vasicek_cap_pricing.py
```
The script prints the estimated cap price from Monte Carlo simulations.

## 🔧 Tech Stack

Python • NumPy • Monte Carlo Simulation • Quantitative Finance

## 📌 Context

This project complements my broader quantitative modeling work,
including econometric forecasting, reinforcement learning for asset management,
and applied machine learning in financial markets.

Short-rate dynamics:
