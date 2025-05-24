# Cap Pricing Using the Vasicek Model

This repository contains a Python implementation to simulate the price of an interest rate cap, where the underlying floating interest rate is modeled by a **1-factor Vasicek model**.

---

## 📋 Description

The code simulates the price of a cap contract based on a three-year yield. The Vasicek model captures the evolution of short-term interest rates with mean reversion.

- **t = 0**: Current time (now)
- **T₀ = 1**: First settlement date
- **T₁, T₂, ..., T_T**: Payout and reset dates

### Key Parameters

| Symbol | Meaning                              |
|--------|------------------------------------|
| `r_0`  | Initial short rate                  |
| `kappa`| Mean reversion rate                 |
| `mu`   | Long-term average short rate       |
| `sigma`| Volatility of the short rate       |
| `R`    | Number of simulation repetitions   |
| `delta`| Time step size within each year    |
| `T`    | Total time (years) until contract end |
| `K`    | Strike rate of the cap              |

---

## 💡 How It Works

The `main` function runs Monte Carlo simulations to estimate the cap price by:

1. Simulating short rate paths with mean reversion and volatility.
2. Calculating the Vasicek yield for payout dates.
3. Computing discounted payoffs when the yield exceeds the strike.
4. Averaging the payoffs over all simulation runs.

The helper function `get_vasicek_yield` computes the theoretical yield at time *t* given the short rate and model parameters.

---

## ⚙️ Usage

Run the script directly to see the estimated cap price printed out:

```bash
python vasicek_cap_pricing.py
