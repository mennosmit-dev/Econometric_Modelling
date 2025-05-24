# Portfolio Optimization & Performance Metrics 📈💼

This Python script calculates and compares different portfolio weights and their performance metrics based on a factor model and mean-variance optimization principles.

---

## 🔧 What it Does
It computes expected returns, variance, and certainty equivalent for portfolios, and calculates weights for Mean-Variance (MV), Certainty Equivalent (CE), Global Minimum Variance (GMV), and Equal Weight (EW) portfolios. Key metrics like expected return (`mu`), volatility (`vol`), Sharpe ratio (`sh`), and certainty equivalent (`ce`) are printed for each portfolio.

The script uses a factor model with a single factor (`beta`) influencing asset returns, calculates the covariance matrix combining factor variance and idiosyncratic risk, applies matrix operations for optimization, and includes a risk aversion parameter for the CE portfolio.

---

## 🧮 Parameters
- `N`: Number of assets (default 10)
- `beta`: Factor loadings for each asset
- `mu_f`: Expected factor return (0.1)
- `sigma_f`: Factor volatility (0.2)
- `error_covariance_matrix`: Diagonal matrix of idiosyncratic risk
- `risk_aversion`: Investor’s risk aversion (default 2)

---

## 💡 Portfolio Metrics

Portfolio metrics include expected return (`mu`), volatility (`vol`), Sharpe ratio (`sh`), and certainty equivalent (`ce`).

---

## 🚀 How to Run

To run, use Python 3 with NumPy installed:
```bash
    python portfolio_optimization.py
```

---

# Dependencies:
- Python 3.x
- NumPy (`pip install numpy`)

This script implements classical portfolio theory concepts based on CAPM and mean-variance optimization frameworks.


