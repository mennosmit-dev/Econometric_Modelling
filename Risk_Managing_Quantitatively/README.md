# Portfolio Risk Analysis and VaR Estimation

This repository contains Python scripts to analyze portfolio risk and estimate Value at Risk (VaR) using various methods including:

- Variance-Covariance approach  
- Historical Simulation  
- Normal Mixture Model (using EM algorithm)  
- Extreme Value Theory (EVT) with Hill estimator  
- GARCH(1,1) model  
- Clayton Copula method  

## Features

- Loads financial loss returns data from Excel  
- Computes portfolio mean, variance, and quantiles  
- Implements multiple VaR estimation techniques  
- Uses copulas and advanced statistical models for dependency structure  
- Includes visualization support (via Matplotlib)  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/portfolio-risk-analysis.git
cd portfolio-risk-analysis
```

2. Install dependencies:
   
```bash
pip install numpy pandas scipy arch matplotlib copulas openpyxl
```

## Usage 

1. Update the file path in the script to point to your Excel file with financial data:

```bash
df = pd.read_excel(
    r"C:\path\to\your\Data_QRM.xlsx",
    sheet_name="Data",
    skiprows=1,
    usecols=[0, 1]
)
```

2. Run the script:

```bash
python portfolio_risk_analysis.py
```

## Files

• portfolio_risk_analysis.py: Main script containing data loading, risk calculations, and VaR estimation methods.


## Methodology
### Variance-Covariance Approach
Calculates portfolio VaR assuming normally distributed returns based on covariance matrix.

### Historical Simulation
Computes empirical quantiles of historical portfolio returns.

### Normal Mixture Model
Fits a two-component normal mixture model using the Expectation-Maximization (EM) algorithm.

### Extreme Value Theory (EVT)
Uses Hill estimator to model tail behavior and estimate VaR.

### GARCH(1,1) Model
Fits a GARCH volatility model to estimate time-varying volatility and VaR.

### Clayton Copula
Models dependence structure between assets using Clayton copula and estimates VaR via simulation.

### License
This project is licensed under the MIT License.

Feel free to open issues or pull requests if you find bugs or want to contribute!
