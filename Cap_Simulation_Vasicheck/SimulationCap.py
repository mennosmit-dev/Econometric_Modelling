# In this assignment we simulate the price of a cap, for a setting where the underlying floating interest
# rate is a three-year yield based on a 1-factor Vasicek model.

#Important: now t=0, T_0 = 1 first settlement date, T_1, T_2 ... T_T payout dates and reset dates (

#r_o = m
#T is in full years
#S is number of steps within year, to be calculated from delta

R= 10
kappa = 0.3
mu = 0.03
sigma = 0.01
r_0 = 0.03
delta = 0.1
T = 3
K = 0.02


from math import exp
import numpy as np


def main(kappa, mu, sigma, r_0, R, delta, T, K):
    """
    This function calculates the price of a cap

    Parameters:
        kappa (float): Is the parameter of mean reversion.
        mu (float): Is the expected value of the short rate.
        sigma (float): Is the volatility of the short rate.
        r_0 (float): Is the starting value of the short rate.
        R (int): Is the number of repititions of the simulation.
        delta (float): Is the step lenght within the year.
        T (int): Is the total time until the end of the contract.
        K (float): This is the strike of the cap.

    Returns:
        float: The average price of the cap.
    """
    S = int(1 / delta)
    short_rate_sdev = np.sqrt(((1 - exp(-2 * kappa * delta)) / (2 *kappa)) * sigma**2)
    tau = 3
    values = []
    for r in range(1, R + 1):
        r_t = r_0
        r_t_values = []
        totalValue = []
        for j in range(2, S + 1):
            var = short_rate_sdev * np.random.normal(0, 1)
            r_t = exp(-kappa * delta) * r_t + mu * (1-exp(-kappa * delta)) + var
            + r_t_values.append(r_t)
        for i in range(2, T + 1):
            currentSetYield = getVasicekYield(3, mu, sigma, r_t, tau)
            for j in range(1, S + 1):
                var = short_rate_sdev * np.random.normal(0, 1)
                r_t = exp(-kappa * delta) * r_t + mu * (1-exp(-kappa * delta)) + var
                r_t_values.append(r_t)
            totalValue.append(exp(-sum(r_t_values)*delta) * max(0, currentSetYield - K))
        values.append(sum(totalValue))
    return sum(values) / R


def getVasicekYield(kappa, mu, sigma, r_t, tau):

    """
    This function calculates the price of a cap

    Parameters:
        kappa (float): Is the parameter of mean reversion.
        mu (float): Is the expected value of the short rate.
        sigma (float): Is the volatility of the short rate.
        r_t (float): Is the value of the short rate at time t.
        tau (float): Is the time to maturity of the yield.

    Returns:
        float: The yield associated with the parameters.
    """
    beta = (exp(-kappa * tau) - 1) / kappa
    alpha = (beta + tau) * (sigma**2 / (2 * kappa**2) - mu) - (sigma**2 / (4 * kappa)) * beta**2
    return (-alpha - beta * r_t) / tau



endresult = main(kappa, mu, sigma, r_0, R, delta, T, K)
print(endresult)

