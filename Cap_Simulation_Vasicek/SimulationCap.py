import numpy as np
from math import exp


# In this assignment, we simulate the price of a cap where the underlying floating interest
# rate is a three-year yield based on a 1-factor Vasicek model.

# Important:
# t = 0 (now),
# T_0 = 1 (first settlement date),
# T_1, T_2, ..., T_T payout dates and reset dates.

# r_0 = initial short rate
# T is in full years
# S is the number of steps within each year, calculated from delta


R = 10
kappa = 0.3
mu = 0.03
sigma = 0.01
r_0 = 0.03
delta = 0.1
T = 3
K = 0.02


def main(kappa, mu, sigma, r_0, R, delta, T, K):
    """
    Calculate the price of a cap using the Vasicek model.

    Parameters:
        kappa (float): Mean reversion rate.
        mu (float): Long-term mean short rate.
        sigma (float): Volatility of the short rate.
        r_0 (float): Initial short rate.
        R (int): Number of simulation repetitions.
        delta (float): Time step size within a year.
        T (int): Total time in years until contract end.
        K (float): Strike rate of the cap.

    Returns:
        float: Average price of the cap.
    """
    S = int(1 / delta)
    short_rate_sdev = np.sqrt(((1 - exp(-2 * kappa * delta)) / (2 * kappa)) * sigma ** 2)
    tau = 3
    values = []

    for _ in range(R):
        r_t = r_0
        r_t_values = []
        total_value = []

        for _ in range(2, S + 1):
            var = short_rate_sdev * np.random.normal(0, 1)
            r_t = exp(-kappa * delta) * r_t + mu * (1 - exp(-kappa * delta)) + var
            r_t_values.append(r_t)

        for _ in range(2, T + 1):
            current_set_yield = get_vasicek_yield(kappa, mu, sigma, r_t, tau)
            for _ in range(1, S + 1):
                var = short_rate_sdev * np.random.normal(0, 1)
                r_t = exp(-kappa * delta) * r_t + mu * (1 - exp(-kappa * delta)) + var
                r_t_values.append(r_t)

            discount_factor = exp(-sum(r_t_values) * delta)
            payoff = max(0, current_set_yield - K)
            total_value.append(discount_factor * payoff)

        values.append(sum(total_value))

    return sum(values) / R


def get_vasicek_yield(kappa, mu, sigma, r_t, tau):
    """
    Calculate the Vasicek yield.

    Parameters:
        kappa (float): Mean reversion rate.
        mu (float): Long-term mean short rate.
        sigma (float): Volatility of the short rate.
        r_t (float): Short rate at time t.
        tau (float): Time to maturity of the yield.

    Returns:
        float: Yield at time t.
    """
    beta = (exp(-kappa * tau) - 1) / kappa
    alpha = (
        (beta + tau) * (sigma ** 2 / (2 * kappa ** 2) - mu)
        - (sigma ** 2 / (4 * kappa)) * beta ** 2
    )
    return (-alpha - beta * r_t) / tau


if __name__ == "__main__":
    result = main(kappa, mu, sigma, r_0, R, delta, T, K)
    print(result)
