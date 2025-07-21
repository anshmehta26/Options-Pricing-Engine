from math import exp, sqrt, log
from scipy.stats import norm

def black_scholes_model(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> dict:
    """
    Calculate the Black-Scholes option price and Greeks.

    Returns a dictionary with:
    - price
    - delta
    - gamma
    - theta
    - vega
    - rho
    """

    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nnd1 = norm.cdf(-d1)
    Nnd2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        price = S * Nd1 - K * exp(-r * T) * Nd2
        delta = Nd1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)) - r * K * exp(-r * T) * Nd2
        rho = K * T * exp(-r * T) * Nd2
    else:
        price = K * exp(-r * T) * Nnd2 - S * Nnd1
        delta = Nd1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)) + r * K * exp(-r * T) * Nnd2
        rho = -K * T * exp(-r * T) * Nnd2

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }


def binomial_tree_model(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int = 100,
    option_type: str = "call",
    american: bool = False
) -> float:
    """
    Binomial Tree option pricing model with Delta and Gamma.

    Parameters:
    - S : float
        Current stock price
    - K : float
        Strike price
    - T : float
        Time to expiration (in years)
    - r : float
        Annualized risk-free interest rate
    - sigma : float
        Annualized volatility of the underlying stock
    - N : int, default=100
        Number of steps in the binomial tree
    - option_type : str, "call" or "put"
        Type of the option
    - american : bool, default=False
        Whether to price as an American-style option

    Returns:
    - dict with price, delta, gamma
    """

    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if N < 2:
        raise ValueError("N must be at least 2 to compute Gamma")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)

    # Step 1: Asset prices at maturity
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    # Step 2: Option values at maturity
    if option_type == "call":
        option_values = [max(0, price - K) for price in asset_prices]
    else:
        option_values = [max(0, K - price) for price in asset_prices]

    # Step 3: Backward induction to root, save step 1 & 2 values for Greeks
    for i in reversed(range(N)):
        for j in range(i + 1):
            expected = exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])

            if american:
                spot = S * (u ** j) * (d ** (i - j))
                exercise = max(0, spot - K) if option_type == "call" else max(0, K - spot)
                option_values[j] = max(expected, exercise)
            else:
                option_values[j] = expected

        if i == 2:
            V_upup = option_values[2]
            V_updown = option_values[1]
            V_downdown = option_values[0]

    # Compute Delta
    S_up = S * u
    S_down = S * d
    delta = (option_values[1] - option_values[0]) / (S_up - S_down)

    # Compute Gamma
    S_upup = S * (u ** 2)
    S_mid = S
    S_downdown = S * (d ** 2)
    gamma = ((V_upup - V_updown) / (S_upup - S_mid) - (V_updown - V_downdown) / (S_mid - S_downdown)) / ((S_upup - S_downdown) / 2)

    return {
        "price": option_values[0],
        "delta": delta,
        "gamma": gamma
    }
