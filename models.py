from math import exp, sqrt, log
from scipy.stats import norm

def black_scholes_model(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying stock (annualized)
    option_type : str : "call" or "put"

    Returns:
    float : The Black-Scholes option price
    """

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


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
    Calculate the option price using the Binomial Tree (Cox-Ross-Rubinstein) model.

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
    - float : Option price
    """

    # Normalize and validate option_type
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    if N < 1:
        raise ValueError("N must be at least 1")

    dt = T / N
    u = exp(sigma * sqrt(dt))       # Up factor
    d = 1 / u                       # Down factor
    p = (exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Step 1: Terminal asset prices
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    # Step 2: Terminal option values
    if option_type == "call":
        option_values = [max(0, price - K) for price in asset_prices]
    else:
        option_values = [max(0, K - price) for price in asset_prices]

    # Step 3: Backward induction
    for i in reversed(range(N)):
        for j in range(i + 1):
            expected = exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])

            if american:
                spot = S * (u ** j) * (d ** (i - j))
                exercise = max(0, spot - K) if option_type == "call" else max(0, K - spot)
                option_values[j] = max(expected, exercise)
            else:
                option_values[j] = expected

    return option_values[0]
