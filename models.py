def black_scholes_model(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying stock (annualized)

    Returns:
    float : The Black-Scholes option price
    """
    from math import log, sqrt, exp
    from scipy.stats import norm

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    call_price = (S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    
    return call_price