import yfinance as yf
from typing import Optional

def get_market_price(ticker_symbol: str, strike: float, expiry: str, option_type: str) -> Optional[float]:
    ticker = yf.Ticker(ticker_symbol)
    opt_chain = ticker.option_chain(expiry)
    
    df = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts
    price_row = df[df['strike'] == strike]

    if price_row.empty:
        return None

    return price_row['lastPrice'].values[0]

def get_expiry_dates(ticker_symbol: str) -> list[str]:
    """Returns a list of available expiration dates for the given ticker."""
    ticker = yf.Ticker(ticker_symbol)
    return ticker.options

def get_strike_prices(ticker_symbol: str, expiry: str, option_type: str = "call") -> list[float]:
    """Returns a list of available strike prices for the given expiration and option type."""
    ticker = yf.Ticker(ticker_symbol)
    opt_chain = ticker.option_chain(expiry)

    df = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts
    return df['strike'].tolist()

def get_current_stock_price(ticker_symbol: str) -> Optional[float]:
    """Returns the current market price of the stock."""
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="1d")
    if data.empty:
        return None
    return data['Close'].iloc[-1]
