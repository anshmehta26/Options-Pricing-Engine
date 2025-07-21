import yfinance as yf

def get_market_price(ticker_symbol, strike, expiry, option_type):
    ticker = yf.Ticker(ticker_symbol)
    opt_chain = ticker.option_chain(expiry)

    df = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts
    price_row = df[df['strike'] == strike]

    if price_row.empty:
        return None  # no match found

    return price_row['lastPrice'].values[0]
