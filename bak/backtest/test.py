import yfinance as yf

stocks = ['AAPL', 'MSFT', 'AMZN']

data = yf.download(stocks, start="2020-01-01", end="2024-11-25")
print(data)

aapl = yf.Ticker("AAPL")

info = aapl.info
