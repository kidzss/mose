import yfinance as yf

def check_dividend_stocks():
    # 检查一些真正的高股息股票
    symbols = ['JPM', 'MRK', 'JNJ', 'KO', 'PG', 'VZ', 'T', 'XOM', 'CVX']
    
    print("股息率检查结果:")
    print("-" * 50)
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 获取股息率（更准确的方法）
            forward_dividend_yield = info.get('dividendYield')
            trailing_dividend_yield = info.get('trailingAnnualDividendYield')
            dividend_rate = info.get('dividendRate', 0)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            # 计算实际股息率
            if dividend_rate and current_price:
                actual_yield = (dividend_rate / current_price) * 100
            else:
                actual_yield = 0
                
            print(f"{symbol:4s}: 价格${current_price:.2f} | 年股息${dividend_rate:.2f} | 股息率{actual_yield:.2f}%")
            
        except Exception as e:
            print(f"{symbol}: 获取失败 - {e}")

if __name__ == "__main__":
    check_dividend_stocks() 