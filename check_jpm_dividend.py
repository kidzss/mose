import yfinance as yf

def check_jpm_info():
    try:
        ticker = yf.Ticker('JPM')
        data = ticker.history(period='5d')
        current_price = data['Close'].iloc[-1]
        
        info = ticker.info
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        dividend_rate = info.get('dividendRate', 0)
        
        print(f"JPM 当前价格: ${current_price:.2f}")
        print(f"股息率: {dividend_yield:.2f}%")
        print(f"年股息: ${dividend_rate:.2f}")
        print(f"市盈率: {info.get('trailingPE', 'N/A')}")
        print(f"52周高点: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52周低点: ${info.get('fiftyTwoWeekLow', 'N/A')}")
        
        return current_price, dividend_yield
    except Exception as e:
        print(f"获取信息失败: {e}")
        return None, None

if __name__ == "__main__":
    check_jpm_info() 