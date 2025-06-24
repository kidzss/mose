import yfinance as yf

def get_msft_price():
    try:
        ticker = yf.Ticker('MSFT')
        data = ticker.history(period='5d')
        current_price = data['Close'].iloc[-1]
        print(f"MSFT 当前价格: ${current_price:.2f}")
        
        # 获取基本信息
        info = ticker.info
        print(f"52周高点: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"52周低点: ${info.get('fiftyTwoWeekLow', 'N/A')}")
        print(f"市盈率: {info.get('trailingPE', 'N/A')}")
        
        return current_price
    except Exception as e:
        print(f"获取价格失败: {e}")
        return None

if __name__ == "__main__":
    get_msft_price() 