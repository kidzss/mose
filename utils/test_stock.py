import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def test_stock_data(symbol):
    print(f"Testing stock: {symbol}")
    try:
        # 创建Ticker对象
        stock = yf.Ticker(symbol)
        
        # 获取当前日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 获取数据
        df = stock.history(
            start=start_date,
            end=end_date,
            interval="1d",
            prepost=False,
            actions=True,
            auto_adjust=False
        )
        
        if df.empty:
            print("返回空数据")
            return
            
        print("数据获取成功:")
        print(f"数据形状: {df.shape}")
        print("\n前5行数据:")
        print(df.head())
        print("\n列名:")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    # 测试几个知名股票
    test_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    
    for symbol in test_stocks:
        test_stock_data(symbol)
        print("\n" + "="*50 + "\n")
        time.sleep(2)  # 添加延时避免请求过快 