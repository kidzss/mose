import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_stock_data():
    # 测试股票列表
    symbols = ['AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    
    # 设置时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("开始测试数据获取...")
    
    for symbol in symbols:
        print(f"\n测试 {symbol} 的数据获取:")
        try:
            # 获取数据
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            # 打印数据信息
            print(f"数据形状: {data.shape}")
            print(f"列名: {data.columns.tolist()}")
            print(f"前5行数据:\n{data.head()}")
            
        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {str(e)}")

if __name__ == "__main__":
    test_stock_data() 