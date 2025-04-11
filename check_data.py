import pandas as pd
from datetime import datetime
from data.data_interface import MySQLDataSource

def check_stock_data():
    # 初始化数据源
    data_source = MySQLDataSource()
    
    # 要检查的股票列表
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'BABA', 'SOFI', 'PFE', 'BRK-B', 'ADBE', 'NKE', 'ELF']
    
    # 设置时间范围
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2025, 4, 1)
    
    print("\n=== 数据检查报告 ===")
    
    for symbol in symbols:
        try:
            # 获取数据
            data = data_source.get_historical_data(symbol, start_date, end_date)
            
            # 打印数据信息
            print(f"\n{symbol}:")
            print(f"数据条数: {len(data)}")
            if len(data) > 0:
                print(f"数据时间范围: {data.index.min()} 到 {data.index.max()}")
                print(f"列名: {list(data.columns)}")
                print(f"缺失值统计:")
                print(data.isnull().sum())
            else:
                print("没有找到数据")
                
        except Exception as e:
            print(f"{symbol}: 获取数据时出错 - {str(e)}")

if __name__ == "__main__":
    check_stock_data() 