<<<<<<< HEAD
from openbb import obb
import matplotlib.pyplot as plt

def main():
    # 获取标普500指数（用SPY ETF作为代表）的历史价格数据，数据来源yfinance
    result = obb.equity.price.historical(symbol="SPY", provider="yfinance")
    
    # 转换成pandas DataFrame
    df = result.to_df()
    
    # 计算简单移动平均线（20天和50天）
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # 打印最近5天数据
    print(df.tail())
    
    # 绘制收盘价和移动平均线图
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.plot(df.index, df['MA50'], label='MA50')
    plt.title('SPY Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
=======
from openbb import obb
import matplotlib.pyplot as plt

def main():
    # 获取标普500指数（用SPY ETF作为代表）的历史价格数据，数据来源yfinance
    result = obb.equity.price.historical(symbol="SPY", provider="yfinance")
    
    # 转换成pandas DataFrame
    df = result.to_df()
    
    # 计算简单移动平均线（20天和50天）
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # 打印最近5天数据
    print(df.tail())
    
    # 绘制收盘价和移动平均线图
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.plot(df.index, df['MA50'], label='MA50')
    plt.title('SPY Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
