import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr

# 设置yfinance的覆盖
yf.pdr_override()


def get_data(stock, start_date, end_date):
    """获取股票数据"""
    return pdr.get_data_yahoo(stock, start_date, end_date)


def calculate_sma(df, sma_periods):
    """计算并添加简单移动平均线(SMA)"""
    for period in sma_periods:
        df[f"SMA_{period}"] = round(df["Adj Close"].rolling(window=period).mean(), 3)
    return df


def calculate_indicators(df, med_len=31, mom_len=5, near_len=3):
    """计算动量和聚类指标"""
    df['lowest_low_med'] = df['Low'].rolling(window=med_len).min()
    df['highest_high_med'] = df['High'].rolling(window=med_len).max()
    df['fastK_I'] = (df['Close'] - df['lowest_low_med']) / (df['highest_high_med'] - df['lowest_low_med']) * 100

    df['lowest_low_near'] = df['Low'].rolling(window=near_len).min()
    df['highest_high_near'] = df['High'].rolling(window=near_len).max()
    df['fastK_N'] = (df['Close'] - df['lowest_low_near']) / (df['highest_high_near'] - df['lowest_low_near']) * 100

    min1 = df['Low'].rolling(window=4).min()
    max1 = df['High'].rolling(window=4).max()
    df['momentum'] = ((df['Close'] - min1) / (max1 - min1)) * 100

    df['bull_cluster'] = (df['momentum'] <= 20) & (df['fastK_I'] <= 20) & (df['fastK_N'] <= 20)
    df['bear_cluster'] = (df['momentum'] >= 80) & (df['fastK_I'] >= 80) & (df['fastK_N'] >= 80)

    return df


def backtest_strategy(df):
    """根据策略进行回测，返回收益率"""
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率

    for i in range(len(df) - 1):
        close = df["Adj Close"].iloc[i]
        moving_average_5 = df["SMA_5"].iloc[i + 1]
        moving_average_10 = df["SMA_10"].iloc[i + 1]

        bull_cluster = df['bull_cluster'].iloc[i]
        bear_cluster = df['bear_cluster'].iloc[i]

        # 产生买入信号
        if bull_cluster and moving_average_5 > moving_average_10 and pos == 0:
            bp = close
            pos = 1  # 开仓

        # 产生卖出信号
        elif pos == 1 and (bear_cluster or moving_average_5 < moving_average_10):
            sp = close
            pos = 0  # 平仓
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)

    # 检查是否有持仓未平仓
    if pos == 1:
        sp = df["Adj Close"].iloc[-1]
        pc = (sp / bp - 1) * 100
        percent_change.append(pc)

    return percent_change


def calculate_statistics(percent_change):
    """计算并返回交易统计数据"""
    gains = sum([x for x in percent_change if x > 0])
    ng = len([x for x in percent_change if x > 0])
    losses = sum([x for x in percent_change if x < 0])
    nl = len([x for x in percent_change if x < 0])

    # 计算总收益率
    totalR = np.prod([(x / 100) + 1 for x in percent_change])
    avgGain = gains / ng if ng > 0 else 0
    avgLoss = losses / nl if nl > 0 else 0
    maxR = max(percent_change) if percent_change else "undefined"
    maxL = min(percent_change) if percent_change else "undefined"
    ratio = avgGain / abs(avgLoss) if avgLoss != 0 else "inf"
    battingAvg = ng / (ng + nl) if (ng + nl) > 0 else 0

    return {
        "Batting Avg": battingAvg,
        "Gain/loss ratio": ratio,
        "Average Gain": avgGain,
        "Average Loss": avgLoss,
        "Max Return": maxR,
        "Max Loss": maxL,
        "Total Return": (totalR - 1) * 100
    }


def main():
    stock = input("Enter a stock ticker symbol: ")
    print(stock)

    # 设置回测时间范围
    start_date = dt.datetime(2022, 1, 1)
    now = dt.datetime.now()

    # 获取数据和计算指标
    df = get_data(stock, start_date, now)
    sma_periods = [5, 10, 20, 100]
    df = calculate_sma(df, sma_periods)
    df = calculate_indicators(df)

    # 回测策略并计算收益率
    percent_change = backtest_strategy(df)
    stats = calculate_statistics(percent_change)

    # 打印交易统计结果
    print("\nResults for " + stock + " going back to " + str(df.index[0]) + ", Sample size: " + str(
        len(percent_change)) + " trades")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()


if __name__ == "__main__":
    main()
