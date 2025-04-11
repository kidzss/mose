import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr

# 设置yfinance的覆盖
yf.pdr_override()


def get_data(stock, start_date):
    """获取股票数据"""
    now = dt.datetime.now()
    return pdr.get_data_yahoo(stock, start_date, now)


def calculate_sma(df, sma_periods):
    """计算移动平均线并添加到DataFrame"""
    for period in sma_periods:
        df[f"SMA_{period}"] = round(df["Adj Close"].rolling(window=period).mean(), 3)
    return df


# 对于一些杠杆产品例如tqqq，可以将risk设置为False，普通的股票可以使用默认值
def trading_strategy(df, no_risk=True):
    """执行交易策略并返回收益率"""
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率

    for i in range(len(df) - 1):
        moving_average_5 = df["SMA_5"].iloc[i + 1]
        moving_average_10 = df["SMA_10"].iloc[i + 1]
        moving_average_20 = df["SMA_20"].iloc[i + 1]
        moving_average_100 = df["SMA_100"].iloc[i + 1]
        close = df["Adj Close"].iloc[i]

        # 定义买入和卖出条件
        cond1 = moving_average_5 > moving_average_10
        cond2 = moving_average_10 > moving_average_20
        cond3 = df["SMA_10"].iloc[i] < df["SMA_20"].iloc[i]
        cond4 = close > moving_average_100  # 股价大于SMA_100

        # 产生买入信号
        if cond1 and cond2 and cond3 and (cond4 or no_risk) and pos == 0:
            bp = close  # 记录买入价格
            pos = 1  # 持仓状态设置为1
        # 产生卖出信号
        elif pos == 1 and not cond2 and not cond3:
            pos = 0  # 清仓
            sp = close  # 记录卖出价格
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)

    # 检查是否有持仓未平仓
    if pos == 1:
        sp = df["Adj Close"].iloc[-1]
        pc = (sp / bp - 1) * 100
        percent_change.append(pc)

    return percent_change


def calculate_statistics(percent_change):
    """计算收益统计数据"""
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

    return (battingAvg, ratio, avgGain, avgLoss, maxR, maxL, totalR, ng, nl)


def check_last_day_signal(df, no_risk=True):
    """检查最新一天的数据是否触发买入或卖出信号"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    moving_average_5 = latest["SMA_5"]
    moving_average_10 = latest["SMA_10"]
    moving_average_20 = latest["SMA_20"]
    moving_average_100 = latest["SMA_100"]
    close = latest["Adj Close"]

    cond1 = moving_average_5 > moving_average_10
    cond2 = moving_average_10 > moving_average_20
    cond3 = prev["SMA_10"] < prev["SMA_20"]
    cond4 = close > moving_average_100  # 股价大于SMA_100

    # 判断买入信号
    if cond1 and cond2 and cond3 and (cond4 or no_risk):
        return "Buy signal on the last day."

    # 判断卖出信号
    elif not cond2 and not cond3:
        return "Sell signal on the last day."

    return "No clear buy or sell signal on the last day."


def main():
    stock = input("Enter a stock ticker symbol: ")
    print(stock)

    start_year = 2022
    start_month = 1
    start_day = 1
    start_date = dt.datetime(start_year, start_month, start_day)

    df = get_data(stock, start_date)
    sma_periods = [5, 10, 20, 100]  # SMA周期
    df = calculate_sma(df, sma_periods)

    percent_change = trading_strategy(df, no_risk=True)
    stats = calculate_statistics(percent_change)

    # 打印交易统计结果
    print("\nResults for " + stock + " going back to " + str(df.index[0]) + ", Sample size: " + str(
        stats[7] + stats[8]) + " trades")
    print("Batting Avg: " + str(stats[0]))
    print("Gain/loss ratio: " + str(stats[1]))
    print("Average Gain: " + str(stats[2]))
    print("Average Loss: " + str(stats[3]))
    print("Max Return: " + str(stats[4]))
    print("Max Loss: " + str(stats[5]))
    print("Total return over " + str(stats[7] + stats[8]) + " trades: " + str((stats[6] - 1) * 100) + "%")

    # 检查最后一天的信号
    last_day_signal = check_last_day_signal(df, no_risk=True)
    print("\n" + last_day_signal)
    print()


if __name__ == "__main__":
    main()
