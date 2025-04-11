import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt


def get_data(stock, start_date, end_date):
    """获取股票数据"""
    ticker = yf.Ticker(stock)
    data = ticker.history(start=start_date, end=end_date)
    return data


def calculate_sma(df, sma_periods):
    """计算并添加简单移动平均线(SMA)"""
    # 检查数据框中的列名
    print(f"数据框中的列: {df.columns.tolist()}")
    
    # 确定使用哪个列来计算SMA
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列计算SMA")
    else:
        raise KeyError("数据框中既没有'Adj Close'也没有'Close'列")
    
    for period in sma_periods:
        df[f"SMA_{period}"] = round(df[price_col].rolling(window=period).mean(), 3)
    return df


def calculate_indicators(df, intermediate_len=20, near_term_len=10, momentum_len=5):
    """计算Market Forecast指标的三条曲线：Momentum、NearTerm和Intermediate"""
    # 确保数据框中有必要的列
    required_columns = ['Low', 'High', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"数据框中缺少必要的列: '{col}'")
    
    # 计算Intermediate曲线 (长期)
    df['lowest_low_intermediate'] = df['Low'].rolling(window=intermediate_len).min()
    df['highest_high_intermediate'] = df['High'].rolling(window=intermediate_len).max()
    df['Intermediate'] = (df['Close'] - df['lowest_low_intermediate']) / (df['highest_high_intermediate'] - df['lowest_low_intermediate']) * 100
    
    # 计算NearTerm曲线 (中期)
    df['lowest_low_near_term'] = df['Low'].rolling(window=near_term_len).min()
    df['highest_high_near_term'] = df['High'].rolling(window=near_term_len).max()
    df['NearTerm'] = (df['Close'] - df['lowest_low_near_term']) / (df['highest_high_near_term'] - df['lowest_low_near_term']) * 100
    
    # 计算Momentum曲线 (短期)
    df['lowest_low_momentum'] = df['Low'].rolling(window=momentum_len).min()
    df['highest_high_momentum'] = df['High'].rolling(window=momentum_len).max()
    df['Momentum'] = (df['Close'] - df['lowest_low_momentum']) / (df['highest_high_momentum'] - df['lowest_low_momentum']) * 100
    
    # 计算反转信号
    # 底部区域定义为低于30，顶部区域定义为高于70
    bottom_zone = 30
    top_zone = 70
    
    # 检测底部反转（上升）
    df['Momentum_bottom_reversal'] = (df['Momentum'].shift(1) < df['Momentum']) & (df['Momentum'].shift(1) < bottom_zone)
    df['NearTerm_bottom_reversal'] = (df['NearTerm'].shift(1) < df['NearTerm']) & (df['NearTerm'].shift(1) < bottom_zone)
    df['Intermediate_bottom_reversal'] = (df['Intermediate'].shift(1) < df['Intermediate']) & (df['Intermediate'].shift(1) < bottom_zone)
    
    # 检测顶部反转（下降）
    df['Momentum_top_reversal'] = (df['Momentum'].shift(1) > df['Momentum']) & (df['Momentum'].shift(1) > top_zone)
    df['NearTerm_top_reversal'] = (df['NearTerm'].shift(1) > df['NearTerm']) & (df['NearTerm'].shift(1) > top_zone)
    df['Intermediate_top_reversal'] = (df['Intermediate'].shift(1) > df['Intermediate']) & (df['Intermediate'].shift(1) > top_zone)
    
    # 买入信号：三条曲线几乎同时在底部区域反转上升
    df['buy_signal'] = df['Momentum_bottom_reversal'] & df['NearTerm_bottom_reversal'] & df['Intermediate_bottom_reversal']
    
    # 卖出信号：三条曲线几乎同时在顶部区域反转下跌
    df['sell_signal'] = df['Momentum_top_reversal'] & df['NearTerm_top_reversal'] & df['Intermediate_top_reversal']
    
    return df


def backtest_strategy(df):
    """根据Market Forecast指标进行回测，返回收益率"""
    # 确保数据框中有必要的列
    required_columns = ['Close', 'buy_signal', 'sell_signal']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"数据框中缺少必要的列: '{col}'")
    
    # 如果数据框中有Adj Close列，使用它作为价格列，否则使用Close
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    else:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列进行回测")
    
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率
    bp = 0  # 买入价格

    for i in range(len(df) - 1):
        close = df[price_col].iloc[i]
        
        # 使用Market Forecast指标的买卖信号
        buy_signal = df['buy_signal'].iloc[i]
        sell_signal = df['sell_signal'].iloc[i]

        # 产生买入信号
        if buy_signal and pos == 0:
            bp = close
            pos = 1  # 开仓

        # 产生卖出信号
        elif pos == 1 and sell_signal:
            sp = close
            pos = 0  # 平仓
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)

    # 检查是否有持仓未平仓
    if pos == 1:
        sp = df[price_col].iloc[-1]
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
        "Total Return(%)": (totalR - 1) * 100
    }


"""检查最新一天的数据是否触发买入或卖出信号"""

def check_last_day_signal(df):
    """检查最新一天的数据是否触发Market Forecast买入或卖出信号"""
    # 确保数据框中有必要的列
    required_columns = ['Momentum', 'NearTerm', 'Intermediate', 'buy_signal', 'sell_signal']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"数据框中缺少必要的列: '{col}'")
    
    latest = df.iloc[-1]
    
    # 检查是否有买入信号
    if latest['buy_signal']:
        return "Buy signal on the last day: Market Forecast指标三条曲线在底部区域同时反转上升。"
    
    # 检查是否有卖出信号
    elif latest['sell_signal']:
        return "Sell signal on the last day: Market Forecast指标三条曲线在顶部区域同时反转下降。"
    
    # 如果没有明确的买卖信号，提供当前三条曲线的状态
    momentum = latest['Momentum']
    near_term = latest['NearTerm']
    intermediate = latest['Intermediate']
    
    return f"No clear buy or sell signal on the last day. Current Market Forecast values: Momentum={momentum:.2f}, NearTerm={near_term:.2f}, Intermediate={intermediate:.2f}"


def main():
    try:
        stock = input("Enter a stock ticker symbol: ")
        print(stock)

        # 设置回测时间范围
        start_date = dt.datetime(2022, 1, 1)
        now = dt.datetime.now()

        # 获取数据和计算指标
        print(f"获取 {stock} 的数据，时间范围: {start_date.strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}...")
        df = get_data(stock, start_date, now)
        
        if df.empty:
            print(f"错误: 无法获取 {stock} 的数据")
            return
            
        print(f"成功获取 {len(df)} 行数据")
        
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

        # 检查最后一天的信号
        last_day_signal = check_last_day_signal(df)
        print("\n" + last_day_signal)
        
        # 打印最后一天的Market Forecast指标值
        latest = df.iloc[-1]
        print(f"\nMarket Forecast指标值 (最后交易日):")
        print(f"Momentum: {latest['Momentum']:.2f}")
        print(f"NearTerm: {latest['NearTerm']:.2f}")
        print(f"Intermediate: {latest['Intermediate']:.2f}")
        print()
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")


if __name__ == "__main__":
    main()
