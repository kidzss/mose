import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

# 不再需要pandas_datareader，直接使用yfinance


def get_data(stock, start_date, end_date):
    """获取股票数据"""
    try:
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"警告: 无法获取 {stock} 的数据")
            return None
            
        print(f"成功获取 {len(data)} 行数据")
        return data
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None


def calculate_sma(df, sma_periods):
    """计算并添加简单移动平均线(SMA)"""
    if df is None or df.empty:
        print("数据为空，无法计算SMA")
        return None
        
    # 确定使用哪个价格列
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列计算SMA")
    else:
        raise KeyError("数据框中既没有'Adj Close'也没有'Close'列")
    
    for period in sma_periods:
        df[f"SMA_{period}"] = df[price_col].rolling(window=period).mean()
    
    return df


def calculate_rsi(df, period=14):
    """计算相对强弱指数（RSI）"""
    if df is None or df.empty:
        print("数据为空，无法计算RSI")
        return None
        
    # 确定使用哪个价格列
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列计算RSI")
    else:
        raise KeyError("数据框中既没有'Adj Close'也没有'Close'列")
    
    delta = df[price_col].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def backtest_strategy(df):
    """动量策略回测，结合均线和RSI指标"""
    if df is None or df.empty:
        print("数据为空，无法执行回测")
        return [], []
        
    # 检查必要的列是否存在
    required_columns = ['SMA_5', 'SMA_10', 'RSI']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告: 数据中缺少 {col} 列")
            return [], []
            
    # 确定使用哪个价格列
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列进行回测")
    else:
        raise KeyError("数据框中既没有'Adj Close'也没有'Close'列")
    
    pos = 0  # 持仓状态
    percent_change = []  # 记录每次交易的收益
    trades = []  # 存储交易详情

    for i in range(1, len(df) - 1):
        close = df[price_col].iloc[i]
        moving_average_5 = df["SMA_5"].iloc[i]
        moving_average_10 = df["SMA_10"].iloc[i]
        rsi = df["RSI"].iloc[i]
        current_date = df.index[i]

        # 买入信号: 5日均线 > 10日均线 且 RSI > 50 且未持仓
        if moving_average_5 > moving_average_10 and rsi > 50 and pos == 0:
            bp = close  # 记录买入价格
            pos = 1  # 持仓状态设为1
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': bp,
                'conditions': f"SMA5 > SMA10: {moving_average_5 > moving_average_10}, RSI > 50: {rsi > 50}"
            })

        # 卖出信号: 5日均线 < 10日均线 或 RSI > 70（止盈） 或 RSI < 30（止损）
        elif pos == 1 and (moving_average_5 < moving_average_10 or rsi > 70 or rsi < 30):
            sp = close  # 记录卖出价格
            pos = 0  # 持仓状态设为0
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': sp,
                'profit_percent': pc,
                'conditions': f"SMA5 < SMA10: {moving_average_5 < moving_average_10}, RSI > 70: {rsi > 70}, RSI < 30: {rsi < 30}"
            })

    # 检查持仓未平仓
    if pos == 1:
        sp = df[price_col].iloc[-1]
        pc = (sp / bp - 1) * 100
        percent_change.append(pc)
        trades.append({
            'date': df.index[-1],
            'action': 'HOLD',
            'current_price': sp,
            'buy_price': bp,
            'unrealized_profit_percent': pc
        })

    return percent_change, trades


def calculate_statistics(percent_change):
    """计算并返回交易统计数据"""
    if not percent_change:
        return {
            "Batting Avg": 0,
            "Gain/loss ratio": "N/A",
            "Average Gain": 0,
            "Average Loss": 0,
            "Max Return": "N/A",
            "Max Loss": "N/A",
            "Total Return(%)": 0,
            "Win Trades": 0,
            "Loss Trades": 0
        }
        
    gains = [x for x in percent_change if x > 0]
    losses = [x for x in percent_change if x < 0]
    
    ng = len(gains)
    nl = len(losses)
    
    total_gains = sum(gains) if gains else 0
    total_losses = sum(losses) if losses else 0

    # 计算总收益率
    totalR = np.prod([(x / 100) + 1 for x in percent_change]) if percent_change else 1
    avgGain = total_gains / ng if ng > 0 else 0
    avgLoss = total_losses / nl if nl > 0 else 0
    maxR = max(percent_change) if percent_change else "N/A"
    maxL = min(percent_change) if percent_change else "N/A"
    ratio = avgGain / abs(avgLoss) if avgLoss != 0 and avgLoss < 0 else "N/A"
    battingAvg = ng / (ng + nl) if (ng + nl) > 0 else 0

    return {
        "Batting Avg": battingAvg,
        "Gain/loss ratio": ratio,
        "Average Gain": avgGain,
        "Average Loss": avgLoss,
        "Max Return": maxR,
        "Max Loss": maxL,
        "Total Return(%)": (totalR - 1) * 100,
        "Win Trades": ng,
        "Loss Trades": nl
    }


def check_signal(df, days_back=1):
    """检查最近几天的买入或卖出信号
    
    参数:
    days_back: 检查最近几天的数据，默认为1（仅检查最后一天）
    """
    if df is None or df.empty:
        return "数据不足，无法检查信号"
        
    # 检查必要的列是否存在
    required_columns = ['SMA_5', 'SMA_10', 'RSI']
    for col in required_columns:
        if col not in df.columns:
            return f"警告: 数据中缺少 {col} 列，无法检查信号"
    
    for i in range(days_back):
        if i >= len(df):
            break
            
        latest = df.iloc[-(i+1)]
        moving_average_5 = latest["SMA_5"]
        moving_average_10 = latest["SMA_10"]
        rsi = latest['RSI']
        
        day_str = "最后一天" if i == 0 else f"倒数第{i+1}天"

        # 判断买入信号
        if moving_average_5 > moving_average_10 and rsi > 50:
            signal_details = f"""
            买入信号 ({day_str}):
            - SMA_5 ({moving_average_5:.2f}) > SMA_10 ({moving_average_10:.2f})
            - RSI ({rsi:.2f}) > 50
            """
            return signal_details

        # 判断卖出信号
        elif moving_average_5 < moving_average_10 or rsi > 70 or rsi < 30:
            signal_details = f"""
            卖出信号 ({day_str}):
            - SMA_5 ({moving_average_5:.2f}) < SMA_10 ({moving_average_10:.2f}): {moving_average_5 < moving_average_10}
            - RSI > 70: {rsi > 70} (当前值: {rsi:.2f})
            - RSI < 30: {rsi < 30} (当前值: {rsi:.2f})
            """
            return signal_details
    
    days_str = "天" if days_back == 1 else f"{days_back}天"
    return f"最近{days_str}没有明确的买入或卖出信号"


def main():
    try:
        stock = input("Enter a stock ticker symbol: ")
        print(stock)

        # 设置回测时间范围
        start_year = int(input("Enter start year (default 2022): ") or "2022")
        start_month = int(input("Enter start month (default 1): ") or "1")
        start_day = int(input("Enter start day (default 1): ") or "1")
        
        start_date = dt.datetime(start_year, start_month, start_day)
        now = dt.datetime.now()
        
        print(f"分析时间范围: {start_date.strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}")

        # 获取数据和计算指标
        df = get_data(stock, start_date, now)
        if df is None:
            print("无法获取数据，程序退出")
            return
            
        sma_periods = [5, 10, 20, 100]
        df = calculate_sma(df, sma_periods)
        if df is None:
            print("无法计算SMA，程序退出")
            return
            
        df = calculate_rsi(df)
        if df is None:
            print("无法计算RSI，程序退出")
            return
            
        # 删除NaN值
        df.dropna(inplace=True)
        if df.empty:
            print("处理NaN值后数据为空，程序退出")
            return

        # 回测策略并计算收益率
        percent_change, trades = backtest_strategy(df)
        if not percent_change:
            print("没有产生任何交易信号，无法计算统计数据")
            return
            
        stats = calculate_statistics(percent_change)

        # 打印交易统计结果
        print("\nResults for " + stock + " going back to " + str(df.index[0]) + ", Sample size: " + str(
            len(percent_change)) + " trades")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 检查最后一天的信号
        last_day_signal = check_signal(df, days_back=3)
        print("\n" + last_day_signal)
        
        # 打印最新的指标值
        latest = df.iloc[-1]
        print(f"\n最新指标值 ({latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else latest.name}):")
        print(f"SMA_5: {latest['SMA_5']:.2f}")
        print(f"SMA_10: {latest['SMA_10']:.2f}")
        print(f"SMA_20: {latest['SMA_20']:.2f}")
        print(f"SMA_100: {latest['SMA_100']:.2f}")
        print(f"RSI: {latest['RSI']:.2f}")
        
        # 打印最近的交易
        if trades:
            print("\n最近的交易:")
            for trade in trades[-3:]:  # 显示最近3笔交易
                print(trade)
        
        print()
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")


if __name__ == "__main__":
    main()
