import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time

# 不再需要pandas_datareader，直接使用yfinance


def get_data(stock, start_date):
    """获取股票数据"""
    try:
        now = dt.datetime.now()
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start_date, end=now)
        
        if data.empty:
            print(f"警告: 无法获取 {stock} 的数据")
            return None
            
        print(f"成功获取 {len(data)} 行数据")
        return data
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None


def calculate_sma(df, sma_periods):
    """计算移动平均线并添加到DataFrame"""
    if df is None or df.empty:
        print("数据为空，无法计算移动平均线")
        return None
        
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
    
    # 删除包含NaN的行，通常是前100行（因为SMA_100需要100个数据点）
    df.dropna(inplace=True)
    return df


def trading_strategy(df, no_risk=True):
    """执行黄金三角交易策略并返回收益率
    
    策略说明:
    1. 买入条件: 
       - SMA_5 > SMA_10 (短期均线上穿中期均线)
       - SMA_10 > SMA_20 (中期均线上穿长期均线)
       - 前一天 SMA_10 < SMA_20 (确认是交叉点)
       - 如果no_risk=False，则还需要价格 > SMA_100
    
    2. 卖出条件:
       - SMA_10 < SMA_20 (中期均线下穿长期均线)
    """
    if df is None or df.empty:
        print("数据为空，无法执行交易策略")
        return []
        
    # 确定使用哪个价格列
    if 'Adj Close' in df.columns:
        price_col = 'Adj Close'
    elif 'Close' in df.columns:
        price_col = 'Close'
        print("警告: 未找到'Adj Close'列，使用'Close'列进行回测")
    else:
        raise KeyError("数据框中既没有'Adj Close'也没有'Close'列")
    
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率
    trades = []  # 存储交易详情
    bp = 0  # 买入价格

    for i in range(len(df) - 1):
        current_date = df.index[i]
        moving_average_5 = df["SMA_5"].iloc[i + 1]
        moving_average_10 = df["SMA_10"].iloc[i + 1]
        moving_average_20 = df["SMA_20"].iloc[i + 1]
        moving_average_100 = df["SMA_100"].iloc[i + 1]
        close = df[price_col].iloc[i]

        # 定义买入和卖出条件
        cond1 = moving_average_5 > moving_average_10  # 短期均线上穿中期均线
        cond2 = moving_average_10 > moving_average_20  # 中期均线上穿长期均线
        cond3 = df["SMA_10"].iloc[i] < df["SMA_20"].iloc[i]  # 确认是交叉点
        cond4 = close > moving_average_100  # 股价大于SMA_100

        # 产生买入信号
        if cond1 and cond2 and cond3 and (cond4 or no_risk) and pos == 0:
            bp = close  # 记录买入价格
            pos = 1  # 持仓状态设置为1
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': bp,
                'conditions': f"SMA5 > SMA10: {cond1}, SMA10 > SMA20: {cond2}, 前日SMA10 < SMA20: {cond3}, 价格 > SMA100: {cond4}"
            })
        # 产生卖出信号
        elif pos == 1 and not cond2 and not cond3:
            pos = 0  # 清仓
            sp = close  # 记录卖出价格
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': sp,
                'profit_percent': pc,
                'conditions': f"SMA10 < SMA20: {not cond2}, 前日SMA10 < SMA20: {not cond3}"
            })

    # 检查是否有持仓未平仓
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
    """计算收益统计数据"""
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
    totalR = np.prod([(x / 100) + 1 for x in percent_change])
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


def check_signal(df, days_back=1, no_risk=True):
    """检查最近几天的数据是否触发买入或卖出信号
    
    参数:
    days_back: 检查最近几天的数据，默认为1（仅检查最后一天）
    """
    if df is None or df.empty or len(df) < days_back + 1:
        return "数据不足，无法检查信号"
        
    for i in range(days_back):
        if i >= len(df):
            break
            
        latest = df.iloc[-(i+1)]
        
        # 如果是检查最后一天以外的日期，需要前一天的数据
        if i+2 <= len(df):
            prev = df.iloc[-(i+2)]
        else:
            continue

        moving_average_5 = latest["SMA_5"]
        moving_average_10 = latest["SMA_10"]
        moving_average_20 = latest["SMA_20"]
        moving_average_100 = latest["SMA_100"]
        
        # 确定使用哪个价格列
        if 'Adj Close' in df.columns:
            close = latest["Adj Close"]
        else:
            close = latest["Close"]

        cond1 = moving_average_5 > moving_average_10
        cond2 = moving_average_10 > moving_average_20
        cond3 = prev["SMA_10"] < prev["SMA_20"]
        cond4 = close > moving_average_100

        day_str = "最后一天" if i == 0 else f"倒数第{i+1}天"

        # 判断买入信号
        if cond1 and cond2 and cond3 and (cond4 or no_risk):
            signal_details = f"""
            黄金三角买入信号 ({day_str}):
            - SMA_5 ({moving_average_5:.2f}) > SMA_10 ({moving_average_10:.2f}): {cond1}
            - SMA_10 ({moving_average_10:.2f}) > SMA_20 ({moving_average_20:.2f}): {cond2}
            - 前一天 SMA_10 < SMA_20: {cond3}
            - 价格 ({close:.2f}) > SMA_100 ({moving_average_100:.2f}): {cond4} {'(已忽略)' if no_risk else ''}
            """
            return signal_details

        # 判断卖出信号
        elif not cond2 and not cond3:
            signal_details = f"""
            卖出信号 ({day_str}):
            - SMA_10 ({moving_average_10:.2f}) < SMA_20 ({moving_average_20:.2f}): {not cond2}
            - 前一天 SMA_10 < SMA_20: {not cond3}
            """
            return signal_details

    # 如果检查了多天但没有发现信号
    days_str = "天" if days_back == 1 else f"{days_back}天"
    return f"最近{days_str}没有明确的买入或卖出信号"


def main():
    try:
        stock = input("Enter a stock ticker symbol: ")
        print(stock)

        start_year = int(input("Enter start year (default 2022): ") or "2022")
        start_month = int(input("Enter start month (default 1): ") or "1")
        start_day = int(input("Enter start day (default 1): ") or "1")
        
        no_risk = input("Ignore price > SMA100 condition? (y/n, default: n): ").lower() == 'y'
        
        start_date = dt.datetime(start_year, start_month, start_day)
        print(f"分析时间范围: {start_date.strftime('%Y-%m-%d')} 至今")
        print(f"风险控制 (价格 > SMA100): {'已禁用' if no_risk else '已启用'}")

        df = get_data(stock, start_date)
        if df is None:
            print("无法获取数据，程序退出")
            return
            
        sma_periods = [5, 10, 20, 100]  # SMA周期
        df = calculate_sma(df, sma_periods)
        if df is None or df.empty:
            print("无法计算移动平均线，程序退出")
            return

        percent_change, trades = trading_strategy(df, no_risk=no_risk)
        stats = calculate_statistics(percent_change)

        # 打印交易统计结果
        print("\nResults for " + stock + " going back to " + str(df.index[0]) + ", Sample size: " + str(
            stats["Win Trades"] + stats["Loss Trades"]) + " trades")
            
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 检查最近几天的信号
        signal = check_signal(df, days_back=3, no_risk=no_risk)
        print("\n" + signal)
        
        # 打印最近的SMA值
        latest = df.iloc[-1]
        print(f"\n最新SMA值 ({latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else latest.name}):")
        print(f"SMA_5: {latest['SMA_5']:.2f}")
        print(f"SMA_10: {latest['SMA_10']:.2f}")
        print(f"SMA_20: {latest['SMA_20']:.2f}")
        print(f"SMA_100: {latest['SMA_100']:.2f}")
        
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
