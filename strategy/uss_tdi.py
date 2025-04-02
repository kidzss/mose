import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

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


def calculate_rsi(df, period):
    """计算 RSI 指标"""
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
    
    close_diff = df[price_col].diff()
    temp1 = close_diff.clip(lower=0)
    temp2 = close_diff.abs()
    rsi = temp1.rolling(window=period).mean() / temp2.rolling(window=period).mean() * 100
    return rsi


def calculate_ma(df, period, column='RSI'):
    """计算移动平均线"""
    if df is None or df.empty or column not in df.columns:
        print(f"数据为空或不包含{column}列，无法计算移动平均线")
        return None
    return df[column].rolling(window=period).mean()


def trading_strategy(df, no_risk=True):
    """执行交易策略并返回收益率"""
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
    
    # 参数设置
    M = 14
    N1 = 34
    N2 = 7

    # 计算RSI和移动平均线
    df['RSI'] = calculate_rsi(df, M)
    df['MA1'] = calculate_ma(df, N1, column='RSI')
    df['MA2'] = calculate_ma(df, N2, column='RSI')

    # 生成趋势线
    df['TREND'] = calculate_ma(df, N1, column='RSI')

    # 计算买卖信号
    df['Signal'] = None  # 初始化信号列
    for i in range(1, len(df)):
        cond = df['MA2'].iloc[i] < df['MA2'].iloc[i - 1]
        prev_cond = df['MA2'].iloc[i - 1] < df['MA2'].iloc[i - 2] if i > 1 else False

        # 买入信号
        if cond and not prev_cond:
            df.loc[df.index[i], 'Signal'] = 'Buy'

        # 卖出信号
        elif not cond and prev_cond:
            df.loc[df.index[i], 'Signal'] = 'Sell'

    # 初始化持仓状态和收益
    pos = 0  # 0表示空仓，1表示持仓
    percent_change = []  # 存储每次交易的收益率
    trades = []  # 存储交易详情

    for i in range(1, len(df) - 1):
        cond = df['MA2'].iloc[i] < df['MA2'].iloc[i - 1]
        prev_cond = df['MA2'].iloc[i - 1] < df['MA2'].iloc[i - 2] if i > 1 else False

        close = df[price_col].iloc[i]
        current_date = df.index[i]

        # 产生买入信号
        if cond and not prev_cond and pos == 0:
            bp = close  # 记录买入价格
            pos = 1  # 持仓状态设置为1
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': bp,
                'conditions': f"MA2下降: {cond}, 前日MA2上升: {not prev_cond}"
            })

        # 产生卖出信号
        elif not cond and prev_cond and pos == 1:
            sp = close  # 记录卖出价格
            pos = 0  # 清仓
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': sp,
                'profit_percent': pc,
                'conditions': f"MA2上升: {not cond}, 前日MA2下降: {prev_cond}"
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


def check_last_day_signal(df, days_back=1):
    """检查最近几天的数据是否触发买入或卖出信号
    
    参数:
    days_back: 检查最近几天的数据，默认为1（仅检查最后一天）
    """
    if df is None or df.empty or 'Signal' not in df.columns:
        return "数据不足或不包含Signal列，无法检查信号"
        
    for i in range(days_back):
        if i >= len(df):
            break
            
        latest = df.iloc[-(i+1)]
        latest_signal = latest['Signal']
        
        day_str = "最后一天" if i == 0 else f"倒数第{i+1}天"
        
        if latest_signal == 'Buy':
            return f"买入信号 ({day_str}): RSI趋势反转"
        elif latest_signal == 'Sell':
            return f"卖出信号 ({day_str}): RSI趋势反转"
    
    days_str = "天" if days_back == 1 else f"{days_back}天"
    return f"最近{days_str}没有明确的买入或卖出信号"


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


def main():
    try:
        stock = input("Enter a stock ticker symbol: ")
        print(stock)

        start_year = int(input("Enter start year (default 2022): ") or "2022")
        start_month = int(input("Enter start month (default 1): ") or "1")
        start_day = int(input("Enter start day (default 1): ") or "1")
        
        start_date = dt.datetime(start_year, start_month, start_day)
        print(f"分析时间范围: {start_date.strftime('%Y-%m-%d')} 至今")

        df = get_data(stock, start_date)
        if df is None:
            print("无法获取数据，程序退出")
            return

        # 计算RSI和MA指标
        percent_change, trades = trading_strategy(df)
        
        if not percent_change:
            print("没有产生任何交易信号，无法计算统计数据")
            return

        # 计算统计数据
        stats = calculate_statistics(percent_change)

        # 打印交易统计结果
        print("\nResults for " + stock + " going back to " + str(df.index[0]) + ", Sample size: " + str(
            len(percent_change)) + " trades")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 检查最后一天的信号
        last_day_signal = check_last_day_signal(df, days_back=3)
        print("\n" + last_day_signal)
        
        # 打印最新的RSI和MA值
        latest = df.iloc[-1]
        print(f"\n最新指标值 ({latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else latest.name}):")
        print(f"RSI: {latest['RSI']:.2f}")
        print(f"MA1: {latest['MA1']:.2f}")
        print(f"MA2: {latest['MA2']:.2f}")
        
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
