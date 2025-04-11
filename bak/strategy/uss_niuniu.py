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


def calculate_indicators(df):
    """计算主力成本线和买卖点"""
    if df is None or df.empty:
        print("数据为空，无法计算指标")
        return None
        
    # 检查必要的列是否存在
    required_columns = ['Close', 'Low', 'Open', 'High']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告: 数据中缺少 {col} 列")
            return None
    
    # 计算 MID 值
    df['MID'] = (3 * df['Close'] + df['Low'] + df['Open'] + df['High']) / 6

    # 计算牛线（主力成本线）
    weights = np.arange(20, 0, -1)  # 权重为20到1
    
    # 确保数据足够计算牛线
    if len(df) < 20:
        print(f"警告: 数据行数 ({len(df)}) 不足以计算牛线 (需要至少20行)")
        return None
        
    weighted_mid = pd.concat([df['MID'].shift(i) * weights[i] for i in range(20)], axis=1).sum(axis=1)
    df['Bull Line'] = weighted_mid / weights.sum()

    # 计算买卖线
    df['Trade Line'] = df['Bull Line'].rolling(window=2).mean()

    # 生成买卖信号
    df['Buy Signal'] = (df['Bull Line'] > df['Trade Line']) & (df['Bull Line'].shift(1) <= df['Trade Line'].shift(1))
    df['Sell Signal'] = (df['Trade Line'] > df['Bull Line']) & (df['Trade Line'].shift(1) <= df['Bull Line'].shift(1))

    # 删除NaN值
    df.dropna(inplace=True)
    
    return df


def backtest_strategy(df):
    """根据策略进行回测，返回收益率"""
    if df is None or df.empty:
        print("数据为空，无法执行回测")
        return [], []
        
    # 检查必要的列是否存在
    required_columns = ['Buy Signal', 'Sell Signal']
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
    percent_change = []  # 存储每次交易的收益率
    trades = []  # 存储交易详情

    for i in range(len(df) - 1):
        close = df[price_col].iloc[i]
        buy_signal = df['Buy Signal'].iloc[i]
        sell_signal = df['Sell Signal'].iloc[i]
        current_date = df.index[i]

        # 产生买入信号
        if buy_signal and pos == 0:
            bp = close
            pos = 1  # 开仓
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': bp,
                'reason': '牛线上穿交易线'
            })

        # 产生卖出信号
        elif sell_signal and pos == 1:
            sp = close
            pos = 0  # 平仓
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'price': sp,
                'profit_percent': pc,
                'reason': '牛线下穿交易线'
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


def check_last_day_signal(df, days_back=1):
    """检查最近几天的数据是否触发买入或卖出信号
    
    参数:
    days_back: 检查最近几天的数据，默认为1（仅检查最后一天）
    """
    if df is None or df.empty or 'Buy Signal' not in df.columns or 'Sell Signal' not in df.columns:
        return "数据不足或不包含信号列，无法检查信号"
        
    for i in range(days_back):
        if i >= len(df):
            break
            
        latest = df.iloc[-(i+1)]
        
        day_str = "最后一天" if i == 0 else f"倒数第{i+1}天"
        
        if latest['Buy Signal']:
            bull_line = latest['Bull Line']
            trade_line = latest['Trade Line']
            return f"买入信号 ({day_str}): 牛线 ({bull_line:.2f}) 上穿交易线 ({trade_line:.2f})"
        elif latest['Sell Signal']:
            bull_line = latest['Bull Line']
            trade_line = latest['Trade Line']
            return f"卖出信号 ({day_str}): 牛线 ({bull_line:.2f}) 下穿交易线 ({trade_line:.2f})"
    
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
            
        df = calculate_indicators(df)
        if df is None:
            print("无法计算指标，程序退出")
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
        last_day_signal = check_last_day_signal(df, days_back=3)
        print("\n" + last_day_signal)
        
        # 打印最新的指标值
        latest = df.iloc[-1]
        print(f"\n最新指标值 ({latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else latest.name}):")
        print(f"牛线 (Bull Line): {latest['Bull Line']:.2f}")
        print(f"交易线 (Trade Line): {latest['Trade Line']:.2f}")
        
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
