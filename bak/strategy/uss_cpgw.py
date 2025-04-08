import pandas as pd
import numpy as np
# import yfinance as yf
import datetime as dt
from sqlalchemy import create_engine
import traceback

# 数据库配置信息
db_config = {
    "host": "localhost",  # 替换为你的数据库地址
    "port": 3306,  # 通常是3306
    "user": "root",  # 替换为你的用户名
    "password": "",  # 替换为你的密码
    "database": "mose"
}

# 获取数据、计算指标的函数（保持不变）


def get_stock_info_from_db2(stock, start_date, end_date):
    """
    从 MySQL 数据库获取股票数据，并返回与 `pdr.get_data_yahoo` 格式一致的数据。

    :param stock: 股票代码
    :param start_date: 开始日期，格式 'YYYY-MM-DD'
    :param end_date: 结束日期，格式 'YYYY-MM-DD'
    :return: 包含股票数据的 Pandas DataFrame，列为 [High, Low, Open, Close, Volume, Adj Close]
    """
    query = """
    SELECT 
        Date AS `date`, 
        Open AS `Open`, 
        High AS `High`, 
        Low AS `Low`, 
        Close AS `Close`, 
        Volume AS `Volume`, 
        AdjClose AS `Adj Close`
    FROM stock_code_time
    WHERE Code = %s
    AND Date BETWEEN %s AND %s
    ORDER BY Date ASC;
    """
    try:
        # 创建 SQLAlchemy 引擎
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # 使用 Pandas 读取查询结果
        data = pd.read_sql_query(query, engine, params=(stock, start_date, end_date))

        if data.empty:
            print(f"警告: 无法获取 {stock} 的数据")
            return None

        # 将日期列设置为索引
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        print(f"成功获取 {len(data)} 行数据")
        return data
    except Exception as e:
        print(f"从数据库读取数据失败: {e}")
        print(traceback.format_exc())
        return None


def calculate_indicators(df):
    """计算长庄线、游资线、主力线及其买卖点"""
    if df is None or df.empty:
        print("数据为空，无法计算指标")
        return None
        
    # 检查必要的列是否存在
    required_columns = ['Close']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告: 数据中缺少 {col} 列")
            return None
    
    try:
        # 计算指标 A、B 和 D
        df['A'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))
        df['A'] = df['A'].rolling(window=19).mean()

        df['B'] = df['Close'].rolling(window=14).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))

        df['D'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min() + 1e-10))
        df['D'] = df['D'].ewm(span=4).mean()

        # 计算长庄线、游资线和主力线
        df['Long Line'] = df['A'] + 100
        df['Hot Money Line'] = df['B'] + 100
        df['Main Force Line'] = df['D'] + 100

        # 买卖信号定义
        df['Sell Signal'] = ((df['Main Force Line'] < df['Main Force Line'].shift(1)) &
                            (df['Main Force Line'].shift(1) > 80) &
                            ((df['Hot Money Line'].shift(1) > 95) | (df['Hot Money Line'].shift(2) > 95)) &
                            (df['Long Line'] > 60) &
                            (df['Hot Money Line'] < 83.5) &
                            (df['Hot Money Line'] < df['Main Force Line']) &
                            (df['Hot Money Line'] < df['Main Force Line'] + 4))

        df['Buy Signal'] = ((df['Long Line'] < 12) &
                            (df['Main Force Line'] < 8) &
                            ((df['Hot Money Line'] < 7.2) | (df['Main Force Line'].shift(1) < 5)) &
                            ((df['Main Force Line'] > df['Main Force Line'].shift(1)) |
                            (df['Hot Money Line'] > df['Hot Money Line'].shift(1)))) | \
                        ((df['Long Line'] < 8) & (df['Main Force Line'] < 7) &
                            (df['Hot Money Line'] < 15) & (df['Hot Money Line'] > df['Hot Money Line'].shift(1))) | \
                        ((df['Long Line'] < 10) & (df['Main Force Line'] < 7) & (df['Hot Money Line'] < 1))
        
        # 删除NaN值
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        print(f"计算指标时出错: {e}")
        print(traceback.format_exc())
        return None


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
                'reason': '长庄线、游资线、主力线满足买入条件'
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
                'reason': '长庄线、游资线、主力线满足卖出条件'
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
            long_line = latest['Long Line']
            hot_money_line = latest['Hot Money Line']
            main_force_line = latest['Main Force Line']
            return f"""
            买入信号 ({day_str}):
            - 长庄线 (Long Line): {long_line:.2f}
            - 游资线 (Hot Money Line): {hot_money_line:.2f}
            - 主力线 (Main Force Line): {main_force_line:.2f}
            """
        elif latest['Sell Signal']:
            long_line = latest['Long Line']
            hot_money_line = latest['Hot Money Line']
            main_force_line = latest['Main Force Line']
            return f"""
            卖出信号 ({day_str}):
            - 长庄线 (Long Line): {long_line:.2f}
            - 游资线 (Hot Money Line): {hot_money_line:.2f}
            - 主力线 (Main Force Line): {main_force_line:.2f}
            """
    
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
        df = get_stock_info_from_db2(stock, start_date, now)
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
        print(f"长庄线 (Long Line): {latest['Long Line']:.2f}")
        print(f"游资线 (Hot Money Line): {latest['Hot Money Line']:.2f}")
        print(f"主力线 (Main Force Line): {latest['Main Force Line']:.2f}")
        
        # 打印最近的交易
        if trades:
            print("\n最近的交易:")
            for trade in trades[-3:]:  # 显示最近3笔交易
                print(trade)
        
        print()
    except Exception as e:
        print(f"程序执行过程中出错: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
