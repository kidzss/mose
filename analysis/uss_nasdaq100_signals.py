import datetime as dt
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

near_day = 3

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

        # 将日期列设置为索引
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        return data
    except Exception as e:
        print(f"从数据库读取数据失败: {e}")
        return pd.DataFrame()  # 返回空的 DataFrame 以防止程序中断

def calculate_sma(df, sma_periods):
    """计算并添加简单移动平均线(SMA)"""
    for period in sma_periods:
        df[f"SMA_{period}"] = df["Adj Close"].rolling(window=period).mean()
    return df


def calculate_rsi(df, period=14):
    """计算相对强弱指数（RSI）"""
    delta = df['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
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


def check_near_day_signal_momentum(df):
    """检查最新两天的数据是否触发动量策略的买入或卖出信号"""
    for i in range(1, near_day):
        latest = df.iloc[-i]
        # prev = df.iloc[-2]

        moving_average_5 = latest["SMA_5"]
        moving_average_10 = latest["SMA_10"]
        rsi = latest["RSI"]

        # 判断买入信号
        if moving_average_5 > moving_average_10 and rsi > 50:
            return "Buy"

        # 判断卖出信号
        elif moving_average_5 < moving_average_10 or rsi > 70 or rsi < 30:
            return "Sell"

    return "None"


def check_near_day_signal_forecast(df):
    """检查最新两天的数据是否触发市场预测策略的买入或卖出信号"""
    for i in range(1, near_day):
        latest = df.iloc[-i]
        # prev = df.iloc[-2]

        moving_average_5 = latest["SMA_5"]
        moving_average_10 = latest["SMA_10"]
        bull_cluster = latest["bull_cluster"]
        bear_cluster = latest["bear_cluster"]

        # 判断买入信号
        if bull_cluster and moving_average_5 > moving_average_10:
            return "Buy"

        # 判断卖出信号
        elif bear_cluster or moving_average_5 < moving_average_10:
            return "Sell"

    return "None"


def check_near_day_signal_triangle(df):
    """检查最新两天的数据是否触发黄金三角策略的买入或卖出信号"""
    for i in range(1, near_day):
        latest = df.iloc[-i]
        prev = df.iloc[-(i + 1)]

        moving_average_5 = latest["SMA_5"]
        moving_average_10 = latest["SMA_10"]
        moving_average_20 = latest["SMA_20"]
        moving_average_100 = latest["SMA_100"]
        close = latest["Adj Close"]

        cond1 = moving_average_5 > moving_average_10
        cond2 = moving_average_10 > moving_average_20
        cond3 = prev["SMA_10"] < prev["SMA_20"]
        cond4 = close > moving_average_100

        # 判断买入信号
        if cond1 and cond2 and cond3 and cond4:
            return "Buy"

        # 判断卖出信号
        elif not cond2 and not cond3:
            return "Sell"

    return "None"


def backtest_combined_with_signals(stock, date):
    """综合回测函数，记录每个策略的买卖信号"""
    start_date = dt.datetime(date, 1, 1)
    now = dt.datetime.now()

    # 获取数据和计算指标
    df = get_stock_info_from_db2(stock, start_date, now)
    sma_periods = [5, 10, 20, 100]
    df = calculate_sma(df, sma_periods)
    df = calculate_rsi(df)
    df = calculate_indicators(df)

    # 检查最近买卖信号
    signals = {
        "momentum": check_near_day_signal_momentum(df),
        "forecast": check_near_day_signal_forecast(df),
        "triangle": check_near_day_signal_triangle(df)
    }

    return signals


def get_signals(file_name, date_year):
    stock_list = pd.read_csv(f"../stock_pool/{file_name}.csv")

    all_signals = []

    for stock in stock_list['Code']:
        try:
            signals = backtest_combined_with_signals(stock, date_year)
            signals["Stock Code"] = stock
            all_signals.append(signals)

        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
            return None

    # 将所有信号保存到一个CSV文件
    signals_df = pd.DataFrame(all_signals)
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y%m%d_%H")
    # 将当前时间添加到文件名
    filename = f"{file_name}_signals_last_two_days_{current_time}.csv"

    # 保存文件
    signals_df.to_csv(filename, index=False)

    print(f"Signal check results have been saved to {file_name}_signals_last_two_days.csv.")


def main():
    date_year = 2022
    for file in ["nasdaq100_stocks", "sp500_stocks", "high_dividend_stocks"]:
        get_signals(file, date_year)


if __name__ == "__main__":
    main()
