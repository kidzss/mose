import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import create_engine

# 数据库配置信息
from utils.get_uss_stocks_datas import get_stock_list_from_db

db_config = {
    "host": "localhost",  # 替换为你的数据库地址
    "port": 3306,  # 通常是3306
    "user": "root",  # 替换为你的用户名
    "password": "",  # 替换为你的密码
    "database": "mose"
}


def get_stock_info_from_db(stock, start_date, end_date):
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

    # cpgw
    # 计算指标 A、B 和 D
    df['A'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))
    df['A'] = df['A'].rolling(window=19).mean()

    df['B'] = df['Close'].rolling(window=14).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))

    df['D'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))
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

    return df


# -----

# def calculate_indicators_cpgw(df):
#     """计算长庄线、游资线、主力线及其买卖点"""
#     # 计算指标 A、B 和 D
#     df['A'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))
#     df['A'] = df['A'].rolling(window=19).mean()
#
#     df['B'] = df['Close'].rolling(window=14).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))
#
#     df['D'] = df['Close'].rolling(window=34).apply(lambda x: -100 * (x.max() - x.iloc[-1]) / (x.max() - x.min()))
#     df['D'] = df['D'].ewm(span=4).mean()
#
#     # 计算长庄线、游资线和主力线
#     df['Long Line'] = df['A'] + 100
#     df['Hot Money Line'] = df['B'] + 100
#     df['Main Force Line'] = df['D'] + 100
#
#     # 买卖信号定义
#     df['Sell Signal'] = ((df['Main Force Line'] < df['Main Force Line'].shift(1)) &
#                          (df['Main Force Line'].shift(1) > 80) &
#                          ((df['Hot Money Line'].shift(1) > 95) | (df['Hot Money Line'].shift(2) > 95)) &
#                          (df['Long Line'] > 60) &
#                          (df['Hot Money Line'] < 83.5) &
#                          (df['Hot Money Line'] < df['Main Force Line']) &
#                          (df['Hot Money Line'] < df['Main Force Line'] + 4))
#
#     df['Buy Signal'] = ((df['Long Line'] < 12) &
#                         (df['Main Force Line'] < 8) &
#                         ((df['Hot Money Line'] < 7.2) | (df['Main Force Line'].shift(1) < 5)) &
#                         ((df['Main Force Line'] > df['Main Force Line'].shift(1)) |
#                          (df['Hot Money Line'] > df['Hot Money Line'].shift(1)))) | \
#                        ((df['Long Line'] < 8) & (df['Main Force Line'] < 7) &
#                         (df['Hot Money Line'] < 15) & (df['Hot Money Line'] > df['Hot Money Line'].shift(1))) | \
#                        ((df['Long Line'] < 10) & (df['Main Force Line'] < 7) & (df['Hot Money Line'] < 1))
#
#     return df


def backtest_strategy_cpgw(df):
    """根据策略进行回测，返回收益率"""
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率

    for i in range(len(df) - 1):
        close = df["Adj Close"].iloc[i]
        buy_signal = df['Buy Signal'].iloc[i]
        sell_signal = df['Sell Signal'].iloc[i]

        # 产生买入信号
        if buy_signal and pos == 0:
            bp = close
            pos = 1  # 开仓

        # 产生卖出信号
        elif sell_signal and pos == 1:
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


# def calculate_statistics_cpgw(percent_change):
#     """计算并返回交易统计数据"""
#     gains = sum([x for x in percent_change if x > 0])
#     ng = len([x for x in percent_change if x > 0])
#     losses = sum([x for x in percent_change if x < 0])
#     nl = len([x for x in percent_change if x < 0])
#
#     # 计算总收益率
#     totalR = np.prod([(x / 100) + 1 for x in percent_change])
#     avgGain = gains / ng if ng > 0 else 0
#     avgLoss = losses / nl if nl > 0 else 0
#     maxR = max(percent_change) if percent_change else "undefined"
#     maxL = min(percent_change) if percent_change else "undefined"
#     ratio = avgGain / abs(avgLoss) if avgLoss != 0 else "inf"
#     battingAvg = ng / (ng + nl) if (ng + nl) > 0 else 0
#
#     return {
#         "Batting Avg": battingAvg,
#         "Gain/loss ratio": ratio,
#         "Average Gain": avgGain,
#         "Average Loss": avgLoss,
#         "Max Return": maxR,
#         "Max Loss": maxL,
#         "Total Return(%)": (totalR - 1) * 100
#     }


def check_last_day_signal_cpgw(df):
    """检查最新一天的数据是否触发买入或卖出信号"""
    latest = df.iloc[-1]
    if latest['Buy Signal']:
        return "Buy signal on the last day."
    elif latest['Sell Signal']:
        return "Sell signal on the last day."
    return "No clear buy or sell signal on the last day."


# ------


def backtest_strategy_momentum_risk(df):
    """动量策略回测，结合均线和RSI指标"""
    pos = 0  # 持仓状态
    percent_change = []  # 记录每次交易的收益

    for i in range(1, len(df) - 1):
        close = df["Adj Close"].iloc[i]
        moving_average_5 = df["SMA_5"].iloc[i]
        moving_average_10 = df["SMA_10"].iloc[i]
        rsi = df["RSI"].iloc[i]

        # 买入信号: 5日均线 > 10日均线 且 RSI > 50 且未持仓
        if moving_average_5 > moving_average_10 and rsi > 50 and pos == 0:
            bp = close  # 记录买入价格
            pos = 1  # 持仓状态设为1

        # 卖出信号: 5日均线 < 10日均线 或 RSI > 70（止盈） 或 RSI < 30（止损）
        elif pos == 1 and (moving_average_5 < moving_average_10 or rsi > 70 or rsi < 30):
            sp = close  # 记录卖出价格
            pos = 0  # 持仓状态设为0
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)

    # 检查持仓未平仓
    if pos == 1:
        sp = df["Adj Close"].iloc[-1]
        pc = (sp / bp - 1) * 100
        percent_change.append(pc)

    return percent_change


def backtest_strategy_gold_triangle(df):
    """黄金三角风险策略回测"""
    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率

    for i in range(len(df) - 1):
        close = df["Adj Close"].iloc[i]
        # 假设的策略逻辑，替换成实际逻辑
        signal = df["SMA_5"].iloc[i] > df["SMA_20"].iloc[i]

        # 买入信号
        if signal and pos == 0:
            bp = close
            pos = 1  # 开仓

        # 卖出信号
        elif pos == 1 and not signal:
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


def backtest_strategy_market_forecast(df):
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
    gains = sum(x for x in percent_change if x > 0)
    ng = len([x for x in percent_change if x > 0])
    losses = sum(x for x in percent_change if x < 0)
    nl = len([x for x in percent_change if x < 0])

    # 计算总收益率
    totalR = np.prod([(x / 100) + 1 for x in percent_change]) if percent_change else 1  # 确保不为空
    avgGain = gains / ng if ng > 0 else 0
    avgLoss = losses / nl if nl > 0 else 0
    maxR = max(percent_change) if percent_change else "undefined"
    maxL = min(percent_change) if percent_change else "undefined"

    # 盈亏比处理
    ratio = avgGain / abs(avgLoss) if avgLoss != 0 else None  # 如果 avgLoss 为 0，避免“inf”

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


def backtest_combined(stock, strategy, date):
    """综合回测"""
    # 设置回测时间范围
    start_date = dt.datetime(date, 1, 1)
    now = dt.datetime.now()

    # 获取数据和计算指标
    df = get_stock_info_from_db(stock, start_date, now)

    sma_periods = [5, 10, 20, 100]
    df = calculate_sma(df, sma_periods)
    df = calculate_rsi(df)
    df = calculate_indicators(df)

    # 分别回测两个策略并计算收益率
    if strategy == "momentum":
        percent_change = backtest_strategy_momentum_risk(df)
    elif strategy == "forecast":
        percent_change = backtest_strategy_market_forecast(df)
    elif strategy == "triangle":
        percent_change = backtest_strategy_gold_triangle(df)
    elif strategy == "cpgw":
        percent_change = backtest_strategy_cpgw(df)

    stats_data = calculate_statistics(percent_change)

    # 合并统计结果
    combined_stats = {
        "Stock Code": stock,
        "Batting Avg": stats_data["Batting Avg"],
        "Gain/loss ratio": stats_data["Gain/loss ratio"],
        "Average Gain": stats_data["Average Gain"],
        "Average Loss": stats_data["Average Loss"],
        "Max Return": stats_data["Max Return"],
        "Max Loss": stats_data["Max Loss"],
        "Total Return(%)": stats_data["Total Return(%)"]
    }

    return combined_stats


def main():
    # 读取股票列表
    stock_list = get_stock_list_from_db()
    date_year = 2022

    for strategy in ["momentum", "forecast", "triangle", "cpgw"]:
        results = []
        for stock in stock_list['Code']:
            result = backtest_combined(stock, strategy, date_year)
            results.append(result)

        # 将结果列表转换为 DataFrame
        results_df = pd.DataFrame(results)
        # 保存结果到 CSV 文件
        results_df.to_csv(f"{strategy}_nasdaq100_backtest_results.csv", index=False)

    print("Results have been saved to three backtest results.csv")


if __name__ == "__main__":
    main()
