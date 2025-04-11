import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import datetime as dt
from datetime import datetime

# Alpha Vantage API Key
API_KEY = "OICCCQG9R742HYZ1"

# 设置 Alpha Vantage 数据接口
ts = TimeSeries(key=API_KEY, output_format='pandas')

near_day = 5


def get_data(stock, start_date, end_date):
    """从 Alpha Vantage 获取股票数据"""
    try:
        df, _ = ts.get_daily_adjusted(symbol=stock, outputsize='full')
        df = df.loc[start_date:end_date]
        df = df.rename(columns={
            '5. adjusted close': 'Adj Close',
            '2. high': 'High',
            '3. low': 'Low',
            '1. open': 'Open',
            '6. volume': 'Volume',
            '4. close': 'Close'
        })
        return df
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        return pd.DataFrame()


# 保持其他函数不变，包括 `calculate_sma`, `calculate_rsi`, `calculate_indicators` 等
# 如前面的代码提供的实现一样，进行指标计算和信号检查

# 综合回测函数
def backtest_combined_with_signals(stock, date):
    """综合回测函数，记录每个策略的买卖信号"""
    start_date = dt.datetime(date, 1, 1)
    now = dt.datetime.now().strftime('%Y-%m-%d')

    # 获取数据和计算指标
    df = get_data(stock, start_date, now)
    if df.empty:
        print(f"No data found for {stock}. Skipping.")
        return None

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
            if signals is None:
                continue
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

    print(f"Signal check results have been saved to {filename}.")


def main():
    date_year = 2022
    for file in ["nasdaq100_stocks", "sp500_stocks", "uss_etf_stocks", "high_dividend_stocks"]:
        get_signals(file, date_year)


if __name__ == "__main__":
    main()
