import futu as ft
import pandas as pd
import numpy as np
import talib
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from futu import OpenQuoteContext, RET_OK
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')

# 设置Futu API
quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)


# 获取指定股票的历史K线数据
def get_stock_data(stock_code, start_date, end_date):
    try:
        ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=end_date,
                                                                  max_count=None)
        if ret != RET_OK:
            raise Exception(f"Error retrieving data: {data}")

        all_data = data

        while page_req_key:
            ret, data, page_req_key = quote_ctx.request_history_kline(stock_code, start=start_date, end=end_date,
                                                                      max_count=None, page_req_key=page_req_key)
            if ret != RET_OK:
                print(f"Error retrieving data: {data}")
                break
            all_data = all_data.append(data, ignore_index=True)

        return all_data
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


# 计算均线策略
def apply_strategy(data, no_risk=True):
    sma_used = [5, 10, 20, 100]
    for sma in sma_used:
        data[f"SMA_{sma}"] = data['close'].rolling(window=sma).mean()

    pos = 0  # 持仓状态
    percent_change = []  # 存储每次交易的收益率
    buy_dates = []  # 存储每次买入的日期
    sell_dates = []  # 存储每次卖出的日期

    # 遍历数据并应用策略
    for i in range(len(data) - 1):
        moving_average_5 = data["SMA_5"].iloc[i + 1]
        moving_average_10 = data["SMA_10"].iloc[i + 1]
        moving_average_20 = data["SMA_20"].iloc[i + 1]
        moving_average_100 = data["SMA_100"].iloc[i + 1]

        moving_average_10_1past = data["SMA_10"].iloc[i]
        moving_average_20_1past = data["SMA_20"].iloc[i]
        close = data["close"].iloc[i]
        datetime = data["time_key"].iloc[i]

        cond1 = moving_average_5 > moving_average_10
        cond2 = moving_average_10 > moving_average_20
        cond3 = moving_average_10_1past < moving_average_20_1past

        cond4 = close > moving_average_100  # 新增条件：股价大于SMA_100,抵抗风险

        # 买入信号
        if cond1 and cond2 and cond3 and pos == 0 and (cond4 or no_risk):
            bp = close
            pos = 1
            buy_dates.append(datetime)
        # 卖出信号
        elif pos == 1 and not cond2 and not cond3:
            pos = 0
            sp = close
            pc = (sp / bp - 1) * 100  # 计算收益率
            percent_change.append(pc)
            sell_dates.append(datetime)

    # 检查最后持仓是否已卖出
    if pos == 1:
        sp = data["close"].iloc[-1]
        pc = (sp / bp - 1) * 100
        percent_change.append(pc)

    return percent_change, buy_dates, sell_dates


# 绩效评估
def evaluate_performance(percent_change, buy_dates, sell_dates):
    gains = sum([x for x in percent_change if x > 0])
    ng = len([x for x in percent_change if x > 0])
    losses = sum([x for x in percent_change if x < 0])
    nl = len([x for x in percent_change if x < 0])

    total_return = np.prod([(x / 100) + 1 for x in percent_change])

    avg_gain = gains / ng if ng > 0 else 0
    avg_loss = losses / nl if nl > 0 else 0
    max_return = max(percent_change) if percent_change else "undefined"
    max_loss = min(percent_change) if percent_change else "undefined"
    ratio = avg_gain / abs(avg_loss) if avg_loss != 0 else "inf"
    batting_avg = ng / (ng + nl) if (ng + nl) > 0 else 0

    # 打印统计结果
    print("\n交易统计:")
    print(percent_change)
    print(buy_dates)
    print(sell_dates)
    print(f"Batting Avg (命中率): {batting_avg:.2%}")
    print(f"Gain/loss ratio (盈亏比): {ratio}")
    print(f"Average Gain (平均盈利): {avg_gain:.2f}%")
    print(f"Average Loss (平均亏损): {avg_loss:.2f}%")
    print(f"Max Return (最大收益): {max_return}%")
    print(f"Max Loss (最大亏损): {max_loss}%")
    print(f"Total Return (总回报): {(total_return - 1) * 100:.2f}%")

    if len(sell_dates) < len(buy_dates):
        sell_dates.append(None)

    # 转换为DataFrame
    performance_data = pd.DataFrame({
        'Buy Date': pd.to_datetime(buy_dates),
        'Sell Date': pd.to_datetime(sell_dates),
        'Return (%)': percent_change
    })

    # 计算累计收益
    performance_data['Cumulative Return'] = performance_data['Return (%)'].cumsum()
    performance_data['Cumulative Return'] = performance_data['Cumulative Return'].shift(fill_value=0) + \
                                            performance_data['Return (%)']

    # 创建图表：柱状图
    plt.figure(figsize=(12, 6))

    # 绘制柱状图
    colors = ['green' if r >= 0 else 'red' for r in performance_data['Return (%)']]
    plt.bar(performance_data['Buy Date'], performance_data['Return (%)'], color=colors)

    # 添加图表标题和标签
    plt.title('Trading Strategy Performance - Bar Chart', fontsize=16)
    plt.xlabel('Buy Date', fontsize=14)
    plt.ylabel('Return (%)', fontsize=14)
    plt.axhline(0, color='grey', linewidth=0.5, linestyle='--')  # 添加0轴
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig("Trading_Strategy_Performance_plot.png")
    plt.show()


# 对于一些杠杆产品例如tqqq，可以将risk设置为false，普通的股票可以使用默认值
def backtest(stock_code, start_date, end_date, no_risk=True):
    data = get_stock_data(stock_code, start_date, end_date)
    if data is not None:
        data['datetime'] = pd.to_datetime(data['time_key'])
        data.set_index('datetime', inplace=True)

        # 应用策略
        percent_change, buy_date, sell_date = apply_strategy(data, no_risk)

        # 评估绩效
        evaluate_performance(percent_change, buy_date, sell_date)

        # 绘制累计收益图表
        if percent_change:  # 确保有交易记录
            # 计算累计收益率
            cumulative_return = (np.array(percent_change) / 100 + 1).cumprod() - 1

            # 使用 sell_date 作为 X 轴的日期
            # 创建与累计收益长度匹配的 sell_date 索引
            sell_dates = pd.to_datetime(sell_date)  # 确保 sell_date 是 DatetimeIndex

            # 将累计收益率转换为 Pandas Series，并使用 sell_date 作为索引
            cumulative_return = pd.Series(cumulative_return, index=sell_dates)

            # 绘制累计收益率
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative_return.index, cumulative_return, marker='o')

            # 保存并显示图片
            plt.savefig("cumulative_return_plot.png")
            print(f"Length of x-axis (cumulative_return.index): {len(cumulative_return.index)}")
            print(f"Length of y-axis (cumulative_return): {len(cumulative_return)}")

            # 添加标题和标签
            plt.title(f'Cumulative Returns for {stock_code}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.axhline(0, color='red', linestyle='--')

            # 自动格式化日期标签
            plt.gcf().autofmt_xdate()  # 自动格式化 x 轴日期标签以避免重叠
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # 控制显示的时间刻度数量

            plt.show()
        else:
            print("没有交易记录，无法绘制累计收益图表。")


# 回测运行示例
backtest('HK.00981', '2024-01-01', dt.datetime.now().strftime('%Y-%m-%d'), no_risk=True)

# 关闭API连接
quote_ctx.close()
