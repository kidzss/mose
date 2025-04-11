import pandas as pd

# 读取三个回测文件
df_momentum = pd.read_csv("../backtest/momentum_nasdaq100_backtest_results.csv")
df_market_forecast = pd.read_csv("../backtest/forecast_nasdaq100_backtest_results.csv")
df_gold_triangle = pd.read_csv("../backtest/triangle_nasdaq100_backtest_results.csv")

# 添加一列标识策略
df_momentum['Strategy'] = 'Momentum'
df_market_forecast['Strategy'] = 'Market Forecast'
df_gold_triangle['Strategy'] = 'Gold Triangle'

# 合并数据
df_all = pd.concat([df_momentum, df_market_forecast, df_gold_triangle], ignore_index=True)

# 基础统计汇总
# 汇总每个股票的各项指标（按股票代码和策略）
summary_stats = df_all.groupby(['Stock Code', 'Strategy']).agg({
    'Total Return(%)': 'mean',
    'Gain/loss ratio': 'mean',
    'Batting Avg': 'mean'
}).reset_index()

# 按 Total Return 排序，找到收益最高的股票
top_returns = summary_stats.sort_values(by='Total Return(%)', ascending=False).head(20)
print("Top Stocks by Total Return:\n", top_returns)

# 筛选潜力股票和策略

# 筛选盈利比率较高的股票
top_gain_loss_ratio = summary_stats.sort_values(by='Gain/loss ratio', ascending=False).head(15)

# 筛选盈亏次数较高的股票
top_batting_avg = summary_stats.sort_values(by='Batting Avg', ascending=False).head(15)

print("Top Stocks by Gain/Loss Ratio:\n", top_gain_loss_ratio)
print("Top Stocks by Batting Avg:\n", top_batting_avg)


