import pandas as pd
import numpy as np
from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
import matplotlib.pyplot as plt
from data.data_interface import DataInterface
import datetime as dt

# 创建数据接口实例，指定使用MySQL数据源
data_interface = DataInterface(default_source='mysql')

# 获取股票数据
start_date = dt.datetime(2024, 1, 1)  # 使用2024年的数据
end_date = dt.datetime(2025, 4, 1)  # 到2024年底
df = data_interface.get_historical_data('BABA', start_date, end_date, source='mysql')

# 创建策略实例
strategy = NiuniuStrategyV3()

# 运行回测
results = strategy.backtest(df)

# 打印回测结果
print("\n=== 回测结果 ===")
print(f"初始资金: ${results['initial_capital']:,.2f}")
print(f"最终资金: ${results['final_capital']:,.2f}")
print(f"总收益率: {results['total_return']*100:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
print(f"胜率: {results['win_rate']*100:.2f}%")
print(f"平均交易收益: {results['avg_trade_return']*100:.2f}%")
print(f"盈亏比: {results['profit_factor']:.2f}")
print(f"平均交易持续时间: {results['avg_trade_duration']:.1f}天")
print(f"总交易次数: {results['total_trades']}")

# 绘制权益曲线
plt.figure(figsize=(12, 6))
plt.plot(results['equity_curve'].index, results['equity_curve'].values)
plt.title('权益曲线')
plt.xlabel('日期')
plt.ylabel('账户价值')
plt.grid(True)
plt.tight_layout()
plt.savefig('equity_curve.png')
plt.close()

# 绘制回撤曲线
plt.figure(figsize=(12, 6))
plt.plot(results['drawdown_curve'].index, results['drawdown_curve'].values)
plt.title('回撤曲线')
plt.xlabel('日期')
plt.ylabel('回撤比例')
plt.grid(True)
plt.tight_layout()
plt.savefig('drawdown_curve.png')
plt.close()

# 打印交易统计
print("\n=== 交易统计 ===")
print(f"盈利交易数量: {results['winning_trades']}")
print(f"亏损交易数量: {results['losing_trades']}")
print(f"平均盈利: ${results['avg_profit']:.2f}")
print(f"平均亏损: ${results['avg_loss']:.2f}")
print(f"最大单笔盈利: ${results['max_profit']:.2f}")
print(f"最大单笔亏损: ${results['max_loss']:.2f}") 