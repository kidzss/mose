"""
示例策略测试脚本

用于测试多指标结合策略的功能。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 导入策略
from strategy.example_strategy import MultiIndicatorStrategy


def generate_test_data(days=200):
    """生成测试数据"""
    # 生成一个波动的价格序列
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    # 模拟价格: 上涨趋势 + 随机波动 + 震荡区间
    base_price = 100
    trend = np.linspace(0, 30, days) # 上涨趋势
    
    # 添加随机波动
    np.random.seed(42)  # 设置随机种子以便结果可重现
    noise = np.random.normal(0, 3, days)
    
    # 添加周期性波动模拟震荡
    cycles = 15 * np.sin(np.linspace(0, 6*np.pi, days))
    
    # 组合所有分量
    closes = base_price + trend + noise + cycles
    
    # 创建最高价和最低价
    daily_range = np.random.uniform(1, 5, days)
    highs = closes + daily_range / 2
    lows = closes - daily_range / 2
    
    # 模拟成交量
    volume = np.random.normal(1000000, 200000, days)
    volume = np.where(volume < 0, 0, volume)  # 确保成交量不为负
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': closes - daily_range / 4,  # 简单模拟开盘价
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    })
    df.set_index('date', inplace=True)
    
    return df


def test_strategy():
    """测试多指标结合策略"""
    # 生成测试数据
    df = generate_test_data()
    print(f"生成了{len(df)}行测试数据")
    
    # 初始化策略（使用默认参数）
    strategy = MultiIndicatorStrategy()
    print(f"策略名称: {strategy.name}")
    print(f"策略参数: {strategy.parameters}")
    
    # 计算指标和信号
    result_df = strategy.generate_signals(df)
    
    # 输出信号统计
    signal_counts = result_df['signal'].value_counts()
    print("\n信号统计:")
    print(f"买入信号: {signal_counts.get(1, 0)}次")
    print(f"卖出信号: {signal_counts.get(-1, 0)}次")
    print(f"无信号: {signal_counts.get(0, 0)}次")
    
    # 输出最后几个信号
    print("\n最近的信号:")
    recent_signals = result_df[['close', 'signal', 'signal_strength']].tail(10)
    print(recent_signals)
    
    # 提取信号组件
    components = strategy.extract_signal_components(df)
    
    # 可视化结果
    plot_results(result_df, components)


def plot_results(df, components):
    """绘制策略结果"""
    plt.figure(figsize=(15, 12))
    
    # 创建子图
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)  # 价格和信号
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=1, sharex=ax1)  # RSI
    ax3 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax1)  # MACD
    ax4 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax1)  # 信号组件
    
    # 绘制价格和信号
    ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.5)
    ax1.plot(df.index, df['fast_ma'], label='Fast MA', color='blue')
    ax1.plot(df.index, df['slow_ma'], label='Slow MA', color='red')
    
    # 标记买入信号
    buy_signals = df[df['signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # 标记卖出信号
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    ax1.set_title('Price with Trading Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 绘制RSI
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3)
    
    # 标记RSI超买超卖退出点
    rsi_overbought_exit = df[df['rsi_overbought_exit'] == 1]
    rsi_oversold_exit = df[df['rsi_oversold_exit'] == 1]
    
    ax2.scatter(rsi_overbought_exit.index, rsi_overbought_exit['rsi'], 
               color='red', marker='o', s=50, label='Overbought Exit')
    ax2.scatter(rsi_oversold_exit.index, rsi_oversold_exit['rsi'], 
               color='green', marker='o', s=50, label='Oversold Exit')
    
    ax2.set_title('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    # 绘制MACD
    ax3.plot(df.index, df['macd'], label='MACD', color='blue')
    ax3.plot(df.index, df['macd_signal'], label='Signal Line', color='red')
    
    # 绘制MACD柱状图
    for i in range(len(df)):
        if df['macd_hist'].iloc[i] >= 0:
            ax3.bar(df.index[i], df['macd_hist'].iloc[i], color='green', width=1)
        else:
            ax3.bar(df.index[i], df['macd_hist'].iloc[i], color='red', width=1)
    
    # 标记MACD交叉点
    macd_cross_up = df[df['macd_cross_up'] == 1]
    macd_cross_down = df[df['macd_cross_down'] == 1]
    
    ax3.scatter(macd_cross_up.index, macd_cross_up['macd'], 
               color='green', marker='*', s=100, label='Golden Cross')
    ax3.scatter(macd_cross_down.index, macd_cross_down['macd'], 
               color='red', marker='*', s=100, label='Death Cross')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('MACD')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制信号组件
    ax4.plot(components['ma_component'].index, components['ma_component'], label='MA Component', color='blue')
    ax4.plot(components['rsi_component'].index, components['rsi_component'], label='RSI Component', color='green')
    ax4.plot(components['macd_component'].index, components['macd_component'], label='MACD Component', color='red')
    ax4.plot(components['composite'].index, components['composite'], label='Composite Signal', color='black', linewidth=2)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
    ax4.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
    ax4.set_title('Signal Components')
    ax4.set_ylim(-1.2, 1.2)
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(current_dir, 'multi_indicator_strategy_test.png'))
    print(f"\n图表已保存到 {os.path.join(current_dir, 'multi_indicator_strategy_test.png')}")


if __name__ == "__main__":
    try:
        test_strategy()
        print("\n策略测试完成!")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc() 