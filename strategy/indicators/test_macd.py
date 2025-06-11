"""
MACD模块测试脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入模块，避免导入整个项目
from macd import macd, macd_crossover, macd_zero_crossover, macd_divergence, macd_histogram_reversal, ppo


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
    prices = base_price + trend + noise + cycles
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    df.set_index('date', inplace=True)
    
    return df


def test_macd():
    """测试MACD函数"""
    # 生成测试数据
    df = generate_test_data()
    
    # 计算MACD
    macd_data = macd(df['close'])
    df['macd'] = macd_data['macd']
    df['signal'] = macd_data['signal']
    df['histogram'] = macd_data['histogram']
    
    # 计算PPO
    ppo_data = ppo(df['close'])
    df['ppo'] = ppo_data['ppo']
    df['ppo_signal'] = ppo_data['signal']
    df['ppo_histogram'] = ppo_data['histogram']
    
    # 计算MACD交叉信号
    crossover_signals = macd_crossover(df['macd'], df['signal'])
    df['golden_cross'] = crossover_signals['golden_cross']
    df['death_cross'] = crossover_signals['death_cross']
    
    # 计算零轴交叉信号
    zero_crossover_signals = macd_zero_crossover(df['macd'])
    df['zero_cross_up'] = zero_crossover_signals['zero_cross_up']
    df['zero_cross_down'] = zero_crossover_signals['zero_cross_down']
    
    # 计算MACD背离
    divergence_signals = macd_divergence(df['close'], df['macd'])
    df['macd_bullish_div'] = divergence_signals['bullish_divergence']
    df['macd_bearish_div'] = divergence_signals['bearish_divergence']
    
    # 计算柱状图反转信号
    histogram_reversal = macd_histogram_reversal(df['histogram'])
    df['bullish_exhaustion'] = histogram_reversal['bullish_exhaustion']
    df['bearish_exhaustion'] = histogram_reversal['bearish_exhaustion']
    
    # 打印结果摘要
    print("数据集大小:", len(df))
    print("\nMACD数据摘要:")
    print(df[['close', 'macd', 'signal', 'histogram']].tail())
    
    print("\nPPO数据摘要:")
    print(df[['close', 'ppo', 'ppo_signal', 'ppo_histogram']].tail())
    
    print("\n交叉信号统计:")
    print(f"金叉次数: {df['golden_cross'].sum()}")
    print(f"死叉次数: {df['death_cross'].sum()}")
    print(f"零轴上穿次数: {df['zero_cross_up'].sum()}")
    print(f"零轴下穿次数: {df['zero_cross_down'].sum()}")
    
    print("\n背离信号统计:")
    print(f"看涨背离次数: {df['macd_bullish_div'].sum()}")
    print(f"看跌背离次数: {df['macd_bearish_div'].sum()}")
    
    print("\n柱状图反转信号统计:")
    print(f"上升动能耗尽次数: {df['bullish_exhaustion'].sum()}")
    print(f"下降动能耗尽次数: {df['bearish_exhaustion'].sum()}")
    
    # 绘制图表
    plt.figure(figsize=(15, 12))
    
    # 创建子图
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=ax1)
    
    # 在第一个子图中绘制价格
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    
    # 标记背离点
    bullish_div_points = df[df['macd_bullish_div'] == 1].index
    bearish_div_points = df[df['macd_bearish_div'] == 1].index
    
    ax1.scatter(bullish_div_points, df.loc[bullish_div_points, 'close'], 
               color='green', marker='^', s=100, label='Bullish Divergence')
    ax1.scatter(bearish_div_points, df.loc[bearish_div_points, 'close'], 
               color='red', marker='v', s=100, label='Bearish Divergence')
    
    # 标记交叉信号
    golden_cross_points = df[df['golden_cross'] == 1].index
    death_cross_points = df[df['death_cross'] == 1].index
    
    ax1.scatter(golden_cross_points, df.loc[golden_cross_points, 'close'], 
               color='green', marker='*', s=150, label='Golden Cross')
    ax1.scatter(death_cross_points, df.loc[death_cross_points, 'close'], 
               color='red', marker='*', s=150, label='Death Cross')
    
    ax1.set_title('Price with MACD Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 在第二个子图中绘制MACD
    ax2.plot(df.index, df['macd'], label='MACD', color='blue')
    ax2.plot(df.index, df['signal'], label='Signal Line', color='red')
    
    # 绘制柱状图
    for i in range(len(df)):
        if df['histogram'].iloc[i] >= 0:
            ax2.bar(df.index[i], df['histogram'].iloc[i], color='green', width=1)
        else:
            ax2.bar(df.index[i], df['histogram'].iloc[i], color='red', width=1)
    
    # 标记零轴交叉
    zero_cross_up_points = df[df['zero_cross_up'] == 1].index
    zero_cross_down_points = df[df['zero_cross_down'] == 1].index
    
    ax2.scatter(zero_cross_up_points, df.loc[zero_cross_up_points, 'macd'], 
               color='green', marker='o', s=50, label='Zero Cross Up')
    ax2.scatter(zero_cross_down_points, df.loc[zero_cross_down_points, 'macd'], 
               color='red', marker='o', s=50, label='Zero Cross Down')
    
    # 标记柱状图反转
    bullish_exhaustion_points = df[df['bullish_exhaustion'] == 1].index
    bearish_exhaustion_points = df[df['bearish_exhaustion'] == 1].index
    
    ax2.scatter(bullish_exhaustion_points, df.loc[bullish_exhaustion_points, 'histogram'], 
               color='red', marker='s', s=50, label='Bullish Exhaustion')
    ax2.scatter(bearish_exhaustion_points, df.loc[bearish_exhaustion_points, 'histogram'], 
               color='green', marker='s', s=50, label='Bearish Exhaustion')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # 在第三个子图中绘制PPO
    ax3.plot(df.index, df['ppo'], label='PPO', color='blue')
    ax3.plot(df.index, df['ppo_signal'], label='PPO Signal', color='red')
    
    # 绘制PPO柱状图
    for i in range(len(df)):
        if df['ppo_histogram'].iloc[i] >= 0:
            ax3.bar(df.index[i], df['ppo_histogram'].iloc[i], color='green', width=1)
        else:
            ax3.bar(df.index[i], df['ppo_histogram'].iloc[i], color='red', width=1)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('PPO (Percentage Price Oscillator)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'macd_test.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'macd_test.png')}")


if __name__ == "__main__":
    test_macd()
    print("\nMACD测试完成!") 