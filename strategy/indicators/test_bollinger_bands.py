"""
布林带模块测试脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入模块，避免导入整个项目
from bollinger_bands import bollinger_bands, bollinger_band_squeeze, bollinger_breakout, bollinger_reversal


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


def test_bollinger_bands():
    """测试布林带函数"""
    # 生成测试数据
    df = generate_test_data()
    window = 20
    num_std = 2.0
    
    # 计算布林带
    bb = bollinger_bands(df['close'], window, num_std)
    
    # 将结果添加到DataFrame
    df['bb_middle'] = bb['middle']
    df['bb_upper'] = bb['upper']
    df['bb_lower'] = bb['lower']
    df['bb_bandwidth'] = bb['bandwidth']
    df['bb_percent'] = bb['b_percent']
    
    # 计算挤压指标
    df['bb_squeeze'] = bollinger_band_squeeze(df['close'], window, num_std)
    
    # 计算突破信号
    breakout = bollinger_breakout(df['close'], window, num_std)
    df['upper_breakout'] = breakout['upper_breakout']
    df['lower_breakout'] = breakout['lower_breakout']
    
    # 计算反转信号
    reversal = bollinger_reversal(df['close'], window, num_std)
    df['bullish_reversal'] = reversal['bullish_reversal']
    df['bearish_reversal'] = reversal['bearish_reversal']
    
    # 打印结果摘要
    print("数据集大小:", len(df))
    print("\n布林带数据摘要:")
    print(df[['close', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent']].tail())
    
    print("\n突破信号统计:")
    print(f"上轨突破次数: {breakout['upper_breakout'].sum()}")
    print(f"下轨突破次数: {breakout['lower_breakout'].sum()}")
    
    print("\n反转信号统计:")
    print(f"看涨反转次数: {reversal['bullish_reversal'].sum()}")
    print(f"看跌反转次数: {reversal['bearish_reversal'].sum()}")
    
    # 绘制图表
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=ax1)
    
    # 在第一个子图中绘制价格和布林带
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    ax1.plot(df.index, df['bb_middle'], label='Middle Band', color='gray', linestyle='--')
    ax1.plot(df.index, df['bb_upper'], label='Upper Band', color='red')
    ax1.plot(df.index, df['bb_lower'], label='Lower Band', color='green')
    
    # 标记突破点
    upper_breakout_points = df[df['upper_breakout'] == 1].index
    lower_breakout_points = df[df['lower_breakout'] == 1].index
    ax1.scatter(upper_breakout_points, df.loc[upper_breakout_points, 'close'], 
               color='red', marker='^', s=50, label='Upper Breakout')
    ax1.scatter(lower_breakout_points, df.loc[lower_breakout_points, 'close'], 
               color='green', marker='v', s=50, label='Lower Breakout')
    
    # 标记反转点
    bullish_reversal_points = df[df['bullish_reversal'] == 1].index
    bearish_reversal_points = df[df['bearish_reversal'] == 1].index
    ax1.scatter(bullish_reversal_points, df.loc[bullish_reversal_points, 'close'], 
               color='green', marker='*', s=100, label='Bullish Reversal')
    ax1.scatter(bearish_reversal_points, df.loc[bearish_reversal_points, 'close'], 
               color='red', marker='*', s=100, label='Bearish Reversal')
    
    ax1.set_title('Bollinger Bands with Breakout and Reversal Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 在第二个子图中绘制带宽和挤压指标
    ax2.plot(df.index, df['bb_bandwidth'], label='Bandwidth', color='purple')
    ax2.plot(df.index, df['bb_squeeze'], label='Squeeze', color='orange')
    ax2.set_title('Bollinger Bandwidth and Squeeze')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bollinger_bands_test.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bollinger_bands_test.png')}")


if __name__ == "__main__":
    test_bollinger_bands()
    print("\n布林带测试完成!") 