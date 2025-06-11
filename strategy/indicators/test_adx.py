"""
ADX模块测试脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入模块，避免导入整个项目
from adx import adx, adx_trend_strength, adx_trend_direction, adx_crossover, adx_reversal, dmi_oscillator


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
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'high': highs,
        'low': lows,
        'close': closes
    })
    df.set_index('date', inplace=True)
    
    return df


def test_adx():
    """测试ADX函数"""
    # 生成测试数据
    df = generate_test_data()
    
    # 计算ADX及相关指标
    adx_data = adx(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_data['adx']
    df['plus_di'] = adx_data['plus_di']
    df['minus_di'] = adx_data['minus_di']
    
    # 计算趋势强度
    df['trend_strength'] = adx_trend_strength(df['adx'])
    
    # 计算趋势方向
    df['trend_direction'] = adx_trend_direction(df['plus_di'], df['minus_di'])
    
    # 计算DI交叉
    crossover_signals = adx_crossover(df['plus_di'], df['minus_di'])
    df['bullish_cross'] = crossover_signals['bullish_cross']
    df['bearish_cross'] = crossover_signals['bearish_cross']
    
    # 计算ADX反转
    adx_reversal_signals = adx_reversal(df['adx'])
    df['adx_peak'] = adx_reversal_signals['adx_peak']
    df['adx_bottom'] = adx_reversal_signals['adx_bottom']
    
    # 计算DMI振荡器
    df['dmi_osc'] = dmi_oscillator(df['plus_di'], df['minus_di'])
    
    # 打印结果摘要
    print("数据集大小:", len(df))
    print("\nADX数据摘要:")
    print(df[['close', 'adx', 'plus_di', 'minus_di']].tail())
    
    print("\n趋势强度统计:")
    strength_counts = df['trend_strength'].value_counts().sort_index()
    strength_labels = {
        0: "无趋势/非常弱 (ADX < 20)",
        1: "弱趋势 (20 <= ADX < 25)",
        2: "强趋势 (25 <= ADX < 40)",
        3: "非常强趋势 (40 <= ADX < 50)",
        4: "极端强趋势 (ADX >= 50)"
    }
    for strength, count in strength_counts.items():
        print(f"{strength_labels[strength]}: {count}天 ({count/len(df)*100:.1f}%)")
    
    print("\n趋势方向统计:")
    direction_counts = df['trend_direction'].value_counts().sort_index()
    direction_labels = {
        -1: "下降趋势",
        0: "无明确趋势",
        1: "上升趋势"
    }
    for direction, count in direction_counts.items():
        print(f"{direction_labels[direction]}: {count}天 ({count/len(df)*100:.1f}%)")
    
    print("\n交叉信号统计:")
    print(f"看涨交叉次数: {df['bullish_cross'].sum()}")
    print(f"看跌交叉次数: {df['bearish_cross'].sum()}")
    
    print("\nADX反转信号统计:")
    print(f"ADX顶部反转次数: {df['adx_peak'].sum()}")
    print(f"ADX底部反转次数: {df['adx_bottom'].sum()}")
    
    # 绘制图表
    plt.figure(figsize=(15, 12))
    
    # 创建子图
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=ax1)
    
    # 在第一个子图中绘制价格和趋势方向
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    
    # 标记交叉信号
    bullish_cross_points = df[df['bullish_cross'] == 1].index
    bearish_cross_points = df[df['bearish_cross'] == 1].index
    
    ax1.scatter(bullish_cross_points, df.loc[bullish_cross_points, 'close'], 
               color='green', marker='^', s=100, label='Bullish Cross')
    ax1.scatter(bearish_cross_points, df.loc[bearish_cross_points, 'close'], 
               color='red', marker='v', s=100, label='Bearish Cross')
    
    # 使用颜色区分趋势方向
    for i in range(len(df)):
        if df['trend_direction'].iloc[i] == 1 and df['trend_strength'].iloc[i] >= 2:
            ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], alpha=0.2, color='green')
        elif df['trend_direction'].iloc[i] == -1 and df['trend_strength'].iloc[i] >= 2:
            ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], alpha=0.2, color='red')
    
    ax1.set_title('Price with Trend Direction')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 在第二个子图中绘制ADX
    ax2.plot(df.index, df['adx'], label='ADX', color='purple')
    
    # 添加ADX阈值线
    ax2.axhline(y=20, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=25, color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=40, color='b', linestyle='--', alpha=0.3)
    ax2.axhline(y=50, color='y', linestyle='--', alpha=0.3)
    
    # 标记ADX反转信号
    adx_peak_points = df[df['adx_peak'] == 1].index
    adx_bottom_points = df[df['adx_bottom'] == 1].index
    
    ax2.scatter(adx_peak_points, df.loc[adx_peak_points, 'adx'], 
               color='red', marker='o', s=50, label='ADX Peak')
    ax2.scatter(adx_bottom_points, df.loc[adx_bottom_points, 'adx'], 
               color='green', marker='o', s=50, label='ADX Bottom')
    
    ax2.set_title('ADX (Average Directional Index)')
    ax2.legend()
    ax2.grid(True)
    
    # 在第三个子图中绘制+DI和-DI
    ax3.plot(df.index, df['plus_di'], label='+DI', color='green')
    ax3.plot(df.index, df['minus_di'], label='-DI', color='red')
    ax3.plot(df.index, df['dmi_osc'], label='DMI Oscillator', color='blue', alpha=0.5)
    
    # 添加零轴线
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax3.set_title('Directional Indicators (+DI, -DI, DMI Oscillator)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adx_test.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adx_test.png')}")


if __name__ == "__main__":
    test_adx()
    print("\nADX测试完成!") 