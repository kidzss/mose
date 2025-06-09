"""
RSI模块测试脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入模块，避免导入整个项目
from rsi import rsi, rsi_divergence, rsi_overbought_oversold, rsi_reversal, stochastic_rsi


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


def test_rsi():
    """测试RSI函数"""
    # 生成测试数据
    df = generate_test_data()
    
    # 计算RSI（使用两种方法）
    df['rsi_wilders'] = rsi(df['close'], window=14, method='wilders')
    df['rsi_ema'] = rsi(df['close'], window=14, method='ema')
    
    # 计算RSI背离
    divergence = rsi_divergence(df['close'], df['rsi_wilders'], window=5)
    df['bullish_div'] = divergence['bullish_divergence']
    df['bearish_div'] = divergence['bearish_divergence']
    
    # 计算RSI超买超卖
    overbought_oversold = rsi_overbought_oversold(df['rsi_wilders'])
    df['overbought'] = overbought_oversold['overbought']
    df['oversold'] = overbought_oversold['oversold']
    df['overbought_exit'] = overbought_oversold['overbought_exit']
    df['oversold_exit'] = overbought_oversold['oversold_exit']
    
    # 计算RSI反转
    reversal = rsi_reversal(df['rsi_wilders'])
    df['rsi_bullish_reversal'] = reversal['bullish_reversal']
    df['rsi_bearish_reversal'] = reversal['bearish_reversal']
    
    # 计算随机RSI
    stoch = stochastic_rsi(df['rsi_wilders'])
    df['stoch_k'] = stoch['k']
    df['stoch_d'] = stoch['d']
    
    # 打印结果摘要
    print("数据集大小:", len(df))
    print("\nRSI数据摘要:")
    print(df[['close', 'rsi_wilders', 'rsi_ema']].tail())
    
    print("\n背离信号统计:")
    print(f"看涨背离次数: {divergence['bullish_divergence'].sum()}")
    print(f"看跌背离次数: {divergence['bearish_divergence'].sum()}")
    
    print("\n超买超卖信号统计:")
    print(f"超买次数: {overbought_oversold['overbought'].sum()}")
    print(f"超卖次数: {overbought_oversold['oversold'].sum()}")
    print(f"超买退出次数: {overbought_oversold['overbought_exit'].sum()}")
    print(f"超卖退出次数: {overbought_oversold['oversold_exit'].sum()}")
    
    print("\nRSI反转信号统计:")
    print(f"看涨反转次数: {reversal['bullish_reversal'].sum()}")
    print(f"看跌反转次数: {reversal['bearish_reversal'].sum()}")
    
    # 绘制图表
    plt.figure(figsize=(15, 12))
    
    # 创建子图
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=ax1)
    
    # 在第一个子图中绘制价格
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    
    # 标记背离点
    bullish_div_points = df[df['bullish_div'] == 1].index
    bearish_div_points = df[df['bearish_div'] == 1].index
    
    ax1.scatter(bullish_div_points, df.loc[bullish_div_points, 'close'], 
               color='green', marker='^', s=100, label='Bullish Divergence')
    ax1.scatter(bearish_div_points, df.loc[bearish_div_points, 'close'], 
               color='red', marker='v', s=100, label='Bearish Divergence')
    
    # 标记RSI反转点
    bullish_rev_points = df[df['rsi_bullish_reversal'] == 1].index
    bearish_rev_points = df[df['rsi_bearish_reversal'] == 1].index
    
    ax1.scatter(bullish_rev_points, df.loc[bullish_rev_points, 'close'], 
               color='green', marker='*', s=150, label='Bullish Reversal')
    ax1.scatter(bearish_rev_points, df.loc[bearish_rev_points, 'close'], 
               color='red', marker='*', s=150, label='Bearish Reversal')
    
    ax1.set_title('Price with RSI Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 在第二个子图中绘制RSI
    ax2.plot(df.index, df['rsi_wilders'], label='RSI (Wilders)', color='blue')
    ax2.plot(df.index, df['rsi_ema'], label='RSI (EMA)', color='orange', alpha=0.7)
    
    # 添加超买超卖区域
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    ax2.fill_between(df.index, 70, 100, color='r', alpha=0.1)
    ax2.fill_between(df.index, 0, 30, color='g', alpha=0.1)
    
    # 标记超买超卖退出点
    overbought_exit_points = df[df['overbought_exit'] == 1].index
    oversold_exit_points = df[df['oversold_exit'] == 1].index
    
    ax2.scatter(overbought_exit_points, df.loc[overbought_exit_points, 'rsi_wilders'], 
               color='green', marker='o', s=50, label='Overbought Exit')
    ax2.scatter(oversold_exit_points, df.loc[oversold_exit_points, 'rsi_wilders'], 
               color='red', marker='o', s=50, label='Oversold Exit')
    
    ax2.set_title('RSI')
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # 在第三个子图中绘制随机RSI
    ax3.plot(df.index, df['stoch_k'], label='Stochastic K', color='blue')
    ax3.plot(df.index, df['stoch_d'], label='Stochastic D', color='red')
    
    # 添加超买超卖区域
    ax3.axhline(y=80, color='r', linestyle='--', alpha=0.3)
    ax3.axhline(y=20, color='g', linestyle='--', alpha=0.3)
    ax3.fill_between(df.index, 80, 100, color='r', alpha=0.1)
    ax3.fill_between(df.index, 0, 20, color='g', alpha=0.1)
    
    ax3.set_title('Stochastic RSI')
    ax3.legend()
    ax3.set_ylim(0, 100)
    ax3.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsi_test.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsi_test.png')}")


if __name__ == "__main__":
    test_rsi()
    print("\nRSI测试完成!") 