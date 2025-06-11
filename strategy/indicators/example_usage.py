"""
技术指标模块使用示例

展示如何使用我们实现的各种技术指标。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入指标模块
from moving_averages import sma, ema, wma, smma, tema, kama, hull_ma
from bollinger_bands import bollinger_bands
from rsi import rsi, rsi_overbought_oversold
from macd import macd, macd_crossover
from adx import adx, adx_trend_strength, adx_trend_direction


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


def demonstrate_indicators():
    """演示如何使用技术指标"""
    # 生成测试数据
    df = generate_test_data()
    print(f"生成了{len(df)}行测试数据")
    
    # 计算各种指标
    result_df = df.copy()
    
    # 1. 移动平均线
    result_df['sma_20'] = sma(df['close'], 20)
    result_df['ema_20'] = ema(df['close'], 20)
    result_df['wma_20'] = wma(df['close'], 20)
    result_df['tema_20'] = tema(df['close'], 20)
    result_df['kama_20'] = kama(df['close'], 20)
    result_df['hull_20'] = hull_ma(df['close'], 20)
    
    # 2. 布林带
    bb = bollinger_bands(df['close'])
    result_df['bb_middle'] = bb['middle']
    result_df['bb_upper'] = bb['upper']
    result_df['bb_lower'] = bb['lower']
    result_df['bb_bandwidth'] = bb['bandwidth']
    result_df['bb_percent'] = bb['b_percent']
    
    # 3. RSI
    result_df['rsi'] = rsi(df['close'])
    
    # 4. MACD
    macd_data = macd(df['close'])
    result_df['macd'] = macd_data['macd']
    result_df['macd_signal'] = macd_data['signal']
    result_df['macd_hist'] = macd_data['histogram']
    
    # 5. ADX
    adx_data = adx(df['high'], df['low'], df['close'])
    result_df['adx'] = adx_data['adx']
    result_df['plus_di'] = adx_data['plus_di']
    result_df['minus_di'] = adx_data['minus_di']
    
    # 生成简单的交易信号
    # 移动平均线交叉
    ma_cross_up = ((result_df['ema_20'] > result_df['sma_20']) & 
                   (result_df['ema_20'].shift(1) <= result_df['sma_20'].shift(1)))
    ma_cross_down = ((result_df['ema_20'] < result_df['sma_20']) & 
                     (result_df['ema_20'].shift(1) >= result_df['sma_20'].shift(1)))
    
    # RSI超买超卖
    rsi_signals = rsi_overbought_oversold(result_df['rsi'])
    rsi_oversold_exit = rsi_signals['oversold_exit']
    rsi_overbought_exit = rsi_signals['overbought_exit']
    
    # MACD交叉
    macd_signals = macd_crossover(result_df['macd'], result_df['macd_signal'])
    macd_cross_up = macd_signals['golden_cross']
    macd_cross_down = macd_signals['death_cross']
    
    # 简单的综合信号 - 使用numpy.where代替布尔索引
    # 买入信号: RSI超卖区域退出，或MACD金叉
    buy_signal = np.zeros(len(result_df))
    for i in range(len(result_df)):
        if (i < len(rsi_oversold_exit) and rsi_oversold_exit.iloc[i]) or \
           (i < len(macd_cross_up) and macd_cross_up.iloc[i]):
            buy_signal[i] = 1
    
    # 卖出信号: RSI超买区域退出，或MACD死叉
    sell_signal = np.zeros(len(result_df))
    for i in range(len(result_df)):
        if (i < len(rsi_overbought_exit) and rsi_overbought_exit.iloc[i]) or \
           (i < len(macd_cross_down) and macd_cross_down.iloc[i]):
            sell_signal[i] = 1
    
    result_df['buy_signal'] = buy_signal
    result_df['sell_signal'] = sell_signal
    
    # 显示结果摘要
    print("\n指标计算结果摘要:")
    print(result_df[['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'adx']].tail())
    
    print("\n信号统计:")
    print(f"买入信号数量: {result_df['buy_signal'].sum()}")
    print(f"卖出信号数量: {result_df['sell_signal'].sum()}")
    
    # 可视化结果
    plot_indicators(result_df)


def plot_indicators(df):
    """绘制技术指标和交易信号"""
    plt.figure(figsize=(15, 15))
    
    # 创建子图
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)  # 价格和移动平均线
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=1, sharex=ax1)  # RSI
    ax3 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax1)  # MACD
    ax4 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax1)  # ADX
    
    # 绘制价格和移动平均线
    ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.5)
    ax1.plot(df.index, df['sma_20'], label='SMA(20)', color='blue')
    ax1.plot(df.index, df['ema_20'], label='EMA(20)', color='red')
    ax1.plot(df.index, df['wma_20'], label='WMA(20)', color='green')
    ax1.plot(df.index, df['hull_20'], label='Hull MA(20)', color='purple')
    
    # 绘制布林带
    ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='orange', linestyle='--')
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='orange', linestyle=':')
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='orange', linestyle=':')
    
    # 标记买入信号
    buy_signals = df[df['buy_signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # 标记卖出信号
    sell_signals = df[df['sell_signal'] == 1]
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    ax1.set_title('Price with Moving Averages, Bollinger Bands and Trading Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 绘制RSI
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3)
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
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('MACD')
    ax3.legend()
    ax3.grid(True)
    
    # 绘制ADX
    ax4.plot(df.index, df['adx'], label='ADX', color='black')
    ax4.plot(df.index, df['plus_di'], label='+DI', color='green')
    ax4.plot(df.index, df['minus_di'], label='-DI', color='red')
    ax4.axhline(y=25, color='blue', linestyle='--', alpha=0.3)
    ax4.set_title('ADX')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indicators_usage_example.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indicators_usage_example.png')}")


if __name__ == "__main__":
    demonstrate_indicators()
    print("\n技术指标使用示例完成!") 