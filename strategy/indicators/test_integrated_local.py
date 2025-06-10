"""
技术指标综合模块测试脚本（仅使用本地导入）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 本地导入指标模块
from moving_averages import sma, ema, wma, smma, tema, kama, hull_ma
from bollinger_bands import bollinger_bands, bollinger_band_squeeze, bollinger_breakout, bollinger_reversal
from rsi import rsi, rsi_overbought_oversold, rsi_reversal, stochastic_rsi
from macd import macd, macd_crossover, macd_zero_crossover, macd_divergence, macd_histogram_reversal
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


def test_indicators():
    """测试各种技术指标"""
    # 生成测试数据
    df = generate_test_data()
    print(f"生成了{len(df)}行测试数据")
    
    # 计算各种指标
    result_df = df.copy()
    
    # 计算移动平均线
    result_df['sma_20'] = sma(df['close'], 20)
    result_df['ema_20'] = ema(df['close'], 20)
    result_df['wma_20'] = wma(df['close'], 20)
    
    # 计算布林带
    bb = bollinger_bands(df['close'])
    result_df['bb_middle'] = bb['middle']
    result_df['bb_upper'] = bb['upper']
    result_df['bb_lower'] = bb['lower']
    
    # 计算RSI
    result_df['rsi'] = rsi(df['close'])
    
    # 计算MACD
    macd_result = macd(df['close'])
    result_df['macd'] = macd_result['macd']
    result_df['macd_signal'] = macd_result['signal']
    result_df['macd_hist'] = macd_result['histogram']
    
    # 计算ADX
    adx_result = adx(df['high'], df['low'], df['close'])
    result_df['adx'] = adx_result['adx']
    result_df['plus_di'] = adx_result['plus_di']
    result_df['minus_di'] = adx_result['minus_di']
    
    # 打印指标摘要
    print("\n==== 指标计算结果摘要 ====")
    indicators_only = result_df.iloc[:, 5:]  # 排除OHLCV列
    print(indicators_only.tail(3))
    
    # 计算交叉信号
    cross_signals = macd_crossover(macd_result['macd'], macd_result['signal'])
    print("\n==== MACD交叉信号 ====")
    print(f"金叉次数: {cross_signals['golden_cross'].sum()}")
    print(f"死叉次数: {cross_signals['death_cross'].sum()}")
    
    # 计算RSI超买超卖信号
    rsi_signals = rsi_overbought_oversold(result_df['rsi'])
    print("\n==== RSI超买超卖信号 ====")
    print(f"超买次数: {rsi_signals['overbought'].sum()}")
    print(f"超卖次数: {rsi_signals['oversold'].sum()}")
    
    # 绘制指标图表
    plot_results(result_df)


def plot_results(df):
    """绘制指标图表"""
    plt.figure(figsize=(15, 15))
    
    # 创建子图
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)  # 价格和移动平均线
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=1, sharex=ax1)  # MACD
    ax3 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax1)  # RSI
    ax4 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax1)  # ADX
    
    # 绘制价格和移动平均线
    ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.5)
    ax1.plot(df.index, df['sma_20'], label='SMA(20)', color='blue')
    ax1.plot(df.index, df['ema_20'], label='EMA(20)', color='red')
    ax1.plot(df.index, df['wma_20'], label='WMA(20)', color='orange')
    ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='purple', linestyle=':')
    ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='green', linestyle='--')
    ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='green', linestyle='--')
    ax1.set_title('Price with Moving Averages and Bollinger Bands')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 绘制MACD
    ax2.plot(df.index, df['macd'], label='MACD', color='blue')
    ax2.plot(df.index, df['macd_signal'], label='Signal', color='red')
    
    # 绘制MACD柱状图
    for i in range(len(df)):
        if df['macd_hist'].iloc[i] >= 0:
            ax2.bar(df.index[i], df['macd_hist'].iloc[i], color='green', width=1)
        else:
            ax2.bar(df.index[i], df['macd_hist'].iloc[i], color='red', width=1)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # 绘制RSI
    ax3.plot(df.index, df['rsi'], label='RSI(14)', color='blue')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
    ax3.set_title('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True)
    
    # 绘制ADX
    ax4.plot(df.index, df['adx'], label='ADX', color='purple')
    ax4.plot(df.index, df['plus_di'], label='+DI', color='green')
    ax4.plot(df.index, df['minus_di'], label='-DI', color='red')
    ax4.axhline(y=25, color='black', linestyle='--', alpha=0.3)
    ax4.set_title('ADX')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integrated_indicators_local_test.png'))
    print(f"\n图表已保存到 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integrated_indicators_local_test.png')}")


if __name__ == "__main__":
    test_indicators()
    print("\n技术指标测试完成!") 