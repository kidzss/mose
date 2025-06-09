"""
技术指标综合模块测试脚本
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 导入技术指标综合模块
from strategy.indicators.indicators import TechnicalIndicators, calculate_indicators


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


def test_integrated_indicators():
    """测试综合技术指标模块"""
    # 生成测试数据
    df = generate_test_data()
    print(f"生成了{len(df)}行测试数据")
    
    # 1. 测试单个指标计算
    print("\n==== 测试单个指标计算 ====")
    
    # 测试移动平均线
    sma_series = TechnicalIndicators.calculate_ma(df['close'], 'sma', 20)
    print("SMA计算结果:", sma_series.tail(3))
    
    # 测试布林带
    bb_data = TechnicalIndicators.calculate_bb(df['close'])
    print("布林带中轨计算结果:", bb_data['middle'].tail(3))
    
    # 测试RSI
    rsi_data = TechnicalIndicators.calculate_rsi(df['close'])
    print("RSI计算结果:", rsi_data.tail(3))
    
    # 测试MACD
    macd_data = TechnicalIndicators.calculate_macd(df['close'])
    print("MACD计算结果:", macd_data['macd'].tail(3))
    
    # 测试ADX
    adx_data = TechnicalIndicators.calculate_adx(df['high'], df['low'], df['close'])
    print("ADX计算结果:", adx_data['adx'].tail(3))
    
    # 2. 测试批量计算所有指标
    print("\n==== 测试批量计算所有指标 ====")
    result_df = calculate_indicators(df)
    print("计算了", len(result_df.columns) - 5, "个指标")  # 减去5个原始OHLCV列
    print("结果DataFrame的列:", result_df.columns.tolist())
    
    # 3. 测试选择性计算部分指标
    print("\n==== 测试选择性计算部分指标 ====")
    selected_df = calculate_indicators(df, ['sma', 'rsi', 'macd'])
    print("选择性计算的指标列:", [col for col in selected_df.columns if col not in df.columns])
    
    # 4. 测试交叉信号计算
    print("\n==== 测试交叉信号计算 ====")
    ema_20 = TechnicalIndicators.calculate_ma(df['close'], 'ema', 20)
    ema_50 = TechnicalIndicators.calculate_ma(df['close'], 'ema', 50)
    crossover_signals = TechnicalIndicators.get_crossover_signals(ema_20, ema_50)
    print("金叉次数:", crossover_signals['cross_up'].sum())
    print("死叉次数:", crossover_signals['cross_down'].sum())
    
    # 5. 可视化计算结果
    plot_integrated_results(df, result_df)


def plot_integrated_results(original_df, result_df):
    """可视化综合技术指标的计算结果"""
    plt.figure(figsize=(15, 15))
    
    # 创建子图
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)  # 价格和移动平均线
    ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=1, sharex=ax1)  # MACD
    ax3 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax1)  # RSI
    ax4 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax1)  # ADX
    
    # 绘制价格和移动平均线
    ax1.plot(result_df.index, result_df['close'], label='Price', color='black', alpha=0.5)
    ax1.plot(result_df.index, result_df['sma'], label='SMA(20)', color='blue')
    ax1.plot(result_df.index, result_df['ema'], label='EMA(20)', color='red')
    ax1.plot(result_df.index, result_df['bb_middle'], label='BB Middle', color='purple', linestyle=':')
    ax1.plot(result_df.index, result_df['bb_upper'], label='BB Upper', color='green', linestyle='--')
    ax1.plot(result_df.index, result_df['bb_lower'], label='BB Lower', color='green', linestyle='--')
    ax1.set_title('Price with Moving Averages and Bollinger Bands')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 绘制MACD
    ax2.plot(result_df.index, result_df['macd'], label='MACD', color='blue')
    ax2.plot(result_df.index, result_df['macd_signal'], label='Signal', color='red')
    
    # 绘制MACD柱状图
    for i in range(len(result_df)):
        if result_df['macd_hist'].iloc[i] >= 0:
            ax2.bar(result_df.index[i], result_df['macd_hist'].iloc[i], color='green', width=1)
        else:
            ax2.bar(result_df.index[i], result_df['macd_hist'].iloc[i], color='red', width=1)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # 绘制RSI
    ax3.plot(result_df.index, result_df['rsi'], label='RSI(14)', color='blue')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.3)
    ax3.set_title('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True)
    
    # 绘制ADX
    ax4.plot(result_df.index, result_df['adx'], label='ADX', color='purple')
    ax4.plot(result_df.index, result_df['plus_di'], label='+DI', color='green')
    ax4.plot(result_df.index, result_df['minus_di'], label='-DI', color='red')
    ax4.axhline(y=25, color='black', linestyle='--', alpha=0.3)
    ax4.set_title('ADX')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(current_dir, 'integrated_indicators_test.png'))
    print(f"\n图表已保存到 {os.path.join(current_dir, 'integrated_indicators_test.png')}")


if __name__ == "__main__":
    test_integrated_indicators()
    print("\n综合技术指标测试完成!") 