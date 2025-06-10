"""
移动平均线模块简单测试脚本
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接导入模块，避免导入整个项目
from moving_averages import sma, ema, wma, smma, tema, kama, hull_ma


def generate_test_data():
    """生成测试数据"""
    # 生成一个简单的正弦波数据集
    dates = [datetime.now() + timedelta(days=i) for i in range(100)]
    values = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    
    # 添加一些噪声
    noise = np.random.normal(0, 1, 100)
    values = values + noise
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': values
    })
    df.set_index('date', inplace=True)
    
    return df


def test_moving_averages():
    """测试移动平均线函数"""
    # 生成测试数据
    df = generate_test_data()
    window = 10
    
    # 计算各种移动平均线
    df['sma'] = sma(df['close'], window)
    df['ema'] = ema(df['close'], window)
    df['wma'] = wma(df['close'], window)
    df['smma'] = smma(df['close'], window)
    df['tema'] = tema(df['close'], window)
    df['kama'] = kama(df['close'])
    df['hull'] = hull_ma(df['close'], window)
    
    # 打印结果
    print(df.tail())
    
    # 打印每种移动平均线的最新值
    print("\n各种移动平均线最新值:")
    print(f"原始价格: {df['close'].iloc[-1]:.2f}")
    print(f"SMA: {df['sma'].iloc[-1]:.2f}")
    print(f"EMA: {df['ema'].iloc[-1]:.2f}")
    print(f"WMA: {df['wma'].iloc[-1]:.2f}")
    print(f"SMMA: {df['smma'].iloc[-1]:.2f}")
    print(f"TEMA: {df['tema'].iloc[-1]:.2f}")
    print(f"KAMA: {df['kama'].iloc[-1]:.2f}")
    print(f"Hull MA: {df['hull'].iloc[-1]:.2f}")
    
    # 每种移动平均线滞后性比较
    df_corr = df.copy()
    lag_shifts = [1, 2, 3, 5]
    
    print("\n各种移动平均线的滞后性比较 (与原始价格的相关系数):")
    for shift in lag_shifts:
        print(f"\n滞后{shift}个时间单位:")
        for col in ['sma', 'ema', 'wma', 'smma', 'tema', 'kama', 'hull']:
            corr = df['close'].corr(df_corr[col].shift(shift))
            print(f"{col.upper():5s}: {corr:.4f}")


if __name__ == "__main__":
    test_moving_averages()
    print("\n移动平均线测试完成!") 