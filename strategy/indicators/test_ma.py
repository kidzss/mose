"""
移动平均线模块测试脚本
"""

import sys
import os
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

from strategy.indicators.moving_averages import sma, ema, wma, smma, tema, kama, hull_ma


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
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['close'], label='Price', color='black', alpha=0.3)
    plt.plot(df.index, df['sma'], label='SMA')
    plt.plot(df.index, df['ema'], label='EMA')
    plt.plot(df.index, df['wma'], label='WMA')
    plt.plot(df.index, df['smma'], label='SMMA')
    plt.plot(df.index, df['tema'], label='TEMA')
    plt.plot(df.index, df['kama'], label='KAMA')
    plt.plot(df.index, df['hull'], label='Hull MA')
    
    plt.title('Moving Averages Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(current_dir, 'moving_averages_comparison.png'))
    
    print(f"图表已保存到 {os.path.join(current_dir, 'moving_averages_comparison.png')}")


if __name__ == "__main__":
    test_moving_averages()
    print("移动平均线测试完成!") 