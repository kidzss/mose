#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单示例脚本

展示策略优化器的基本功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入数据生成器
from strategy_optimizer.utils import DataGenerator

def main():
    """展示数据生成器功能的简单示例"""
    print("简单示例启动...")
    
    # 初始化数据生成器
    generator = DataGenerator(seed=42)
    
    # 生成合成信号和收益率数据
    signals, returns = generator.generate_synthetic_data(
        n_samples=200,
        n_signals=3,
        signal_strength={0: 0.6, 1: 0.4, 2: 0.2},
        noise_level=0.3,
        start_date="2021-01-01"
    )
    
    # 打印数据形状
    print(f"生成的数据:")
    print(f"- 信号数据: 形状 {signals.shape}")
    print(f"- 收益率数据: 形状 {returns.shape}")
    
    # 打印信号的前几行
    print("\n信号数据前5行:")
    print(signals.head())
    
    # 打印收益率的前几行
    print("\n收益率数据前5行:")
    print(returns.head())
    
    # 简单策略: 使用第一个信号的符号
    positions = np.sign(signals.iloc[:, 0])
    
    # 计算策略收益率
    strategy_returns = positions * returns
    
    # 计算策略的累积收益
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    
    # 打印策略指标
    print("\n策略表现:")
    print(f"- 总收益率: {cumulative_returns.iloc[-1]:.4f}")
    print(f"- 平均日收益率: {strategy_returns.mean():.6f}")
    print(f"- 年化收益率: {(1 + strategy_returns.mean()) ** 252 - 1:.4f}")
    print(f"- 年化波动率: {strategy_returns.std() * np.sqrt(252):.4f}")
    print(f"- 夏普比率: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.4f}")
    
    # 绘制累积收益曲线
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns.values)
    plt.title('策略累积收益')
    plt.xlabel('日期')
    plt.ylabel('累积收益')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('strategy_returns.png')
    print("\n累积收益曲线已保存为 'strategy_returns.png'")
    
    print("\n简单示例完成!")

if __name__ == "__main__":
    main() 