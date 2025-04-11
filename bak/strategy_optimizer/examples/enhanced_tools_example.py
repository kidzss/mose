#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强工具示例

展示如何使用增强的时间序列交叉验证、策略评估和头寸规模管理功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入增强工具
from strategy_optimizer.utils import (
    # 数据生成
    DataGenerator,
    
    # 时间序列交叉验证
    TimeSeriesCV,
    BlockingTimeSeriesCV,
    
    # 策略评估
    calculate_drawdowns,
    calculate_statistical_significance,
    
    # 头寸规模管理
    fixed_risk_position_size,
    kelly_position_size,
    calculate_risk_of_ruin
)


def generate_test_data():
    """生成测试数据"""
    print("生成测试数据...")
    # 初始化数据生成器
    generator = DataGenerator(seed=42)
    
    # 生成合成信号和收益率数据
    signals, returns = generator.generate_synthetic_data(
        n_samples=500,
        n_signals=5,
        signal_strength={0: 0.7, 1: 0.5, 2: 0.3},
        noise_level=0.3,
        start_date="2020-01-01"
    )
    
    # 创建持仓序列（简单的符号策略）
    positions = np.sign(signals.iloc[:, 0])
    
    # 创建波动率序列（用于头寸规模计算）
    volatility = returns.rolling(20).std() * np.sqrt(252)
    volatility.fillna(0.2, inplace=True)
    
    return signals, returns, positions, volatility


def demo_time_series_cv(signals, returns):
    """展示时间序列交叉验证功能"""
    print("\n1. 时间序列交叉验证示例")
    print("=" * 50)
    
    # 1.1 基本的时间序列交叉验证
    print("\n1.1 基本的时间序列交叉验证")
    tscv = TimeSeriesCV(n_splits=5, test_size=50, window_type="expanding")
    
    for i, (train_data, test_data) in enumerate(tscv.split(None, signals, returns)):
        train_size = len(train_data['signals'])
        test_size = len(test_data['signals'])
        print(f"折 {i+1}: 训练集大小={train_size}, 测试集大小={test_size}")
    
    # 1.4 分块交叉验证
    print("\n1.2 分块交叉验证")
    block_cv = BlockingTimeSeriesCV(n_splits=4, validation_size=0.2)
    
    for i, (train_data, val_data) in enumerate(block_cv.split(None, signals, returns)):
        print(f"块 {i+1}: 训练集大小={len(train_data['signals'])}, 验证集大小={len(val_data['signals'])}")


def demo_strategy_evaluation(returns, positions):
    """展示策略评估功能"""
    print("\n2. 策略评估示例")
    print("=" * 50)
    
    # 创建一个基准收益率（简单地使用原始收益率的移动平均）
    benchmark_returns = returns.rolling(10).mean().shift(1)
    benchmark_returns.fillna(returns.mean(), inplace=True)
    
    # 2.1 基本策略评估
    print("\n2.1 基本策略评估 - 手动计算主要指标")
    strategy_returns = returns * positions
    
    # 计算一些基本指标
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + strategy_returns.mean()) ** 252 - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    print(f"总收益率: {total_return:.4f}")
    print(f"年化收益率: {annual_return:.4f}")
    print(f"年化波动率: {volatility:.4f}")
    print(f"夏普比率: {sharpe:.4f}")
    
    # 2.2 计算回撤
    print("\n2.2 回撤分析")
    drawdowns = calculate_drawdowns(strategy_returns)
    max_dd = drawdowns.min()
    print(f"最大回撤: {max_dd:.4f}")
    
    # 2.3 统计显著性测试
    print("\n2.3 统计显著性测试")
    significance = calculate_statistical_significance(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns
    )
    
    # 打印t检验结果
    t_test = significance.get('mean_test', {})
    print(f"策略收益显著性: p值={t_test.get('p_value', 1):.4f}, 显著={t_test.get('significant', False)}")
    
    # 与基准比较的测试
    benchmark_test = significance.get('benchmark_comparison', {})
    print(f"相对基准显著性: p值={benchmark_test.get('p_value', 1):.4f}, 显著={benchmark_test.get('significant', False)}")


def demo_position_sizing(returns, positions, volatility):
    """展示头寸规模管理功能"""
    print("\n3. 头寸规模管理示例")
    print("=" * 50)
    
    # 假设的模拟账户资本
    capital = 100000.0
    
    # 3.1 固定风险头寸规模
    print("\n3.1 固定风险头寸规模")
    risk_pct = 0.01  # 风险1%
    stop_loss_pct = 0.05  # 5%止损距离
    
    fixed_risk_pos = fixed_risk_position_size(capital, risk_pct, stop_loss_pct)
    print(f"固定风险头寸规模: ${fixed_risk_pos:.2f}")
    
    # 3.2 凯利公式头寸规模
    print("\n3.2 凯利公式头寸规模")
    # 计算历史胜率和盈亏比
    strategy_returns = returns * positions
    win_rate = (strategy_returns > 0).mean()
    
    avg_win = strategy_returns[strategy_returns > 0].mean()
    avg_loss = np.abs(strategy_returns[strategy_returns < 0].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    kelly_pct = kelly_position_size(win_rate, win_loss_ratio, fraction=0.5)  # 半凯利
    kelly_pos = capital * kelly_pct
    
    print(f"历史胜率: {win_rate:.2f}")
    print(f"盈亏比: {win_loss_ratio:.2f}")
    print(f"凯利比例: {kelly_pct:.4f}")
    print(f"凯利头寸: ${kelly_pos:.2f}")
    
    # 3.3 破产风险计算
    print("\n3.3 破产风险计算")
    ruin_risk = calculate_risk_of_ruin(win_rate, risk_pct, trades=100)
    print(f"破产风险 (100笔交易): {ruin_risk:.4f}")


def main():
    """主函数"""
    # 生成测试数据
    signals, returns, positions, volatility = generate_test_data()
    
    # 演示时间序列交叉验证
    demo_time_series_cv(signals, returns)
    
    # 演示策略评估
    demo_strategy_evaluation(returns, positions)
    
    # 演示头寸规模管理
    demo_position_sizing(returns, positions, volatility)
    
    print("\n演示完成！")


if __name__ == "__main__":
    main() 