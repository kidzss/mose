#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号组合模型示例

演示如何使用信号组合模型系统
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional
import argparse
import os
from datetime import datetime

from strategy_optimizer.models import LinearCombinationModel, NeuralCombinationModel
from strategy_optimizer.utils import (
    normalize_signals, evaluate_strategy, plot_strategy_performance, 
    calculate_ic, plot_ic_heatmap, walk_forward_validation
)


def load_demo_data() -> tuple:
    """
    加载示例数据
    
    返回:
        (信号数据, 收益率数据)
    """
    # 生成随机信号数据，通常这里会从CSV或数据库加载真实数据
    np.random.seed(42)
    
    # 假设我们有500个交易日和10个交易信号
    n_days = 500
    n_signals = 10
    
    # 创建随机信号
    signals_data = np.random.randn(n_days, n_signals)
    
    # 创建一些有预测能力的信号
    underlying_factor = np.random.randn(n_days)
    noise = np.random.randn(n_days)
    
    # 目标收益率 = 潜在因子 + 噪声
    returns = 0.01 * underlying_factor + 0.005 * noise
    
    # 让一些信号与收益率相关
    for i in range(5):
        signals_data[:, i] = 0.7 * underlying_factor + 0.3 * np.random.randn(n_days)
    
    # 创建日期索引
    date_index = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # 转换为DataFrame和Series
    signal_names = [f"Signal_{i+1}" for i in range(n_signals)]
    signals_df = pd.DataFrame(signals_data, index=date_index, columns=signal_names)
    returns_series = pd.Series(returns, index=date_index, name="Returns")
    
    return signals_df, returns_series


def train_linear_model(
    signals: pd.DataFrame, 
    returns: pd.Series,
    allow_short: bool = True
) -> LinearCombinationModel:
    """
    训练线性信号组合模型
    
    参数:
        signals: 信号数据
        returns: 收益率数据
        allow_short: 是否允许做空
        
    返回:
        训练好的模型
    """
    print("训练线性信号组合模型...")
    
    # 创建线性模型
    model = LinearCombinationModel(
        model_name="线性组合模型",
        normalize_signals=True,
        normalize_method="zscore",
        allow_short=allow_short,
        weights_constraint="unit_sum",
        optimization_method="sharpe",
        random_state=42
    )
    
    # 训练模型
    model.fit(signals, returns, verbose=True)
    
    # 绘制权重
    fig = model.plot_weights()
    plt.savefig("linear_model_weights.png")
    plt.close(fig)
    
    return model


def train_neural_model(
    signals: pd.DataFrame, 
    returns: pd.Series,
    allow_short: bool = True
) -> NeuralCombinationModel:
    """
    训练神经网络信号组合模型
    
    参数:
        signals: 信号数据
        returns: 收益率数据
        allow_short: 是否允许做空
        
    返回:
        训练好的模型
    """
    print("训练神经网络信号组合模型...")
    
    # 创建神经网络模型
    model = NeuralCombinationModel(
        model_name="神经网络组合模型",
        normalize_signals=True,
        normalize_method="zscore",
        allow_short=allow_short,
        hidden_dims=[32, 16],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        early_stopping=10,
        random_state=42
    )
    
    # 分割训练集和验证集
    train_size = int(len(signals) * 0.8)
    train_signals = signals.iloc[:train_size]
    train_returns = returns.iloc[:train_size]
    val_signals = signals.iloc[train_size:]
    val_returns = returns.iloc[train_size:]
    
    # 训练模型
    model.fit(
        train_signals, 
        train_returns, 
        val_signals=val_signals, 
        val_targets=val_returns,
        verbose=True
    )
    
    # 绘制训练历史
    fig = model.plot_training_history()
    plt.savefig("neural_model_training.png")
    plt.close(fig)
    
    # 绘制近似权重
    fig = model.plot_weights()
    plt.savefig("neural_model_weights.png")
    plt.close(fig)
    
    return model


def run_walk_forward_validation(signals: pd.DataFrame, returns: pd.Series) -> Dict:
    """
    运行滚动窗口验证
    
    参数:
        signals: 信号数据
        returns: 收益率数据
        
    返回:
        验证结果
    """
    print("进行滚动窗口验证...")
    
    # 创建模型
    model = LinearCombinationModel(
        model_name="验证模型",
        normalize_signals=True,
        normalize_method="zscore",
        weights_constraint="unit_sum",
        optimization_method="sharpe"
    )
    
    # 运行验证
    fold_metrics, avg_metrics, final_model = walk_forward_validation(
        model,
        signals,
        returns,
        n_splits=5,
        train_size=None,
        test_size=100,
        gap=0,
        verbose=True
    )
    
    # 绘制最终权重
    fig = final_model.plot_weights()
    plt.savefig("validation_model_weights.png")
    plt.close(fig)
    
    return avg_metrics


def compare_models(
    signals: pd.DataFrame, 
    returns: pd.Series
) -> None:
    """
    比较不同模型的表现
    
    参数:
        signals: 信号数据
        returns: 收益率数据
    """
    print("比较不同模型表现...")
    
    # 训练模型
    linear_model = train_linear_model(signals, returns)
    neural_model = train_neural_model(signals, returns)
    
    # 测试集表现
    test_size = 100
    test_signals = signals.iloc[-test_size:]
    test_returns = returns.iloc[-test_size:]
    
    # 预测信号
    linear_signals = linear_model.predict(test_signals)
    neural_signals = neural_model.predict(test_signals)
    
    # 评估表现
    linear_perf = evaluate_strategy(test_returns, np.sign(linear_signals))
    neural_perf = evaluate_strategy(test_returns, np.sign(neural_signals))
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    # 累积收益
    linear_cumulative = (1 + test_returns * np.sign(linear_signals)).cumprod() - 1
    neural_cumulative = (1 + test_returns * np.sign(neural_signals)).cumprod() - 1
    
    plt.plot(linear_cumulative * 100, label=f"线性模型 (夏普比率: {linear_perf['sharpe_ratio']:.2f})")
    plt.plot(neural_cumulative * 100, label=f"神经网络模型 (夏普比率: {neural_perf['sharpe_ratio']:.2f})")
    
    plt.title("模型表现对比")
    plt.xlabel("交易日")
    plt.ylabel("累积收益 (%)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("model_comparison.png")
    plt.close()
    
    # 绘制详细表现
    fig = plot_strategy_performance(test_returns, np.sign(linear_signals), title="线性模型表现")
    plt.savefig("linear_model_performance.png")
    plt.close(fig)
    
    fig = plot_strategy_performance(test_returns, np.sign(neural_signals), title="神经网络模型表现")
    plt.savefig("neural_model_performance.png")
    plt.close(fig)
    
    # 计算信息系数
    ic_df = calculate_ic(signals, returns, method='spearman', periods=[1, 5, 10, 20])
    fig = plot_ic_heatmap(ic_df, title="信号信息系数 (IC)")
    plt.savefig("signal_ic_heatmap.png")
    plt.close(fig)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="信号组合模型示例")
    parser.add_argument("--allow_short", action="store_true", help="是否允许做空")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 切换到输出目录
    os.chdir(output_dir)
    
    # 加载示例数据
    signals, returns = load_demo_data()
    
    # 训练和比较模型
    compare_models(signals, returns)
    
    # 运行滚动窗口验证
    validation_metrics = run_walk_forward_validation(signals, returns)
    
    print(f"\n所有结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main() 