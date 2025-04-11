#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号组合器演示脚本

此脚本展示如何使用 SignalCombinerModel 模型来组合多个交易信号。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import logging
import sys
from typing import List, Tuple, Optional

# 配置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_optimizer.models.signal_combiner import SignalCombinerModel, CombinerConfig, MarketRegime
from strategy_optimizer.utils.evaluation import calc_sharpe, calc_max_drawdown

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_mock_data(
    n_samples: int = 1000,
    n_strategies: int = 5,
    feature_dim: int = 10,
    seq_len: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成模拟数据
    
    参数:
        n_samples: 样本数量
        n_strategies: 策略数量
        feature_dim: 特征维度
        seq_len: 序列长度
        
    返回:
        features, strategy_signals, returns
    """
    logger.info("生成模拟数据...")
    
    # 创建特征: 形状为 [n_samples, seq_len, feature_dim]
    features = np.random.randn(n_samples, seq_len, feature_dim)
    
    # 创建策略信号: 形状为 [n_samples, n_strategies]
    # 我们创建几个具有不同特性的策略信号
    strategy_signals = np.zeros((n_samples, n_strategies))
    
    # 策略1: 随机信号，但有一些趋势
    trend = np.cumsum(np.random.randn(n_samples) * 0.1)
    strategy_signals[:, 0] = np.random.randn(n_samples) * 0.5 + np.sin(np.linspace(0, 10, n_samples)) + trend * 0.1
    
    # 策略2: 周期性信号
    strategy_signals[:, 1] = np.sin(np.linspace(0, 20, n_samples))
    
    # 策略3: 跳跃信号
    strategy_signals[:, 2] = np.random.randn(n_samples) * 0.2
    for i in range(5):
        jump_point = np.random.randint(0, n_samples)
        jump_direction = np.random.choice([-1, 1])
        strategy_signals[jump_point:, 2] += jump_direction * 0.5
    
    # 策略4: 具有趋势的信号
    strategy_signals[:, 3] = np.cumsum(np.random.randn(n_samples) * 0.05)
    
    # 策略5: 噪声信号
    if n_strategies > 4:
        strategy_signals[:, 4] = np.random.randn(n_samples) * 0.1
    
    # 创建目标收益率: 形状为 [n_samples]
    # 目标是几个策略的加权组合加上一些噪声
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])[:n_strategies]
    returns = np.sum(strategy_signals * weights.reshape(1, -1), axis=1) + np.random.randn(n_samples) * 0.1
    
    logger.info(f"生成了 {n_samples} 个样本数据")
    
    return features, strategy_signals, returns

def main():
    """主函数"""
    # 生成模拟数据
    n_samples = 1000
    n_strategies = 5
    feature_dim = 15
    seq_len = 60
    
    features, strategy_signals, returns = generate_mock_data(
        n_samples=n_samples,
        n_strategies=n_strategies,
        feature_dim=feature_dim,
        seq_len=seq_len
    )
    
    # 创建模型配置
    config = CombinerConfig(
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        sequence_length=seq_len,
        batch_size=32,
        epochs=50,
        learning_rate=0.001,
        market_feature_dim=feature_dim,
        use_market_state=True,
        time_varying_weights=True,
        early_stopping_patience=10
    )
    
    # 创建模型
    model = SignalCombinerModel(
        n_strategies=n_strategies,
        input_dim=feature_dim,
        config=config
    )
    
    # 训练模型
    logger.info("训练模型...")
    train_losses, val_losses = model.fit(features, strategy_signals, returns)
    
    # 绘制训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程')
    plt.legend()
    plt.savefig('strategy_optimizer/outputs/train_process.png')
    plt.close()
    
    # 预测
    logger.info("进行预测...")
    combined_signals, weights, market_states = model.predict(features, strategy_signals)
    
    # 评估结果
    logger.info("评估结果...")
    strategy_names = [f"Strategy {i+1}" for i in range(n_strategies)]
    
    # 计算每个策略的收益率序列
    strategy_returns = {}
    for i, name in enumerate(strategy_names):
        strategy_returns[name] = strategy_signals[:, i] * returns / np.abs(returns)
    
    # 计算组合策略的收益率序列
    combined_returns = combined_signals.flatten() * returns / np.abs(returns)
    strategy_returns['组合策略'] = combined_returns
    
    # 计算夏普比率
    sharpe_ratios = {}
    for name, ret in strategy_returns.items():
        sharpe_ratios[name] = calc_sharpe(ret)
    
    # 计算最大回撤
    max_drawdowns = {}
    for name, ret in strategy_returns.items():
        max_drawdowns[name] = calc_max_drawdown(np.cumsum(ret))
    
    # 打印结果
    logger.info("========== 评估结果 ==========")
    logger.info("夏普比率:")
    for name, sharpe in sharpe_ratios.items():
        logger.info(f"{name}: {sharpe:.4f}")
    
    logger.info("\n最大回撤:")
    for name, dd in max_drawdowns.items():
        logger.info(f"{name}: {dd:.4f}")
    
    # 绘制组合权重
    model.plot_weights(weights, strategy_names)
    plt.savefig('strategy_optimizer/outputs/weights_distribution.png')
    plt.close()
    
    # 绘制累积收益曲线
    plt.figure(figsize=(12, 8))
    cumulative_returns = {}
    for name, ret in strategy_returns.items():
        cumulative_returns[name] = np.cumsum(ret)
        plt.plot(cumulative_returns[name], label=name)
    
    plt.xlabel('时间')
    plt.ylabel('累积收益')
    plt.title('各策略累积收益曲线')
    plt.legend()
    plt.savefig('strategy_optimizer/outputs/cumulative_returns.png')
    plt.close()
    
    # 如果存在市场状态，绘制市场状态转换
    if market_states is not None:
        logger.info("绘制市场状态...")
        market_state_labels = [
            f"状态{i+1}" for i in range(market_states.shape[1])
        ]
        
        regime_detector = MarketRegime(market_states)
        regimes = regime_detector.detect_regimes()
        
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        for i, label in enumerate(market_state_labels):
            plt.plot(market_states[:, i], label=label)
        plt.xlabel('时间')
        plt.ylabel('状态概率')
        plt.title('市场状态概率')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(regimes, 'r-', linewidth=2)
        plt.xlabel('时间')
        plt.ylabel('市场状态')
        plt.title('市场状态分类结果')
        plt.yticks(range(len(market_state_labels)), market_state_labels)
        
        plt.tight_layout()
        plt.savefig('strategy_optimizer/outputs/market_states.png')
        plt.close()
    
    logger.info("演示完成! 结果保存在 strategy_optimizer/outputs/ 目录下")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("strategy_optimizer/outputs", exist_ok=True)
    main() 