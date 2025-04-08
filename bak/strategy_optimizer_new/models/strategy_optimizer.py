#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略优化器

用于优化交易策略的参数和权重
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import argparse
import os
from datetime import datetime

from strategy_optimizer.models import LinearCombinationModel, NeuralCombinationModel
from strategy_optimizer.utils import (
    normalize_signals, evaluate_strategy, plot_strategy_performance, 
    calculate_ic, plot_ic_heatmap, walk_forward_validation
)

@dataclass
class OptimizationConfig:
    """优化配置"""
    input_size: int
    hidden_size: int
    num_heads: int
    num_layers: int
    dropout: float
    output_size: int
    learning_rate: float
    batch_size: int
    epochs: int
    sequence_length: int
    n_heads: int
    n_layers: int
    d_model: int
    weight_decay: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    validation_split: float
    test_split: float
    grad_clip: float

class StrategyDataset(torch.utils.data.Dataset):
    """策略数据集"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class StrategyOptimizer(nn.Module):
    """策略优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
        # 定义模型架构
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_size,
                dropout=config.dropout
            ),
            num_layers=config.n_layers
        )
        
        self.fc = nn.Linear(config.d_model, config.output_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  # (sequence_length, batch_size, input_size)
        x = self.encoder(x)
        x = x.transpose(0, 1)  # (batch_size, sequence_length, d_model)
        x = x[:, -1, :]  # 只使用最后一个时间步的输出
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x)
            y = self.forward(x)
            return y.numpy()
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'StrategyOptimizer':
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

def train_epoch(
    model: StrategyOptimizer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(
    model: StrategyOptimizer,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

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