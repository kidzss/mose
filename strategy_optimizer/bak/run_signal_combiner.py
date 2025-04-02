#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import torch

from strategy.strategy_factory import StrategyFactory
from strategy_optimizer.data_processors.data_processor import DataProcessor
from strategy_optimizer.models.signal_combiner import SignalCombinerModel, CombinerConfig, MarketRegime
from strategy_optimizer.market_state import MarketState, create_market_features

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('strategy_optimizer/outputs/signal_combiner.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多策略信号组合模型')
    
    # 数据参数
    parser.add_argument('--symbol', type=str, default='QQQ', help='股票代码')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='开始日期')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='结束日期')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比例')
    parser.add_argument('--sequence_length', type=int, default=60, help='序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=15, help='早停轮数')
    
    # 信号组合参数
    parser.add_argument('--use_market_state', action='store_true', help='是否使用市场状态')
    parser.add_argument('--time_varying_weights', action='store_true', help='是否使用时变权重')
    parser.add_argument('--regularization', type=float, default=0.01, help='正则化强度')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='strategy_optimizer/outputs', help='输出目录')
    parser.add_argument('--save_model', action='store_true', help='是否保存模型')
    
    return parser.parse_args()

def prepare_strategy_signals(
    symbol: str, 
    start_date: str, 
    end_date: str
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """准备策略信号数据
    
    参数:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        stock_data: 股票数据
        strategy_signals: 策略信号字典
    """
    logger.info(f"准备策略信号数据: {symbol} 从 {start_date} 到 {end_date}")
    
    # 初始化数据处理器
    data_processor = DataProcessor()
    
    # 获取股票数据
    stock_data = data_processor.get_stock_data(symbol, start_date, end_date)
    if stock_data.empty:
        raise ValueError(f"未能获取到 {symbol} 的数据")
    
    # 初始化策略工厂
    strategy_factory = StrategyFactory()
    
    # 获取所有可用策略
    strategies = {
        'Momentum': strategy_factory.create_strategy('MomentumStrategy'),
        'GoldTriangle': strategy_factory.create_strategy('GoldTriangleStrategy'),
        'TDI': strategy_factory.create_strategy('TDIStrategy'),
        'MarketForecast': strategy_factory.create_strategy('MarketForecastStrategy'),
        'CPGW': strategy_factory.create_strategy('CPGWStrategy'),
        'Volume': strategy_factory.create_strategy('VolumeStrategy')
    }
    
    # 对每个策略计算信号
    strategy_signals = {}
    for name, strategy in strategies.items():
        signals = strategy.calculate_signals(stock_data)
        
        # 归一化信号到 [-1, 1]
        normalized_signal = signals['signal'].clip(-1, 1)
        strategy_signals[name] = normalized_signal
        
        logger.info(f"策略 {name} 信号统计: 平均值={normalized_signal.mean():.4f}, "
                   f"标准差={normalized_signal.std():.4f}, "
                   f"最小值={normalized_signal.min():.4f}, "
                   f"最大值={normalized_signal.max():.4f}")
    
    # 将所有信号合并到一个DataFrame
    signal_df = pd.DataFrame(strategy_signals)
    
    # 计算单独策略的收益表现
    returns = stock_data['close'].pct_change()
    for name in strategies.keys():
        # 简单的策略收益计算 (前一天信号 * 当天收益)
        signal_returns = signal_df[name].shift(1) * returns
        signal_df[f'{name}_Return'] = signal_returns
        
        # 计算策略累积收益
        cumulative_return = (1 + signal_returns).cumprod() - 1
        logger.info(f"策略 {name} 累积收益: {cumulative_return.iloc[-1]:.4f}")
    
    return stock_data, strategy_signals

def prepare_market_features(
    stock_data: pd.DataFrame,
    market_state: Optional[MarketState] = None
) -> np.ndarray:
    """准备市场特征数据
    
    参数:
        stock_data: 股票数据
        market_state: 市场状态对象
        
    返回:
        市场特征数组 [n_samples, seq_len, feature_dim]
    """
    # 使用股票数据计算基本特征
    logger.info("准备市场特征数据")
    
    # 计算技术指标
    features = pd.DataFrame(index=stock_data.index)
    
    # 价格特征
    for window in [5, 10, 20, 50, 200]:
        features[f'ma_{window}'] = stock_data['close'].rolling(window=window).mean()
        features[f'ma_ratio_{window}'] = stock_data['close'] / features[f'ma_{window}']
    
    # 波动率特征
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = stock_data['close'].pct_change().rolling(window=window).std() * np.sqrt(252)
    
    # 动量特征
    for window in [5, 10, 20, 50, 200]:
        features[f'momentum_{window}'] = stock_data['close'].pct_change(periods=window)
    
    # 成交量特征
    for window in [5, 10, 20]:
        features[f'volume_ma_{window}'] = stock_data['volume'].rolling(window=window).mean()
        features[f'volume_ratio_{window}'] = stock_data['volume'] / features[f'volume_ma_{window}']
    
    # RSI
    delta = stock_data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = stock_data['close'].ewm(span=12, adjust=False).mean()
    exp26 = stock_data['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = macd - signal
    
    # ATR
    high_low = stock_data['high'] - stock_data['low']
    high_close = (stock_data['high'] - stock_data['close'].shift()).abs()
    low_close = (stock_data['low'] - stock_data['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(window=14).mean()
    
    # 将NaN值填充为0
    features = features.fillna(0)
    
    # 准备序列数据
    sequence_length = 60  # 使用60天的历史数据
    n_samples = len(features) - sequence_length
    n_features = features.shape[1]
    
    # 创建特征序列
    feature_sequences = np.zeros((n_samples, sequence_length, n_features))
    
    for i in range(n_samples):
        feature_sequences[i] = features.iloc[i:i+sequence_length].values
    
    logger.info(f"市场特征数据形状: {feature_sequences.shape}")
    
    return feature_sequences

def run_signal_combiner(args):
    """运行信号组合模型"""
    logger.info("开始运行信号组合模型")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 准备数据
    stock_data, strategy_signals = prepare_strategy_signals(
        args.symbol, args.start_date, args.end_date
    )
    
    # 创建特征
    feature_sequences = prepare_market_features(stock_data)
    
    # 准备策略信号矩阵
    signal_df = pd.DataFrame(strategy_signals)
    signals = signal_df.values
    
    # 准备目标变量 - 这里使用未来一天的收益率作为目标
    returns = stock_data['close'].pct_change().shift(-1)
    
    # 调整数据长度
    sequence_length = args.sequence_length
    signals = signals[sequence_length:]
    returns = returns[sequence_length:-1]  # 去掉最后一个NaN
    
    # 确保长度一致
    min_length = min(len(feature_sequences), len(signals), len(returns))
    feature_sequences = feature_sequences[:min_length]
    signals = signals[:min_length]
    returns = returns[:min_length]
    
    logger.info(f"数据准备完成: 特征={feature_sequences.shape}, 信号={signals.shape}, 收益={returns.shape}")
    
    # 创建模型配置
    config = CombinerConfig(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        use_market_state=args.use_market_state,
        time_varying_weights=args.time_varying_weights,
        regularization_strength=args.regularization,
        market_feature_dim=feature_sequences.shape[2]
    )
    
    # 创建模型
    model = SignalCombinerModel(
        n_strategies=signals.shape[1],
        input_dim=feature_sequences.shape[2],
        config=config
    )
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = model.prepare_data(
        feature_sequences, signals, returns
    )
    
    # 训练模型
    history = model.train(train_loader, val_loader)
    
    # 评估模型
    test_loss, _ = model.evaluate(test_loader)
    logger.info(f"测试集损失: {test_loss:.6f}")
    
    # 使用模型进行预测
    predicted_signals, predicted_weights, market_states = model.predict(
        feature_sequences, signals
    )
    
    # 计算组合信号的表现
    predicted_signals = predicted_signals.flatten()
    predicted_returns = predicted_signals[:-1] * returns[1:]  # 前一天信号 * 当天收益
    cumulative_return = (1 + predicted_returns).cumprod() - 1
    
    logger.info(f"组合策略累积收益: {cumulative_return[-1]:.4f}")
    
    # 绘制训练过程
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制策略权重分布
    strategy_names = list(strategy_signals.keys())
    weight_fig = model.plot_weights(predicted_weights, strategy_names)
    
    # 绘制累积收益曲线
    plt.figure(figsize=(12, 6))
    # 绘制组合策略收益
    plt.plot(cumulative_return, label='组合策略')
    
    # 绘制各单独策略的收益
    for name in strategy_signals.keys():
        strategy_return = signal_df[f'{name}_Return'].iloc[sequence_length:min_length]
        strategy_cum_return = (1 + strategy_return).cumprod() - 1
        plt.plot(strategy_cum_return, label=name, alpha=0.7)
    
    plt.title('累积收益曲线')
    plt.xlabel('时间')
    plt.ylabel('累积收益')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f"{args.output_dir}/signal_combiner_performance.png")
    weight_fig.savefig(f"{args.output_dir}/signal_combiner_weights.png")
    
    # 如果有市场状态，绘制市场状态分布
    if args.use_market_state and market_states is not None:
        market_fig = model.plot_market_states(market_states)
        market_fig.savefig(f"{args.output_dir}/signal_combiner_market_states.png")
    
    # 保存模型
    if args.save_model:
        model.save(f"{args.output_dir}/signal_combiner_model.pt")
    
    logger.info("信号组合模型运行完成")
    
    return model, history

if __name__ == "__main__":
    args = parse_args()
    run_signal_combiner(args) 