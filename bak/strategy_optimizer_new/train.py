#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略优化模型训练脚本

该脚本是策略优化器的主要训练入口点，用于训练各种市场状态感知的交易信号组合模型。
它支持多种模型架构，包括Transformer、LSTM和前馈神经网络，用于将多个交易策略的信号
组合成一个优化的组合信号。

主要功能:
1. 从数据库加载历史市场数据和交易信号
2. 数据预处理和特征工程
3. 构建和训练信号组合模型
4. 模型评估和性能指标计算
5. 保存训练好的模型和评估结果

使用方法:
```
python -m strategy_optimizer.train \
    --config configs/optimizer_config.json \
    --output_dir outputs/models \
    --symbols "AAPL,MSFT,GOOG" \
    --start_date "2018-01-01" \
    --end_date "2023-01-01" \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --model_type transformer
```

配置参数:
- config: 配置文件路径
- output_dir: 输出目录，用于保存模型和结果
- symbols: 股票代码列表
- start_date/end_date: 训练数据的起止时间
- epochs: 训练轮数
- batch_size: 批量大小
- learning_rate: 学习率
- model_type: 模型类型 (transformer, lstm, mlp)
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from strategy_optimizer.models.strategy_optimizer import (
    StrategyOptimizer,
    OptimizationConfig,
    StrategyDataset
)
from strategy_optimizer.data_processors.data_processor import DataProcessor
from sqlalchemy import create_engine
import sys
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from strategy_optimizer.models.transformer import StrategyTransformer, train_epoch, validate
from strategy_optimizer.models.weighted_mse_loss import WeightedMSELoss
from strategy_optimizer.models.early_stopping import EarlyStopping
from strategy_optimizer.models.simple_mse_loss import SimpleMSELoss

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_stock_list() -> list:
    """从数据库获取监控的股票列表"""
    try:
        # 创建数据库连接
        db_config = default_config.database
        engine = create_engine(
            f"mysql+pymysql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        )

        # 查询监控的股票列表
        query = """
        SELECT symbol as stock_code, name, sector, industry
        FROM monitored_stocks
        WHERE is_active = TRUE
        ORDER BY symbol
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning("未从数据库获取到股票列表，使用默认列表")
            return ["GOOG", "NVDA", "AMD", "TSLA", "AAPL", "ASML", "MSFT", "AMZN", "META", "GOOGL"]  # 默认列表作为后备

        # 获取股票代码列表
        stock_list = df['stock_code'].tolist()

        logger.info(f"从数据库获取到 {len(stock_list)} 只股票")
        logger.info(f"股票列表: {', '.join(stock_list)}")

        return stock_list

    except Exception as e:
        logger.error(f"获取股票列表时出错: {str(e)}")
        return ["GOOG", "NVDA", "AMD", "TSLA", "AAPL", "ASML", "MSFT", "AMZN", "META", "GOOGL"]  # 出错时使用默认列表


# 策略列表
STRATEGY_LIST = [
    "GoldTriangleStrategy",
    "MomentumStrategy",
    "NiuniuStrategy",
    "TDIStrategy",
    "MarketForecastStrategy",
    "CPGWStrategy",
    "VolumeStrategy"
]


def load_config(config_path: str) -> dict:
    """加载配置"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件时出错: {e}")
        raise


def prepare_data(
        data_processor: DataProcessor,
        symbols: list,
        config: dict
) -> tuple:
    """准备训练数据"""
    try:
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['data_config']['lookback_period'])

        all_features = []
        all_targets = []
        sequence_length = config['model_config']['sequence_length']

        # 移除只处理两只股票的限制
        for symbol in symbols:
            logger.info(f"处理股票 {symbol} 的数据")

            # 获取股票数据
            df = data_processor.get_stock_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if df.empty:
                logger.warning(f"股票 {symbol} 数据为空，跳过")
                continue

            # 数据质量检查
            if not data_processor.validate_data(df):
                logger.warning(f"股票 {symbol} 数据质量检查未通过，跳过")
                continue

            # 准备特征
            X = data_processor.prepare_features(df, sequence_length)

            if len(X) == 0:
                logger.warning(f"股票 {symbol} 特征准备失败，跳过")
                continue

            # 准备目标变量
            y = data_processor.prepare_targets(df, sequence_length)

            if len(y) == 0:
                logger.warning(f"股票 {symbol} 目标变量准备失败，跳过")
                continue

            # 检查特征和目标变量的维度是否匹配
            if len(y) != len(X):
                logger.warning(f"股票 {symbol} 特征和目标变量维度不匹配: X={len(X)}, y={len(y)}，跳过")
                continue

            all_features.append(X)
            all_targets.append(y)
            logger.info(f"股票 {symbol} 数据处理成功，特征形状: {X.shape}, 目标变量形状: {y.shape}")

        if not all_features or not all_targets:
            raise ValueError("没有有效的训练数据")

        # 合并数据
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0)

        # 数据检查和统计
        logger.info(f"合并后数据形状 - X: {X.shape}, y: {y.shape}")
        logger.info(f"特征统计: mean={np.mean(X):.4f}, std={np.std(X):.4f}, min={np.min(X):.4f}, max={np.max(X):.4f}")
        logger.info(f"目标统计: mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")

        return X, y

    except Exception as e:
        logger.error(f"准备训练数据时出错: {e}")
        raise


def train_model(model, train_loader, val_loader, config, device):
    """
    训练模型的简化版本
    """
    # 使用简单的MSE损失
    criterion = SimpleMSELoss()

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training_config']['learning_rate']
    )

    # 早停
    early_stopping = EarlyStopping(
        patience=config['training_config'].get('early_stopping_patience', 10),
        min_delta=1e-4
    )

    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(config['training_config']['epochs']):
        # 训练
        train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.6f}')

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.6f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = os.path.join(
                config['output_dir'],
                f'model_v{timestamp}'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            logger.info(f'保存最佳模型到: {best_model_path}')

        # 早停检查
        if early_stopping(val_loss):
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    return best_model_path


def evaluate_model(
        optimizer: StrategyOptimizer,
        test_data: tuple,
        config: dict
) -> dict:
    """评估模型"""
    try:
        X_test, y_test = test_data

        # 验证数据
        if X_test.size == 0 or y_test.size == 0:
            logger.error("测试数据为空")
            return {
                'overall_mse': float('nan'),
                'overall_mae': float('nan'),
                'strategy_errors': {}
            }

        # 验证维度
        if y_test.shape[1] != len(config['strategy_config']['strategies']):
            logger.error(f"目标变量维度 ({y_test.shape[1]}) 与策略数量 ({len(config['strategy_config']['strategies'])}) 不匹配")
            return {
                'overall_mse': float('nan'),
                'overall_mae': float('nan'),
                'strategy_errors': {}
            }

        # 预测测试集
        y_pred = optimizer.predict(X_test)
        if y_pred is None or y_pred.size == 0:
            logger.error("模型预测结果为空")
            return {
                'overall_mse': float('nan'),
                'overall_mae': float('nan'),
                'strategy_errors': {}
            }

        # 验证预测结果维度
        if y_pred.shape != y_test.shape:
            logger.error(f"预测结果维度 ({y_pred.shape}) 与目标变量维度 ({y_test.shape}) 不匹配")
            return {
                'overall_mse': float('nan'),
                'overall_mae': float('nan'),
                'strategy_errors': {}
            }

        # 移除无效值
        valid_mask = ~(np.isnan(y_test) | np.isnan(y_pred))
        if not np.any(valid_mask):
            logger.error("所有预测结果都是无效值")
            return {
                'overall_mse': float('nan'),
                'overall_mae': float('nan'),
                'strategy_errors': {}
            }

        # 计算评估指标（仅使用有效值）
        y_test_valid = y_test[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        mse = np.mean((y_test_valid - y_pred_valid) ** 2)
        mae = np.mean(np.abs(y_test_valid - y_pred_valid))

        # 计算每个策略的权重误差
        strategy_errors = {}
        for i, strategy in enumerate(config['strategy_config']['strategies'].keys()):
            strategy_valid_mask = ~(np.isnan(y_test[:, i]) | np.isnan(y_pred[:, i]))
            if np.any(strategy_valid_mask):
                strategy_errors[strategy] = {
                    'mse': float(np.mean((y_test[strategy_valid_mask, i] - y_pred[strategy_valid_mask, i]) ** 2)),
                    'mae': float(np.mean(np.abs(y_test[strategy_valid_mask, i] - y_pred[strategy_valid_mask, i])))
                }
            else:
                strategy_errors[strategy] = {
                    'mse': float('nan'),
                    'mae': float('nan')
                }

        logger.info(f"评估结果 - MSE: {mse:.6f}, MAE: {mae:.6f}")
        return {
            'overall_mse': float(mse),
            'overall_mae': float(mae),
            'strategy_errors': strategy_errors
        }

    except Exception as e:
        logger.error(f"评估模型时出错: {e}")
        return {
            'overall_mse': float('nan'),
            'overall_mae': float('nan'),
            'strategy_errors': {}
        }


def save_results(
        results: dict,
        optimizer: StrategyOptimizer,
        config: dict,
        output_dir: str
) -> None:
    """保存结果"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存评估结果
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 保存模型
        model_path = os.path.join(output_dir, 'model.pth')
        optimizer.save_model(model_path)

        # 保存配置
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info(f"结果已保存到 {output_dir}")

    except Exception as e:
        logger.error(f"保存结果时出错: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练策略优化模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--symbols', type=str, required=True, help='股票代码列表，用逗号分隔')
    parser.add_argument('--start_date', type=str, required=True, help='开始日期')
    parser.add_argument('--end_date', type=str, required=True, help='结束日期')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--model_type', type=str, default='transformer', help='模型类型')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    config['model_config']['epochs'] = args.epochs
    config['model_config']['batch_size'] = args.batch_size
    config['model_config']['learning_rate'] = args.learning_rate
    config['data_config']['start_date'] = args.start_date
    config['data_config']['end_date'] = args.end_date
    config['data_config']['symbols'] = args.symbols.split(',')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 准备数据
    data_processor = DataProcessor()
    X, y = prepare_data(data_processor, config['data_config']['symbols'], config)

    # 创建数据集
    dataset = StrategyDataset(X, y)
    train_size = int(len(dataset) * (1 - config['model_config']['validation_split']))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model_config']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model_config']['batch_size']
    )

    # 创建模型
    model = StrategyOptimizer(OptimizationConfig(**config['model_config']))
    model = model.to(device)

    # 训练模型
    train_model(model, train_loader, val_loader, config, device)

    # 保存模型
    model_path = os.path.join(args.output_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    model.save(model_path)
    logger.info(f'模型已保存到: {model_path}')


if __name__ == '__main__':
    main()
