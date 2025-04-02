#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据生成器

创建用于训练信号组合模型的数据集
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from strategy_optimizer.data_processors.feature_engineer import FeatureEngineer
from strategy.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)

@dataclass
class DataGeneratorConfig:
    """训练数据生成器配置"""
    # 数据范围
    start_date: str = "2018-01-01"
    end_date: str = "2023-01-01"
    symbols: List[str] = None  # 默认为None时将使用默认股票列表
    
    # 特征配置
    window_size: int = 60  # 使用60天的历史数据作为特征
    feature_list: List[str] = None  # 默认为None时将使用所有可用特征
    
    # 策略配置
    strategies: List[str] = None  # 默认为None时将使用所有可用策略
    
    # 数据处理配置
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False  # 时间序列数据通常不打乱
    
    # 标签生成配置
    target_type: str = "returns"  # 'returns' 或 'direction'
    target_horizon: int = 5  # 预测5天后的收益率
    
    # 缓存配置
    use_cache: bool = True
    cache_dir: str = "strategy_optimizer/cache"

class TrainingDataGenerator:
    """训练数据生成器
    
    生成用于训练信号组合模型的数据集
    """
    
    DEFAULT_SYMBOLS = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "TSLA", "NVDA", "JPM", "V", "JNJ"
    ]
    
    DEFAULT_STRATEGIES = [
        "GoldTriangle", "Momentum", "TDI", 
        "MarketForecast", "CPGW", "Niuniu"
    ]
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        """初始化训练数据生成器
        
        参数:
            config: 数据生成器配置，默认为None时使用默认配置
        """
        self.config = config if config is not None else DataGeneratorConfig()
        
        # 设置默认值
        if self.config.symbols is None:
            self.config.symbols = self.DEFAULT_SYMBOLS
            
        if self.config.strategies is None:
            self.config.strategies = self.DEFAULT_STRATEGIES
            
        # 创建必要的目录
        if self.config.use_cache and not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
        # 初始化特征工程器
        self.feature_engineer = FeatureEngineer()
        
        # 初始化策略工厂
        self.strategy_factory = StrategyFactory()
        
        # 初始化数据缓存
        self.data_cache = {}
        
        # 初始化标准化器
        self.feature_scaler = StandardScaler()
        self.signal_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def generate_dataset(
        self,
        force_refresh: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """生成训练数据集
        
        参数:
            force_refresh: 是否强制刷新缓存
            
        返回:
            train_data, val_data, test_data
            每个数据包含:
                - 'features': 特征序列 [n_samples, seq_len, feature_dim]
                - 'signals': 策略信号 [n_samples, n_strategies]
                - 'targets': 目标值 [n_samples,]
        """
        cache_file = os.path.join(
            self.config.cache_dir, 
            f"dataset_{self.config.start_date}_{self.config.end_date}.joblib"
        )
        
        # 如果使用缓存且缓存文件存在，则直接加载
        if self.config.use_cache and os.path.exists(cache_file) and not force_refresh:
            logger.info(f"从缓存加载数据集: {cache_file}")
            return joblib.load(cache_file)
        
        logger.info("生成新的训练数据集...")
        
        # 处理每个股票的数据
        all_features = []
        all_signals = []
        all_targets = []
        all_dates = []
        
        for symbol in self.config.symbols:
            logger.info(f"处理股票数据: {symbol}")
            features, signals, targets, dates = self._process_symbol(symbol)
            
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_signals.append(signals)
                all_targets.append(targets)
                all_dates.extend(dates)
        
        # 合并所有股票的数据
        if len(all_features) > 0:
            combined_features = np.concatenate(all_features, axis=0)
            combined_signals = np.concatenate(all_signals, axis=0)
            combined_targets = np.concatenate(all_targets, axis=0)
            
            # 打乱数据（如果需要）
            if self.config.shuffle:
                indices = np.arange(combined_features.shape[0])
                np.random.shuffle(indices)
                combined_features = combined_features[indices]
                combined_signals = combined_signals[indices]
                combined_targets = combined_targets[indices]
            
            # 划分数据集
            n_samples = combined_features.shape[0]
            n_val = int(n_samples * self.config.validation_ratio)
            n_test = int(n_samples * self.config.test_ratio)
            n_train = n_samples - n_val - n_test
            
            # 标准化特征和目标
            features_flat = combined_features.reshape(-1, combined_features.shape[-1])
            self.feature_scaler.fit(features_flat)
            
            signals_flat = combined_signals.reshape(-1, combined_signals.shape[-1])
            self.signal_scaler.fit(signals_flat)
            
            targets_flat = combined_targets.reshape(-1, 1)
            self.target_scaler.fit(targets_flat)
            
            # 特征标准化
            normalized_features = np.array([
                self.feature_scaler.transform(combined_features[i])
                for i in range(n_samples)
            ])
            
            # 信号标准化
            normalized_signals = self.signal_scaler.transform(combined_signals)
            
            # 目标标准化
            normalized_targets = self.target_scaler.transform(targets_flat).flatten()
            
            # 创建数据集
            train_data = {
                "features": normalized_features[:n_train],
                "signals": normalized_signals[:n_train],
                "targets": normalized_targets[:n_train]
            }
            
            val_data = {
                "features": normalized_features[n_train:n_train+n_val],
                "signals": normalized_signals[n_train:n_train+n_val],
                "targets": normalized_targets[n_train:n_train+n_val]
            }
            
            test_data = {
                "features": normalized_features[n_train+n_val:],
                "signals": normalized_signals[n_train+n_val:],
                "targets": normalized_targets[n_train+n_val:]
            }
            
            # 保存数据集到缓存
            if self.config.use_cache:
                dataset = (train_data, val_data, test_data)
                joblib.dump(dataset, cache_file)
                logger.info(f"数据集已保存到缓存: {cache_file}")
            
            return train_data, val_data, test_data
        else:
            logger.error("无法生成有效的训练数据")
            return None, None, None
    
    def _process_symbol(self, symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
        """处理单个股票的数据
        
        参数:
            symbol: 股票代码
            
        返回:
            features, signals, targets, dates
        """
        try:
            # 获取股票数据
            from strategy_optimizer.data_processors.data_processor import DataProcessor
            data_processor = DataProcessor()
            df = data_processor.get_stock_data(
                symbol, 
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if df is None or len(df) < self.config.window_size + self.config.target_horizon:
                logger.warning(f"股票 {symbol} 数据不足，跳过处理")
                return None, None, None, None
            
            # 特征工程
            df = self.feature_engineer.add_all_features(df)
            
            # 移除包含NaN的行
            df = df.dropna()
            
            if len(df) < self.config.window_size + self.config.target_horizon:
                logger.warning(f"处理后股票 {symbol} 数据不足，跳过处理")
                return None, None, None, None
                
            # 生成目标变量
            if self.config.target_type == "returns":
                # 使用未来n天的收益率作为目标
                df['target'] = df['Close'].shift(-self.config.target_horizon) / df['Close'] - 1
            elif self.config.target_type == "direction":
                # 使用未来n天的价格方向作为目标（1上涨，-1下跌）
                future_returns = df['Close'].shift(-self.config.target_horizon) / df['Close'] - 1
                df['target'] = np.sign(future_returns)
            else:
                raise ValueError(f"不支持的目标类型: {self.config.target_type}")
            
            # 由于target使用了future数据，我们需要移除末尾的行
            df = df.iloc[:-self.config.target_horizon]
            
            # 保存索引（日期）
            dates = df.index.tolist()
            
            # 获取策略信号
            signals = np.zeros((len(df), len(self.config.strategies)))
            for i, strategy_name in enumerate(self.config.strategies):
                try:
                    strategy = self.strategy_factory.create_strategy(strategy_name)
                    signal_df = strategy.generate_signals(df.copy())
                    
                    # 假设信号列名为'signal'
                    if 'signal' in signal_df.columns:
                        signals[:, i] = signal_df['signal'].values
                    else:
                        logger.warning(f"策略 {strategy_name} 未生成'signal'列")
                except Exception as e:
                    logger.error(f"生成策略 {strategy_name} 信号时出错: {e}")
            
            # 创建滑动窗口特征
            features = []
            filtered_signals = []
            filtered_targets = []
            filtered_dates = []
            
            feature_columns = [col for col in df.columns if col not in ['target', 'signal']]
            if self.config.feature_list is not None:
                available_cols = set(feature_columns).intersection(set(self.config.feature_list))
                feature_columns = list(available_cols)
            
            for i in range(self.config.window_size, len(df)):
                # 窗口特征
                window_df = df.iloc[i-self.config.window_size:i][feature_columns]
                features.append(window_df.values)
                
                # 当前策略信号
                filtered_signals.append(signals[i])
                
                # 目标值
                filtered_targets.append(df.iloc[i]['target'])
                
                # 日期
                filtered_dates.append(dates[i])
            
            return np.array(features), np.array(filtered_signals), np.array(filtered_targets), filtered_dates
            
        except Exception as e:
            logger.error(f"处理股票 {symbol} 数据时出错: {e}")
            return None, None, None, None
    
    def get_feature_dim(self) -> int:
        """获取特征维度
        
        返回:
            特征维度
        """
        # 可以通过生成一个示例数据集来获取特征维度
        train_data, _, _ = self.generate_dataset()
        if train_data and 'features' in train_data:
            return train_data['features'].shape[2]
        return 0
    
    def get_n_strategies(self) -> int:
        """获取策略数量
        
        返回:
            策略数量
        """
        return len(self.config.strategies)
    
    def get_strategy_names(self) -> List[str]:
        """获取策略名称列表
        
        返回:
            策略名称列表
        """
        return self.config.strategies 