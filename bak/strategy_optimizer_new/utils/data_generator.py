#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级数据生成器

提供用于生成和预处理训练数据的工具
支持合成数据生成、市场数据加载和数据增强
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    高级数据生成器类
    
    用于创建模型训练和回测的数据集
    支持合成数据、历史数据加载和数据增强
    """
    
    def __init__(self, 
                 seed: int = 42,
                 normalize: bool = True,
                 normalize_method: str = "zscore"):
        """
        初始化数据生成器
        
        参数:
            seed: 随机种子
            normalize: 是否标准化数据
            normalize_method: 标准化方法，可选 "zscore", "minmax", "robust"
        """
        self.seed = seed
        self.normalize = normalize
        self.normalize_method = normalize_method
        np.random.seed(seed)
        
        self.scalers = {}
    
    def generate_synthetic_data(self, 
                                n_samples: int = 500, 
                                n_signals: int = 10,
                                start_date: str = "2020-01-01",
                                freq: str = "B",
                                signal_names: Optional[List[str]] = None,
                                correlation_matrix: Optional[np.ndarray] = None,
                                mean_return: float = 0.0005,
                                volatility: float = 0.01,
                                signal_strength: Dict[int, float] = None,
                                noise_level: float = 0.3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        生成合成数据
        
        参数:
            n_samples: 样本数量
            n_signals: 信号数量
            start_date: 起始日期
            freq: 频率，例如 'B' 表示工作日, 'D' 表示日历日
            signal_names: 信号名称列表，如果为None则自动生成
            correlation_matrix: 信号之间的相关性矩阵，如果为None则随机生成
            mean_return: 平均日收益率
            volatility: 收益率波动率
            signal_strength: 每个信号的预测强度，格式为 {信号索引: 强度}
            noise_level: 噪声水平 (0-1)
            
        返回:
            (信号DataFrame, 收益率Series)
        """
        # 生成日期
        dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
        
        # 生成信号名称
        if signal_names is None:
            signal_names = [f"Signal_{i+1}" for i in range(n_signals)]
        else:
            assert len(signal_names) == n_signals, "信号名称数量必须与信号数量一致"
        
        # 生成相关性矩阵
        if correlation_matrix is None:
            # 生成一个半正定矩阵作为相关性矩阵
            A = np.random.randn(n_signals, n_signals)
            correlation_matrix = np.dot(A, A.T)
            # 规范化为相关系数矩阵（对角线为1）
            d = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(d, d)
        
        # 生成多元正态分布的信号
        signals = np.random.multivariate_normal(
            mean=np.zeros(n_signals),
            cov=correlation_matrix,
            size=n_samples
        )
        
        # 生成潜在因子（市场因子）
        market_factor = np.random.normal(0, 1, n_samples)
        
        # 生成目标收益率
        returns = np.random.normal(mean_return, volatility, n_samples)
        
        # 如果指定了信号强度，让部分信号与收益率相关
        if signal_strength is not None:
            for idx, strength in signal_strength.items():
                if idx < n_signals:
                    # 将信号与市场因子和噪声的加权组合
                    noise = np.random.normal(0, 1, n_samples)
                    signals[:, idx] = strength * market_factor + (1 - strength) * noise
            
            # 让收益率也与市场因子相关
            returns = (1 - noise_level) * market_factor * volatility + noise_level * returns
        
        # 转换为pandas对象
        signals_df = pd.DataFrame(signals, index=dates, columns=signal_names)
        returns_series = pd.Series(returns, index=dates, name="Returns")
        
        # 标准化信号
        if self.normalize:
            signals_df = self._normalize_data(signals_df)
        
        return signals_df, returns_series
    
    def load_market_data(self, 
                         price_data: pd.DataFrame, 
                         return_period: int = 1,
                         calculate_features: bool = True,
                         feature_periods: List[int] = [5, 10, 20, 60],
                         volume_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从市场价格数据加载并计算信号和收益率
        
        参数:
            price_data: 价格数据DataFrame，索引为日期
            return_period: 收益率计算周期（天数）
            calculate_features: 是否计算基本技术指标作为特征
            feature_periods: 计算技术指标的周期
            volume_data: 交易量数据，用于计算成交量相关指标
            
        返回:
            (信号DataFrame, 收益率Series)
        """
        # 确保输入数据包含日期索引
        assert isinstance(price_data.index, pd.DatetimeIndex), "价格数据必须具有日期索引"
        
        # 计算未来收益率
        future_returns = price_data.pct_change(return_period).shift(-return_period)
        
        # 确保只有一列收益率，如果有多列则取第一列
        if isinstance(future_returns, pd.DataFrame):
            future_returns = future_returns.iloc[:, 0]
        
        # 初始化信号/特征列表
        features = pd.DataFrame(index=price_data.index)
        
        if calculate_features:
            # 1. 计算移动平均
            for period in feature_periods:
                features[f'MA_{period}'] = price_data.rolling(period).mean().iloc[:, 0]
                
                # 计算价格相对于移动平均的位置
                features[f'Price_to_MA_{period}'] = price_data.iloc[:, 0] / features[f'MA_{period}'] - 1
            
            # 2. 计算动量指标
            for period in feature_periods:
                features[f'Momentum_{period}'] = price_data.pct_change(period).iloc[:, 0]
            
            # 3. 计算波动率指标
            for period in feature_periods:
                features[f'Volatility_{period}'] = price_data.pct_change().rolling(period).std().iloc[:, 0]
            
            # 4. 计算RSI指标
            for period in [14, 30]:
                delta = price_data.diff().iloc[:, 0]
                up, down = delta.copy(), delta.copy()
                up[up < 0] = 0
                down[down > 0] = 0
                roll_up = up.rolling(period).mean()
                roll_down = down.abs().rolling(period).mean()
                rs = roll_up / roll_down
                features[f'RSI_{period}'] = 100.0 - (100.0 / (1.0 + rs))
            
            # 5. 添加交易量特征（如果提供）
            if volume_data is not None:
                # 交易量变化
                features['Volume_Change'] = volume_data.pct_change().iloc[:, 0]
                
                # 交易量移动平均
                features['Volume_MA_10'] = volume_data.rolling(10).mean().iloc[:, 0]
                
                # 相对交易量（当前交易量/平均交易量）
                features['Relative_Volume'] = volume_data.iloc[:, 0] / features['Volume_MA_10']
        
        # 删除含有NaN的行
        features = features.dropna()
        
        # 过滤掉对应的收益率
        filtered_returns = future_returns.loc[features.index]
        
        # 标准化特征
        if self.normalize:
            features = self._normalize_data(features)
        
        return features, filtered_returns
    
    def apply_data_augmentation(self, 
                                signals: pd.DataFrame, 
                                returns: pd.Series,
                                methods: List[str] = ["noise", "bootstrap", "mixup"],
                                augmentation_factor: int = 2,
                                noise_scale: float = 0.1,
                                bootstrap_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        应用数据增强技术
        
        参数:
            signals: 信号DataFrame
            returns: 收益率Series
            methods: 增强方法列表，可包含 "noise", "bootstrap", "mixup"
            augmentation_factor: 增强后数据集大小是原始数据集的倍数
            noise_scale: 添加噪声的标准差
            bootstrap_size: 自助采样的样本大小，默认为原始数据大小
            
        返回:
            (增强后的信号DataFrame, 增强后的收益率Series)
        """
        orig_index = signals.index
        orig_columns = signals.columns
        
        # 转换为numpy数组以便处理
        X = signals.values
        y = returns.values
        
        # 初始化增强后的数据
        X_aug = X.copy()
        y_aug = y.copy()
        
        # 应用指定的增强方法
        for method in methods:
            if method.lower() == "noise":
                # 添加高斯噪声
                for _ in range(augmentation_factor - 1):
                    noise = np.random.normal(0, noise_scale, X.shape)
                    X_noisy = X + noise
                    X_aug = np.vstack([X_aug, X_noisy])
                    y_aug = np.hstack([y_aug, y])
            
            elif method.lower() == "bootstrap":
                # 自助采样（有放回抽样）
                if bootstrap_size is None:
                    bootstrap_size = len(X)
                
                for _ in range(augmentation_factor - 1):
                    indices = np.random.choice(len(X), size=bootstrap_size, replace=True)
                    X_aug = np.vstack([X_aug, X[indices]])
                    y_aug = np.hstack([y_aug, y[indices]])
            
            elif method.lower() == "mixup":
                # Mixup增强（混合样本对）
                for _ in range(augmentation_factor - 1):
                    indices = np.random.permutation(len(X))
                    lam = np.random.beta(0.2, 0.2)  # 混合比例
                    X_mixed = lam * X + (1 - lam) * X[indices]
                    y_mixed = lam * y + (1 - lam) * y[indices]
                    X_aug = np.vstack([X_aug, X_mixed])
                    y_aug = np.hstack([y_aug, y_mixed])
        
        # 创建新索引
        new_dates = []
        for i in range(len(X_aug) // len(X)):
            new_dates.extend(orig_index)
        
        # 如果有余数，添加额外的日期
        remaining = len(X_aug) % len(X)
        if remaining > 0:
            new_dates.extend(orig_index[:remaining])
            
        # 转回DataFrame和Series
        augmented_signals = pd.DataFrame(
            X_aug, 
            index=new_dates[:len(X_aug)],
            columns=orig_columns
        )
        
        augmented_returns = pd.Series(
            y_aug,
            index=new_dates[:len(y_aug)],
            name=returns.name
        )
        
        return augmented_signals, augmented_returns
    
    def create_time_series_features(self, 
                                    signals: pd.DataFrame,
                                    window_sizes: List[int] = [5, 10, 20],
                                    functions: List[str] = ["mean", "std", "diff", "min", "max"],
                                    use_market_indicators: bool = False) -> pd.DataFrame:
        """
        创建时间序列特征
        
        参数:
            signals: 原始信号DataFrame
            window_sizes: 窗口大小列表
            functions: 要应用的函数列表
            use_market_indicators: 是否添加市场状态指标（波动率等）
            
        返回:
            扩展后的特征DataFrame
        """
        features = signals.copy()
        
        function_map = {
            "mean": lambda x: x.rolling(window=ws).mean(),
            "std": lambda x: x.rolling(window=ws).std(),
            "diff": lambda x: x.diff(ws),
            "min": lambda x: x.rolling(window=ws).min(),
            "max": lambda x: x.rolling(window=ws).max(),
            "skew": lambda x: x.rolling(window=ws).skew(),
            "kurt": lambda x: x.rolling(window=ws).kurt(),
            "quantile25": lambda x: x.rolling(window=ws).quantile(0.25),
            "quantile75": lambda x: x.rolling(window=ws).quantile(0.75)
        }
        
        # 为每个窗口大小和每个函数创建特征
        for ws in window_sizes:
            for func_name in functions:
                if func_name in function_map:
                    func = function_map[func_name]
                    feature_name_suffix = f"{func_name}_{ws}"
                    
                    for col in signals.columns:
                        series = signals[col]
                        features[f"{col}_{feature_name_suffix}"] = func(series)
        
        # 添加市场状态指标
        if use_market_indicators:
            # 例如：计算所有信号的相关性作为市场压力指标
            for ws in [20, 50]:
                # 计算滚动相关性矩阵
                roll_corr = signals.rolling(window=ws).corr()
                
                # 提取每个时间点的平均相关性（市场压力指标）
                avg_corr = []
                for dt in signals.index:
                    try:
                        # 获取该时间点的相关矩阵
                        corr_matrix = roll_corr.loc[dt]
                        # 提取上三角矩阵（不包括对角线）
                        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                        # 计算平均相关性
                        avg_corr.append(corr_matrix.where(mask).mean().mean())
                    except:
                        avg_corr.append(np.nan)
                
                features[f'Market_Pressure_{ws}'] = avg_corr
        
        # 删除NaN值
        features = features.dropna()
        
        # 标准化新特征
        if self.normalize:
            # 只标准化新添加的特征列
            new_columns = [col for col in features.columns if col not in signals.columns]
            if new_columns:
                features[new_columns] = self._normalize_data(features[new_columns])
        
        return features
    
    def split_data(self, 
                   signals: pd.DataFrame, 
                   returns: pd.Series,
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.2,
                   by_date: bool = True,
                   split_date: Optional[List[str]] = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        分割数据为训练集、验证集和测试集
        
        参数:
            signals: 信号DataFrame
            returns: 收益率Series
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            by_date: 是否按日期分割（时间序列分割）
            split_date: 显式指定分割日期，格式为 ["train_end", "val_end"]
            
        返回:
            包含分割后数据集的字典
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
        
        # 确保索引匹配
        common_index = signals.index.intersection(returns.index)
        signals = signals.loc[common_index]
        returns = returns.loc[common_index]
        
        # 按日期分割（适用于时间序列数据）
        if by_date:
            if split_date is not None:
                # 使用指定的分割日期
                assert len(split_date) == 2, "分割日期必须指定两个点：训练集结束和验证集结束"
                train_end = pd.Timestamp(split_date[0])
                val_end = pd.Timestamp(split_date[1])
                
                train_signals = signals[signals.index <= train_end]
                val_signals = signals[(signals.index > train_end) & (signals.index <= val_end)]
                test_signals = signals[signals.index > val_end]
                
                train_returns = returns[returns.index <= train_end]
                val_returns = returns[(returns.index > train_end) & (returns.index <= val_end)]
                test_returns = returns[returns.index > val_end]
            else:
                # 使用比例分割
                sorted_dates = sorted(signals.index)
                n = len(sorted_dates)
                train_end_idx = int(n * train_ratio)
                val_end_idx = int(n * (train_ratio + val_ratio))
                
                train_end = sorted_dates[train_end_idx]
                val_end = sorted_dates[val_end_idx]
                
                train_signals = signals[signals.index <= train_end]
                val_signals = signals[(signals.index > train_end) & (signals.index <= val_end)]
                test_signals = signals[signals.index > val_end]
                
                train_returns = returns[returns.index <= train_end]
                val_returns = returns[(returns.index > train_end) & (returns.index <= val_end)]
                test_returns = returns[returns.index > val_end]
        else:
            # 随机分割（不适用于时间序列数据，但对某些实验有用）
            n = len(signals)
            indices = np.random.permutation(n)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            train_signals = signals.iloc[train_idx]
            val_signals = signals.iloc[val_idx]
            test_signals = signals.iloc[test_idx]
            
            train_returns = returns.iloc[train_idx]
            val_returns = returns.iloc[val_idx]
            test_returns = returns.iloc[test_idx]
        
        # 返回分割后的数据集
        return {
            'train_signals': train_signals,
            'train_returns': train_returns,
            'val_signals': val_signals,
            'val_returns': val_returns,
            'test_signals': test_signals,
            'test_returns': test_returns
        }
    
    def create_pipeline(self, 
                        steps: List[Callable], 
                        signals: pd.DataFrame = None, 
                        returns: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        创建数据处理管道
        
        参数:
            steps: 处理步骤函数列表，每个函数接受(signals, returns)并返回(signals, returns)
            signals: 初始信号DataFrame，如果为None则需要在步骤中生成
            returns: 初始收益率Series，如果为None则需要在步骤中生成
            
        返回:
            (处理后的信号DataFrame, 处理后的收益率Series)
        """
        current_signals = signals
        current_returns = returns
        
        for step_func in steps:
            current_signals, current_returns = step_func(current_signals, current_returns)
        
        return current_signals, current_returns
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据
        
        参数:
            data: 要标准化的数据
            
        返回:
            标准化后的数据
        """
        result = data.copy()
        
        if self.normalize_method == "zscore":
            scaler = StandardScaler()
        elif self.normalize_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalize_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {self.normalize_method}")
        
        # 对每列单独标准化
        for col in data.columns:
            values = data[col].values.reshape(-1, 1)
            self.scalers[col] = scaler.fit(values)
            result[col] = scaler.transform(values).flatten()
        
        return result


# 使用示例
if __name__ == "__main__":
    # 初始化数据生成器
    data_gen = DataGenerator(seed=42, normalize=True)
    
    # 生成合成数据
    signals, returns = data_gen.generate_synthetic_data(
        n_samples=500,
        n_signals=10,
        signal_strength={0: 0.7, 1: 0.6, 2: 0.5},
        noise_level=0.3
    )
    
    print(f"生成的数据形状: 信号={signals.shape}, 收益率={returns.shape}")
    
    # 创建时间序列特征
    extended_features = data_gen.create_time_series_features(
        signals,
        window_sizes=[5, 10],
        functions=["mean", "std"],
        use_market_indicators=True
    )
    
    print(f"扩展后的特征形状: {extended_features.shape}")
    
    # 应用数据增强
    aug_signals, aug_returns = data_gen.apply_data_augmentation(
        signals, 
        returns,
        methods=["noise", "bootstrap"],
        augmentation_factor=2
    )
    
    print(f"增强后的数据形状: 信号={aug_signals.shape}, 收益率={aug_returns.shape}")
    
    # 分割数据
    split_data = data_gen.split_data(signals, returns)
    
    print(f"训练集大小: {split_data['train_signals'].shape}")
    print(f"验证集大小: {split_data['val_signals'].shape}")
    print(f"测试集大小: {split_data['test_signals'].shape}") 