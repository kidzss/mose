#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强的时间序列交叉验证模块

提供专门用于时间序列数据的交叉验证工具，防止数据泄露和过拟合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Iterator, Any
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

logger = logging.getLogger(__name__)

class TimeSeriesCV:
    """
    时间序列交叉验证类
    
    专为金融时间序列数据设计的交叉验证工具，支持多种时间序列分割策略
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 max_train_size: Optional[int] = None,
                 expanding_window: bool = True,
                 min_train_size: Optional[int] = None,
                 shuffle: bool = False,
                 window_type: str = "rolling"):
        """
        初始化时间序列交叉验证
        
        参数:
            n_splits: 交叉验证折数
            test_size: 测试集大小，若为None则使用1/n_splits的数据
            gap: 训练集和测试集之间的间隔
            max_train_size: 训练集最大大小，若为None则使用所有可用数据
            expanding_window: 是否使用扩展窗口（增量训练）
            min_train_size: 训练集最小大小
            shuffle: 是否打乱数据（通常不推荐用于时间序列）
            window_type: 窗口类型，可选 "rolling", "expanding", "anchored"
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size
        self.shuffle = shuffle
        self.window_type = window_type
        
        if window_type not in ["rolling", "expanding", "anchored"]:
            raise ValueError(f"不支持的窗口类型: {window_type}")
        
        # 检查参数兼容性
        if window_type == "expanding" and max_train_size is not None:
            logger.warning("扩展窗口与max_train_size同时设置，max_train_size将被忽略")
        
        if shuffle:
            logger.warning("对时间序列数据启用shuffle可能导致未来数据泄露")
    
    def split(self, features, signals, targets):
        """
        生成训练集和测试集索引
        
        参数:
            features: 特征数据（可选，可以是None）
            signals: 信号数据
            targets: 目标数据
        
        返回:
            生成器，每次产生(train_data, test_data)，
            其中data是包含'features', 'signals', 'targets'的字典
        """
        # 确定样本数量
        if signals is not None:
            n_samples = len(signals)
        elif targets is not None:
            n_samples = len(targets)
        else:
            raise ValueError("signals和targets不能同时为None")
        
        # 创建基础分割器
        if self.window_type == "rolling":
            base_splitter = SklearnTimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                gap=self.gap,
                max_train_size=self.max_train_size
            )
            
            # 生成索引
            for train_idx, test_idx in base_splitter.split(np.arange(n_samples)):
                # 创建训练集数据
                train_data = {}
                if features is not None:
                    train_data['features'] = features[train_idx] if isinstance(features, np.ndarray) else features.iloc[train_idx]
                if signals is not None:
                    train_data['signals'] = signals[train_idx] if isinstance(signals, np.ndarray) else signals.iloc[train_idx]
                if targets is not None:
                    train_data['targets'] = targets[train_idx] if isinstance(targets, np.ndarray) else targets.iloc[train_idx]
                
                # 创建测试集数据
                test_data = {}
                if features is not None:
                    test_data['features'] = features[test_idx] if isinstance(features, np.ndarray) else features.iloc[test_idx]
                if signals is not None:
                    test_data['signals'] = signals[test_idx] if isinstance(signals, np.ndarray) else signals.iloc[test_idx]
                if targets is not None:
                    test_data['targets'] = targets[test_idx] if isinstance(targets, np.ndarray) else targets.iloc[test_idx]
                
                yield train_data, test_data
        else:  # expanding or anchored
            # 直接使用自定义分割器的生成器
            for train_idx, test_idx in self._create_custom_splitter(n_samples):
                # 创建训练集数据
                train_data = {}
                if features is not None:
                    train_data['features'] = features[train_idx] if isinstance(features, np.ndarray) else features.iloc[train_idx]
                if signals is not None:
                    train_data['signals'] = signals[train_idx] if isinstance(signals, np.ndarray) else signals.iloc[train_idx]
                if targets is not None:
                    train_data['targets'] = targets[train_idx] if isinstance(targets, np.ndarray) else targets.iloc[train_idx]
                
                # 创建测试集数据
                test_data = {}
                if features is not None:
                    test_data['features'] = features[test_idx] if isinstance(features, np.ndarray) else features.iloc[test_idx]
                if signals is not None:
                    test_data['signals'] = signals[test_idx] if isinstance(signals, np.ndarray) else signals.iloc[test_idx]
                if targets is not None:
                    test_data['targets'] = targets[test_idx] if isinstance(targets, np.ndarray) else targets.iloc[test_idx]
                
                yield train_data, test_data
    
    def _create_custom_splitter(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        创建自定义分割策略
        
        参数:
            n_samples: 样本数量
            
        返回:
            分割迭代器
        """
        # 计算测试集大小
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # 计算初始训练集大小
        if self.min_train_size is None:
            min_train_size = test_size * 2  # 默认至少是测试集大小的两倍
        else:
            min_train_size = self.min_train_size
        
        # 创建锚定窗口分割
        if self.window_type == "anchored":
            indices = np.arange(n_samples)
            anchor_end = min_train_size
            
            for i in range(self.n_splits):
                # 计算测试集开始位置
                test_start = n_samples - (self.n_splits - i) * test_size
                # 确保至少有最小训练集大小
                if test_start <= anchor_end + self.gap:
                    test_start = anchor_end + self.gap
                
                # 计算测试集结束位置
                test_end = test_start + test_size
                if test_end > n_samples:
                    test_end = n_samples
                
                # 提取训练集和测试集索引
                train_indices = indices[:anchor_end]
                test_indices = indices[test_start:test_end]
                
                yield train_indices, test_indices
                
                # 更新锚点位置（如果是扩展窗口）
                if self.expanding_window:
                    anchor_end = test_end
        
        # 创建扩展窗口分割
        else:  # window_type == "expanding"
            indices = np.arange(n_samples)
            initial_train_end = min_train_size
            
            for i in range(self.n_splits):
                # 计算测试集开始位置
                test_start = n_samples - (self.n_splits - i) * test_size
                # 确保至少有最小训练集大小加上间隔
                if test_start <= initial_train_end + self.gap:
                    test_start = initial_train_end + self.gap
                
                # 计算测试集结束位置
                test_end = test_start + test_size
                if test_end > n_samples:
                    test_end = n_samples
                
                # 提取训练集和测试集索引
                train_indices = indices[:test_start - self.gap]
                test_indices = indices[test_start:test_end]
                
                yield train_indices, test_indices
                
                # 扩展窗口会自动在下一次迭代中包含更多训练数据


class BlockingTimeSeriesCV:
    """
    分块时间序列交叉验证
    
    通过将时间序列分成几个连续的块进行交叉验证，特别适用于存在周期性的数据
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 validation_size: float = 0.2,
                 stride: Optional[int] = None,
                 start_offset: int = 0,
                 end_offset: int = 0,
                 purge_buffer: int = 0):
        """
        初始化分块时间序列交叉验证
        
        参数:
            n_splits: 数据块数量
            validation_size: 验证集比例
            stride: 块之间的步长，None表示均匀分布
            start_offset: 开始的样本数量偏移
            end_offset: 结束的样本数量偏移
            purge_buffer: 清除缓冲区大小，用于防止数据泄露
        """
        self.n_splits = n_splits
        self.validation_size = validation_size
        self.stride = stride
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.purge_buffer = purge_buffer
    
    def split(self, features, signals, targets):
        """
        生成训练集和测试集索引
        
        参数:
            features: 特征数据（可选，可以是None）
            signals: 信号数据
            targets: 目标数据
        
        返回:
            生成器，每次产生(train_data, test_data)
        """
        # 确定样本数量
        if signals is not None:
            n_samples = len(signals)
        elif targets is not None:
            n_samples = len(targets)
        else:
            raise ValueError("signals和targets不能同时为None")
        
        # 调整可用样本数量
        effective_samples = n_samples - self.start_offset - self.end_offset
        
        # 计算每个块的大小
        block_size = effective_samples // self.n_splits
        if self.stride is None:
            self.stride = block_size
        
        # 为每个块生成索引
        for i in range(self.n_splits):
            # 计算当前块的开始和结束位置
            block_start = self.start_offset + i * self.stride
            block_end = min(block_start + block_size, n_samples - self.end_offset)
            
            # 确保没有超出范围
            if block_start >= n_samples - self.end_offset:
                break
            
            # 计算验证集大小
            val_size = int(block_size * self.validation_size)
            
            # 计算训练集和验证集的索引
            train_start = block_start
            train_end = block_end - val_size
            val_start = train_end
            val_end = block_end
            
            # 应用清除缓冲区
            if self.purge_buffer > 0:
                # 移除训练集末尾的缓冲区
                train_end = max(train_start, train_end - self.purge_buffer)
                # 移除验证集开始的缓冲区
                val_start = min(val_start + self.purge_buffer, val_end)
            
            # 创建索引数组
            all_indices = np.arange(n_samples)
            train_indices = all_indices[train_start:train_end]
            val_indices = all_indices[val_start:val_end]
            
            # 创建训练集数据
            train_data = {}
            if features is not None:
                train_data['features'] = features[train_indices] if isinstance(features, np.ndarray) else features.iloc[train_indices]
            if signals is not None:
                train_data['signals'] = signals[train_indices] if isinstance(signals, np.ndarray) else signals.iloc[train_indices]
            if targets is not None:
                train_data['targets'] = targets[train_indices] if isinstance(targets, np.ndarray) else targets.iloc[train_indices]
            
            # 创建验证集数据
            val_data = {}
            if features is not None:
                val_data['features'] = features[val_indices] if isinstance(features, np.ndarray) else features.iloc[val_indices]
            if signals is not None:
                val_data['signals'] = signals[val_indices] if isinstance(signals, np.ndarray) else signals.iloc[val_indices]
            if targets is not None:
                val_data['targets'] = targets[val_indices] if isinstance(targets, np.ndarray) else targets.iloc[val_indices]
            
            yield train_data, val_data


class NestedTimeSeriesCV:
    """
    嵌套时间序列交叉验证
    
    用于超参数调优和模型选择的嵌套交叉验证，外层CV用于评估，内层CV用于调参
    """
    
    def __init__(self, 
                 outer_cv, 
                 inner_cv,
                 shuffle: bool = False):
        """
        初始化嵌套时间序列交叉验证
        
        参数:
            outer_cv: 外层交叉验证器
            inner_cv: 内层交叉验证器
            shuffle: 是否打乱数据（通常不推荐用于时间序列）
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.shuffle = shuffle
        
        if shuffle:
            logger.warning("对时间序列数据启用shuffle可能导致未来数据泄露")
    
    def split(self, features, signals, targets):
        """
        生成训练集、验证集和测试集索引
        
        参数:
            features: 特征数据（可选，可以是None）
            signals: 信号数据
            targets: 目标数据
        
        返回:
            生成器，每次产生(outer_train_data, test_data, inner_cv_generator)
            其中inner_cv_generator是内层CV的生成器
        """
        # 为外层CV生成分割
        for outer_train_data, test_data in self.outer_cv.split(features, signals, targets):
            # 提取外层训练数据
            outer_train_features = outer_train_data.get('features', None)
            outer_train_signals = outer_train_data.get('signals', None)
            outer_train_targets = outer_train_data.get('targets', None)
            
            # 创建内层CV生成器
            inner_cv_generator = self.inner_cv.split(
                outer_train_features, 
                outer_train_signals, 
                outer_train_targets
            )
            
            yield outer_train_data, test_data, inner_cv_generator


# 主要函数API

def walk_forward_validation(model, signals, returns, n_splits=5, test_size=None, 
                            gap=0, window_type="rolling", verbose=False):
    """
    执行时间序列交叉验证
    
    参数:
        model: 模型对象
        signals: 信号数据
        returns: 收益率数据
        n_splits: 交叉验证折数
        test_size: 测试集大小
        gap: 训练集和测试集之间的间隔
        window_type: 窗口类型，"rolling", "expanding", "anchored"
        verbose: 是否打印详细信息
        
    返回:
        (fold_metrics, avg_metrics, final_model)
    """
    # 创建时间序列交叉验证器
    tscv = TimeSeriesCV(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        window_type=window_type
    )
    
    # 初始化结果存储
    fold_metrics = []
    
    # 对每个折进行训练和评估
    fold = 1
    for train_data, test_data in tscv.split(None, signals, returns):
        if verbose:
            print(f"\n训练折 {fold}/{n_splits}")
            train_size = len(train_data['signals'])
            test_size = len(test_data['signals'])
            print(f"训练集索引: 0..{train_size-1} ({train_size}个样本)")
            print(f"测试集索引: {train_size}..{train_size+test_size-1} ({test_size}个样本)")
        
        # 复制模型以避免状态泄露
        model_copy = model.clone() if hasattr(model, 'clone') else model
        
        # 训练模型
        if verbose:
            print("训练模型...")
        model_copy.fit(train_data['signals'], train_data['targets'])
        
        # 评估模型
        if verbose:
            print("评估模型...")
        metrics = model_copy.evaluate(test_data['signals'], test_data['targets'])
        fold_metrics.append(metrics)
        
        if verbose:
            print(f"折 {fold} 性能:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        fold += 1
    
    # 计算平均指标
    all_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [fold[metric] for fold in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        all_metrics[metric] = (mean_val, std_val)
    
    if verbose:
        print("\n平均性能:")
        for metric, (mean_val, std_val) in all_metrics.items():
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 在全部数据上重新拟合最终模型
    if verbose:
        print("\n在全部数据上重新拟合模型...")
    final_model = model.clone() if hasattr(model, 'clone') else model
    final_model.fit(signals, returns)
    
    # 计算全部数据上的性能
    full_metrics = final_model.evaluate(signals, returns)
    
    if verbose:
        print("全部数据性能:")
        for metric, value in full_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 创建简化的平均指标（只有均值）
    avg_metrics = {k: v[0] for k, v in all_metrics.items()}
    
    return fold_metrics, avg_metrics, final_model


def multi_horizon_validation(model, signals, returns, horizons=[1, 5, 10, 20], 
                             n_splits=5, test_size=None, window_type="rolling", verbose=False):
    """
    多周期时间序列验证
    
    参数:
        model: 模型对象
        signals: 信号数据
        returns: 收益率数据
        horizons: 预测周期列表
        n_splits: 交叉验证折数
        test_size: 测试集大小
        window_type: 窗口类型
        verbose: 是否打印详细信息
        
    返回:
        {horizon: (fold_metrics, avg_metrics, final_model)}字典
    """
    results = {}
    
    for horizon in horizons:
        if verbose:
            print(f"\n=== 预测周期: {horizon} ===")
        
        # 生成适用于该周期的数据
        # 对于不同周期，需要调整目标变量
        horizon_returns = returns.shift(-horizon+1)
        valid_idx = ~horizon_returns.isna()
        
        # 执行交叉验证
        fold_metrics, avg_metrics, final_model = walk_forward_validation(
            model, signals.loc[valid_idx], horizon_returns.loc[valid_idx],
            n_splits=n_splits, test_size=test_size, 
            window_type=window_type, verbose=verbose
        )
        
        results[horizon] = (fold_metrics, avg_metrics, final_model)
    
    return results


def regime_based_validation(model, signals, returns, regime_indicator,
                             n_splits=5, test_size=None, window_type="rolling", verbose=False):
    """
    基于市场状态的交叉验证
    
    参数:
        model: 模型对象
        signals: 信号数据
        returns: 收益率数据
        regime_indicator: 市场状态指标，1=牛市，0=震荡市，-1=熊市
        n_splits: 交叉验证折数
        test_size: 测试集大小
        window_type: 窗口类型
        verbose: 是否打印详细信息
        
    返回:
        {regime: (fold_metrics, avg_metrics, final_model)}字典
    """
    results = {}
    regimes = np.unique(regime_indicator)
    
    for regime in regimes:
        if verbose:
            regime_names = {1: "牛市", 0: "震荡市", -1: "熊市"}
            regime_name = regime_names.get(regime, str(regime))
            print(f"\n=== 市场状态: {regime_name} ===")
        
        # 筛选该市场状态下的数据
        regime_mask = regime_indicator == regime
        regime_signals = signals.loc[regime_mask]
        regime_returns = returns.loc[regime_mask]
        
        # 检查数据量是否足够
        if len(regime_signals) < n_splits * 2:
            if verbose:
                print(f"该市场状态下的数据不足: {len(regime_signals)}个样本，跳过验证")
            continue
        
        # 执行交叉验证
        fold_metrics, avg_metrics, final_model = walk_forward_validation(
            model, regime_signals, regime_returns,
            n_splits=min(n_splits, len(regime_signals) // 10),  # 确保每折至少有10个样本
            test_size=test_size, 
            window_type=window_type, verbose=verbose
        )
        
        results[regime] = (fold_metrics, avg_metrics, final_model)
    
    return results


# 使用示例
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建模拟数据
    dates = pd.date_range(start='2020-01-01', periods=200, freq='B')
    signals = pd.DataFrame(np.random.randn(200, 5), index=dates, 
                        columns=[f'Signal_{i+1}' for i in range(5)])
    returns = pd.Series(np.random.randn(200) * 0.01, index=dates, name='Returns')
    
    # 创建模拟模型（这里用一个简单的类代替）
    class DummyModel:
        def fit(self, X, y):
            pass
            
        def predict(self, X):
            return np.zeros(len(X))
            
        def evaluate(self, X, y):
            return {'metric1': 0.5, 'metric2': 0.7}
    
    # 测试基本交叉验证
    print("测试基本交叉验证:")
    tscv = TimeSeriesCV(n_splits=5, test_size=20)
    for i, (train, test) in enumerate(tscv.split(None, signals, returns)):
        print(f"折 {i+1}:")
        print(f"  训练集大小: {len(train['signals'])}")
        print(f"  测试集大小: {len(test['signals'])}")
    
    # 测试分块交叉验证
    print("\n测试分块交叉验证:")
    block_cv = BlockingTimeSeriesCV(n_splits=4, validation_size=0.2)
    for i, (train, val) in enumerate(block_cv.split(None, signals, returns)):
        print(f"块 {i+1}:")
        print(f"  训练集大小: {len(train['signals'])}")
        print(f"  验证集大小: {len(val['signals'])}")
    
    # 测试嵌套交叉验证
    print("\n测试嵌套交叉验证:")
    outer_cv = TimeSeriesCV(n_splits=3, test_size=20)
    inner_cv = TimeSeriesCV(n_splits=2, test_size=10)
    nested_cv = NestedTimeSeriesCV(outer_cv, inner_cv)
    
    for i, (outer_train, test, inner_gen) in enumerate(nested_cv.split(None, signals, returns)):
        print(f"外层折 {i+1}:")
        print(f"  外层训练集大小: {len(outer_train['signals'])}")
        print(f"  测试集大小: {len(test['signals'])}")
        
        for j, (inner_train, inner_val) in enumerate(inner_gen):
            print(f"    内层折 {j+1}:")
            print(f"      内层训练集大小: {len(inner_train['signals'])}")
            print(f"      内层验证集大小: {len(inner_val['signals'])}") 