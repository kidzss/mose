#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标准化工具模块

提供信号和特征的各种标准化方法
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List, Any, Callable


def normalize_signals(
    signals: Union[np.ndarray, pd.DataFrame],
    method: str = "zscore",
    window: Optional[int] = None,
    axis: int = 0,
    params: Optional[Dict[str, Any]] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    标准化信号数据
    
    参数:
        signals: 信号数据，形状为 [n_samples, n_signals]
        method: 标准化方法
            - "zscore": Z-score标准化 (减均值除标准差)
            - "minmax": Min-Max标准化 (缩放到[0,1])
            - "maxabs": MaxAbs标准化 (除以绝对值最大值)
            - "robust": Robust标准化 (使用中位数和四分位数范围)
            - "rank": 排序标准化 (转换为排名)
            - "quantile": 分位数标准化 (转换为分位数)
            - "winsorize": Winsorize标准化 (截断极端值)
            - "tanh": tanh标准化 (双曲正切变换)
            - "none": 不标准化
        window: 滚动窗口大小，None表示全局标准化
        axis: 标准化的轴，0表示按时间，1表示按特征
        params: 其他参数，特定于不同方法
    
    返回:
        标准化后的信号
    """
    # 检查参数
    if method not in [
        "zscore", "minmax", "maxabs", "robust", 
        "rank", "quantile", "winsorize", "tanh", "none"
    ]:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    if method == "none":
        return signals
    
    # 如果是None或空字典，初始化为空字典
    if params is None:
        params = {}
    
    # 保存原始数据格式
    is_dataframe = isinstance(signals, pd.DataFrame)
    if is_dataframe:
        index = signals.index
        columns = signals.columns
        signals_array = signals.values
    else:
        signals_array = signals
    
    # 全局标准化
    if window is None:
        if method == "zscore":
            epsilon = params.get("epsilon", 1e-8)
            mean = np.mean(signals_array, axis=axis, keepdims=True)
            std = np.std(signals_array, axis=axis, keepdims=True)
            std = np.where(std < epsilon, epsilon, std)  # 防止除零
            normalized = (signals_array - mean) / std
            
        elif method == "minmax":
            min_val = np.min(signals_array, axis=axis, keepdims=True)
            max_val = np.max(signals_array, axis=axis, keepdims=True)
            denominator = max_val - min_val
            # 防止除零
            denominator = np.where(denominator == 0, 1, denominator)
            normalized = (signals_array - min_val) / denominator
            
        elif method == "maxabs":
            max_abs = np.max(np.abs(signals_array), axis=axis, keepdims=True)
            max_abs = np.where(max_abs == 0, 1, max_abs)  # 防止除零
            normalized = signals_array / max_abs
            
        elif method == "robust":
            q1 = np.percentile(signals_array, 25, axis=axis, keepdims=True)
            q3 = np.percentile(signals_array, 75, axis=axis, keepdims=True)
            median = np.median(signals_array, axis=axis, keepdims=True)
            iqr = q3 - q1
            iqr = np.where(iqr == 0, 1e-8, iqr)  # 防止除零
            normalized = (signals_array - median) / iqr
            
        elif method == "rank":
            if axis == 0:  # 按时间
                normalized = np.zeros_like(signals_array)
                for i in range(signals_array.shape[1]):
                    normalized[:, i] = np.argsort(np.argsort(signals_array[:, i])) / (signals_array.shape[0] - 1)
            else:  # 按特征
                normalized = np.zeros_like(signals_array)
                for i in range(signals_array.shape[0]):
                    normalized[i, :] = np.argsort(np.argsort(signals_array[i, :])) / (signals_array.shape[1] - 1)
            
        elif method == "quantile":
            # 转换为分位数
            normalized = np.zeros_like(signals_array)
            n = signals_array.shape[axis]
            
            if axis == 0:  # 按时间
                for i in range(signals_array.shape[1]):
                    normalized[:, i] = _to_quantiles(signals_array[:, i])
            else:  # 按特征
                for i in range(signals_array.shape[0]):
                    normalized[i, :] = _to_quantiles(signals_array[i, :])
                    
        elif method == "winsorize":
            # 默认截断1%和99%分位数
            lower = params.get("lower", 0.01)
            upper = params.get("upper", 0.99)
            
            if axis == 0:  # 按时间
                normalized = np.zeros_like(signals_array)
                for i in range(signals_array.shape[1]):
                    normalized[:, i] = _winsorize(signals_array[:, i], lower, upper)
            else:  # 按特征
                normalized = np.zeros_like(signals_array)
                for i in range(signals_array.shape[0]):
                    normalized[i, :] = _winsorize(signals_array[i, :], lower, upper)
                    
        elif method == "tanh":
            # tanh标准化，先Z-score再用tanh变换
            scale = params.get("scale", 1.0)
            mean = np.mean(signals_array, axis=axis, keepdims=True)
            std = np.std(signals_array, axis=axis, keepdims=True)
            std = np.where(std < 1e-8, 1e-8, std)  # 防止除零
            z_scores = (signals_array - mean) / std
            normalized = np.tanh(scale * z_scores)
    
    # 滚动窗口标准化
    else:
        if axis != 0:
            raise ValueError("滚动窗口标准化仅支持按时间轴(axis=0)")
            
        normalized = np.zeros_like(signals_array)
        
        for i in range(signals_array.shape[0]):
            # 计算窗口起始位置
            start = max(0, i - window + 1)
            # 获取当前窗口
            window_data = signals_array[start:i+1]
            
            # 应用相应的标准化方法
            if method == "zscore":
                epsilon = params.get("epsilon", 1e-8)
                mean = np.mean(window_data, axis=0)
                std = np.std(window_data, axis=0)
                std = np.where(std < epsilon, epsilon, std)  # 防止除零
                normalized[i] = (signals_array[i] - mean) / std
                
            elif method == "minmax":
                min_val = np.min(window_data, axis=0)
                max_val = np.max(window_data, axis=0)
                denominator = max_val - min_val
                # 防止除零
                denominator = np.where(denominator == 0, 1, denominator)
                normalized[i] = (signals_array[i] - min_val) / denominator
                
            elif method == "maxabs":
                max_abs = np.max(np.abs(window_data), axis=0)
                max_abs = np.where(max_abs == 0, 1, max_abs)  # 防止除零
                normalized[i] = signals_array[i] / max_abs
                
            elif method == "robust":
                q1 = np.percentile(window_data, 25, axis=0)
                q3 = np.percentile(window_data, 75, axis=0)
                median = np.median(window_data, axis=0)
                iqr = q3 - q1
                iqr = np.where(iqr == 0, 1e-8, iqr)  # 防止除零
                normalized[i] = (signals_array[i] - median) / iqr
                
            elif method == "rank":
                for j in range(signals_array.shape[1]):
                    ranks = np.argsort(np.argsort(window_data[:, j]))
                    # 找到当前值在窗口中的排名
                    idx = np.searchsorted(np.sort(window_data[:, j]), signals_array[i, j])
                    normalized[i, j] = idx / (window_data.shape[0] - 1)
            
            elif method == "quantile":
                for j in range(signals_array.shape[1]):
                    normalized[i, j] = np.searchsorted(np.sort(window_data[:, j]), signals_array[i, j]) / window_data.shape[0]
                    
            elif method == "winsorize":
                lower = params.get("lower", 0.01)
                upper = params.get("upper", 0.99)
                
                for j in range(signals_array.shape[1]):
                    window_col = window_data[:, j]
                    low_val = np.percentile(window_col, lower * 100)
                    high_val = np.percentile(window_col, upper * 100)
                    val = signals_array[i, j]
                    
                    if val < low_val:
                        normalized[i, j] = low_val
                    elif val > high_val:
                        normalized[i, j] = high_val
                    else:
                        normalized[i, j] = val
                        
            elif method == "tanh":
                scale = params.get("scale", 1.0)
                mean = np.mean(window_data, axis=0)
                std = np.std(window_data, axis=0)
                std = np.where(std < 1e-8, 1e-8, std)  # 防止除零
                z_score = (signals_array[i] - mean) / std
                normalized[i] = np.tanh(scale * z_score)
    
    # 转换回原始数据格式
    if is_dataframe:
        return pd.DataFrame(normalized, index=index, columns=columns)
    else:
        return normalized


def normalize_features(
    features: Union[np.ndarray, pd.DataFrame],
    scaler_type: str = "standard",
    fit_scaler: bool = True,
    scaler: Optional[Any] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Any]:
    """
    标准化特征数据，使用sklearn标准化器
    
    参数:
        features: 特征数据，形状为 [n_samples, n_features]
        scaler_type: 标准化器类型
            - "standard": 标准化器 (Z-score)
            - "minmax": 最小最大标准化器
            - "maxabs": 最大绝对值标准化器
            - "robust": 稳健标准化器
            - "none": 不标准化
        fit_scaler: 是否拟合标准化器
        scaler: 已拟合的标准化器，如果提供则忽略scaler_type
    
    返回:
        (标准化后的特征, 标准化器)
    """
    # 如果提供了标准化器，使用提供的标准化器
    if scaler is not None:
        if fit_scaler:
            raise ValueError("提供了标准化器但fit_scaler=True，冲突的参数")
        
        # 转换特征
        if isinstance(features, pd.DataFrame):
            index = features.index
            columns = features.columns
            scaled_features = scaler.transform(features)
            return pd.DataFrame(scaled_features, index=index, columns=columns), scaler
        else:
            return scaler.transform(features), scaler
    
    # 如果scaler_type为none，返回原始特征
    if scaler_type == "none":
        return features, None
    
    # 导入sklearn标准化器
    try:
        from sklearn import preprocessing
    except ImportError:
        raise ImportError("需要安装scikit-learn才能使用sklearn标准化器")
    
    # 创建标准化器
    if scaler_type == "standard":
        scaler = preprocessing.StandardScaler()
    elif scaler_type == "minmax":
        scaler = preprocessing.MinMaxScaler()
    elif scaler_type == "maxabs":
        scaler = preprocessing.MaxAbsScaler()
    elif scaler_type == "robust":
        scaler = preprocessing.RobustScaler()
    else:
        raise ValueError(f"不支持的标准化器类型: {scaler_type}")
    
    # 拟合标准化器
    if fit_scaler:
        scaler.fit(features)
    
    # 转换特征
    if isinstance(features, pd.DataFrame):
        index = features.index
        columns = features.columns
        scaled_features = scaler.transform(features)
        return pd.DataFrame(scaled_features, index=index, columns=columns), scaler
    else:
        return scaler.transform(features), scaler


def _to_quantiles(x: np.ndarray) -> np.ndarray:
    """
    将数组转换为分位数
    
    参数:
        x: 输入数组
        
    返回:
        分位数数组
    """
    n = len(x)
    result = np.zeros_like(x)
    sorted_idx = np.argsort(x)
    
    for i, idx in enumerate(sorted_idx):
        result[idx] = i / (n - 1) if n > 1 else 0.5
    
    return result


def _winsorize(x: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """
    Winsorize数组（截断极端值）
    
    参数:
        x: 输入数组
        lower: 下限分位数
        upper: 上限分位数
        
    返回:
        Winsorize后的数组
    """
    result = x.copy()
    lower_val = np.percentile(x, lower * 100)
    upper_val = np.percentile(x, upper * 100)
    
    result[result < lower_val] = lower_val
    result[result > upper_val] = upper_val
    
    return result 