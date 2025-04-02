#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础模型类

提供所有信号组合模型的通用功能
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import torch
import pickle
from datetime import datetime

from strategy_optimizer.utils.evaluation import evaluate_strategy
from strategy_optimizer.utils.normalization import normalize_signals, normalize_features


class BaseSignalModel:
    """信号组合模型的基类"""
    
    def __init__(
        self, 
        model_name: str = "base_model",
        normalize_signals: bool = True,
        normalize_method: str = "zscore",
        normalize_window: Optional[int] = None,
        allow_short: bool = True,
        random_state: int = 42
    ):
        """
        初始化基础模型
        
        参数:
            model_name: 模型名称
            normalize_signals: 是否标准化信号
            normalize_method: 标准化方法
            normalize_window: 标准化窗口大小
            allow_short: 是否允许做空
            random_state: 随机种子
        """
        self.model_name = model_name
        self.normalize_signals = normalize_signals
        self.normalize_method = normalize_method
        self.normalize_window = normalize_window
        self.allow_short = allow_short
        self.random_state = random_state
        
        self.is_fitted = False
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}
        self.performance_ = {}
        
        # 设置随机种子
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        torch.manual_seed(random_state)
        
    def fit(
        self, 
        signals: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        val_signals: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        val_targets: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> 'BaseSignalModel':
        """
        训练模型
        
        参数:
            signals: 信号数据，形状为[n_samples, n_signals]
            targets: 目标收益率，形状为[n_samples]
            val_signals: 验证集信号数据
            val_targets: 验证集目标收益率
            **kwargs: 额外参数
            
        返回:
            self
        """
        raise NotImplementedError("子类必须实现fit方法")
    
    def predict(
        self, 
        signals: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series]:
        """
        预测组合信号
        
        参数:
            signals: 信号数据，形状为[n_samples, n_signals]
            
        返回:
            组合信号，形状为[n_samples]
        """
        raise NotImplementedError("子类必须实现predict方法")
    
    def get_weights(
        self, 
        signals: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Union[np.ndarray, pd.Series]:
        """
        获取模型权重
        
        参数:
            signals: 可选的信号数据，用于动态权重模型
            
        返回:
            权重数组或Series
        """
        raise NotImplementedError("子类必须实现get_weights方法")
    
    def evaluate(
        self, 
        signals: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        detailed: bool = False
    ) -> Dict[str, float]:
        """
        评估模型表现
        
        参数:
            signals: 信号数据
            targets: 目标收益率
            detailed: 是否返回详细评估结果
            
        返回:
            评估指标字典
        """
        # 预测组合信号
        combined_signal = self.predict(signals)
        
        # 获取交易方向
        if self.allow_short:
            positions = np.sign(combined_signal)
        else:
            positions = np.maximum(np.sign(combined_signal), 0)
        
        # 转换数据类型
        if isinstance(targets, pd.Series):
            targets_np = targets.values
        else:
            targets_np = targets
            
        # 评估策略
        performance = evaluate_strategy(targets_np, positions)
        
        # 存储表现指标
        self.performance_ = performance
        
        return performance
    
    def preprocess_signals(
        self, 
        signals: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        预处理信号数据
        
        参数:
            signals: 原始信号数据
            
        返回:
            预处理后的信号数据
        """
        # 数据类型检查
        is_dataframe = isinstance(signals, pd.DataFrame)
        
        # 标准化信号
        if self.normalize_signals:
            signals_processed = normalize_signals(
                signals,
                method=self.normalize_method,
                window=self.normalize_window
            )
        else:
            signals_processed = signals
        
        return signals_processed
    
    def save(self, filepath: str, overwrite: bool = False) -> None:
        """
        保存模型
        
        参数:
            filepath: 保存路径
            overwrite: 是否覆盖已有文件
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"文件 {filepath} 已存在，设置 overwrite=True 可覆盖")
        
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 构建保存数据
        save_data = {
            "model_name": self.model_name,
            "parameters": {
                "normalize_signals": self.normalize_signals,
                "normalize_method": self.normalize_method,
                "normalize_window": self.normalize_window,
                "allow_short": self.allow_short,
                "random_state": self.random_state
            },
            "is_fitted": self.is_fitted,
            "history": self.history,
            "performance": self.performance_,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 子类特定的保存逻辑
        self._save_model_specific(save_data)
        
        # 保存到文件
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
            
        print(f"模型已保存到 {filepath}")
    
    def _save_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的保存逻辑
        
        参数:
            save_data: 保存数据字典
        """
        pass
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseSignalModel':
        """
        加载模型
        
        参数:
            filepath: 模型文件路径
            
        返回:
            加载的模型实例
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件 {filepath} 不存在")
        
        # 加载数据
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        
        # 创建模型实例
        model = cls(
            model_name=save_data["model_name"],
            **save_data["parameters"]
        )
        
        # 恢复通用属性
        model.is_fitted = save_data["is_fitted"]
        model.history = save_data["history"]
        model.performance_ = save_data["performance"]
        
        # 子类特定的加载逻辑
        model._load_model_specific(save_data)
        
        return model
    
    def _load_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的加载逻辑
        
        参数:
            save_data: 保存数据字典
        """
        pass
    
    def plot_training_history(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        绘制训练历史
        
        参数:
            figsize: 图形大小
            
        返回:
            图形对象
        """
        if not self.history["train_loss"]:
            raise ValueError("模型尚未训练")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制训练损失
        epochs = range(1, len(self.history["train_loss"]) + 1)
        ax.plot(epochs, self.history["train_loss"], 'b-', label='训练损失')
        
        # 如果有验证损失，也绘制
        if self.history["val_loss"]:
            ax.plot(epochs, self.history["val_loss"], 'r-', label='验证损失')
            ax.legend()
        
        ax.set_title(f'{self.model_name} 训练历史')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(True)
        
        plt.tight_layout()
        
        return fig 