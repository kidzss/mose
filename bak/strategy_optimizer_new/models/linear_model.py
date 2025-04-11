#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
线性信号组合模型

使用线性方法组合多个交易信号
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from strategy_optimizer.models.base_model import BaseSignalModel
from strategy_optimizer.utils.signal_optimizer import SignalOptimizer


class LinearCombinationModel(BaseSignalModel):
    """
    线性信号组合模型
    
    使用线性模型组合多个交易信号
    """
    
    def __init__(
        self, 
        model_name: str = "linear_model",
        normalize_signals: bool = True,
        normalize_method: str = "zscore",
        normalize_window: Optional[int] = None,
        allow_short: bool = True,
        weights_constraint: str = "unit_sum",
        optimization_method: str = "sharpe",
        regularization: Optional[float] = None,
        random_state: int = 42
    ):
        """
        初始化线性组合模型
        
        参数:
            model_name: 模型名称
            normalize_signals: 是否标准化信号
            normalize_method: 标准化方法
            normalize_window: 标准化窗口大小
            allow_short: 是否允许做空
            weights_constraint: 权重约束方式
                - "unit_sum": 权重和为1
                - "unit_norm": 权重的L2范数为1
                - "simplex": 权重和为1且非负
                - None: 无约束
            optimization_method: 优化方法
                - "sharpe": 最大化夏普比率
                - "sortino": 最大化索提诺比率
                - "returns": 最大化收益率
                - "regression": 使用回归方法
                - "ensemble": 使用集成学习方法
            regularization: 正则化参数
            random_state: 随机种子
        """
        super().__init__(
            model_name=model_name,
            normalize_signals=normalize_signals,
            normalize_method=normalize_method,
            normalize_window=normalize_window,
            allow_short=allow_short,
            random_state=random_state
        )
        
        self.weights_constraint = weights_constraint
        self.optimization_method = optimization_method
        self.regularization = regularization
        
        self.optimizer = None
        self.weights_ = None
        self.signal_names_ = None
    
    def fit(
        self, 
        signals: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        val_signals: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        val_targets: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = False
    ) -> 'LinearCombinationModel':
        """
        训练线性组合模型
        
        参数:
            signals: 信号数据，形状为[n_samples, n_signals]
            targets: 目标收益率，形状为[n_samples]
            val_signals: 验证集信号数据
            val_targets: 验证集目标收益率
            sample_weights: 样本权重
            verbose: 是否显示详细信息
            
        返回:
            self
        """
        # 保存信号名称
        if isinstance(signals, pd.DataFrame):
            self.signal_names_ = signals.columns.tolist()
        else:
            self.signal_names_ = [f"Signal_{i+1}" for i in range(signals.shape[1])]
            
        # 创建优化器
        self.optimizer = SignalOptimizer(
            method=self.optimization_method,
            normalize=self.normalize_signals,
            normalize_method=self.normalize_method,
            normalize_window=self.normalize_window,
            weights_constraint=self.weights_constraint,
            allow_short=self.allow_short,
            regularization=self.regularization,
            random_state=self.random_state
        )
        
        # 优化权重
        self.weights_ = self.optimizer.optimize(
            signals=signals,
            targets=targets,
            sample_weights=sample_weights,
            verbose=verbose
        )
        
        # 设置为已训练
        self.is_fitted = True
        
        # 评估性能
        self.evaluate(signals, targets)
        
        # 如果有验证集，也评估
        if val_signals is not None and val_targets is not None:
            val_performance = self.evaluate(val_signals, val_targets)
            if verbose:
                print("\n验证集性能:")
                for metric, value in val_performance.items():
                    print(f"{metric}: {value:.4f}")
        
        return self
    
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
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if self.optimizer is not None:
            return self.optimizer.predict(signals)
        else:
            # 预处理信号
            signals_processed = self.preprocess_signals(signals)
            
            # 转换为numpy数组
            if isinstance(signals_processed, pd.DataFrame):
                signals_np = signals_processed.values
                index = signals_processed.index
            else:
                signals_np = signals_processed
                index = None
            
            # 计算组合信号
            combined_signal = np.dot(signals_np, self.weights_)
            
            # 返回与输入相同的格式
            if index is not None:
                return pd.Series(combined_signal, index=index, name="combined_signal")
            else:
                return combined_signal
    
    def get_weights(
        self, 
        signals: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> pd.Series:
        """
        获取模型权重
        
        参数:
            signals: 对于线性模型，此参数被忽略
            
        返回:
            权重Series
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        return self.weights_
    
    def _save_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的保存逻辑
        
        参数:
            save_data: 保存数据字典
        """
        save_data["weights"] = self.weights_.values if hasattr(self.weights_, "values") else self.weights_
        save_data["signal_names"] = self.signal_names_
        save_data["model_params"] = {
            "weights_constraint": self.weights_constraint,
            "optimization_method": self.optimization_method,
            "regularization": self.regularization
        }
    
    def _load_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的加载逻辑
        
        参数:
            save_data: 保存数据字典
        """
        self.weights_constraint = save_data["model_params"]["weights_constraint"]
        self.optimization_method = save_data["model_params"]["optimization_method"]
        self.regularization = save_data["model_params"]["regularization"]
        
        self.signal_names_ = save_data["signal_names"]
        weights = save_data["weights"]
        self.weights_ = pd.Series(weights, index=self.signal_names_)
    
    def plot_weights(
        self, 
        figsize: Tuple[int, int] = (10, 6),
        top_n: Optional[int] = None,
        sort_weights: bool = True
    ) -> plt.Figure:
        """
        绘制权重分布图
        
        参数:
            figsize: 图形大小
            top_n: 仅显示前N个权重（按绝对值排序）
            sort_weights: 是否按权重大小排序
            
        返回:
            图形对象
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if self.optimizer is not None:
            return self.optimizer.plot_weights(figsize, top_n, sort_weights)
        else:
            weights = self.weights_
            
            if sort_weights:
                # 按绝对值排序
                sorted_idx = np.argsort(-np.abs(weights))
                weights = weights.iloc[sorted_idx] if hasattr(weights, "iloc") else weights[sorted_idx]
            
            # 如果指定了top_n
            if top_n is not None and top_n < len(weights):
                weights = weights.iloc[:top_n] if hasattr(weights, "iloc") else weights[:top_n]
            
            # 创建图形
            fig, ax = plt.subplots(figsize=figsize)
            
            # 区分正负权重
            pos_mask = weights > 0
            neg_mask = weights < 0
            
            index = np.arange(len(weights))
            
            bars = ax.bar(
                index[pos_mask],
                weights[pos_mask],
                color="skyblue",
                label="正权重"
            )
            
            if np.any(neg_mask):
                bars = ax.bar(
                    index[neg_mask],
                    weights[neg_mask],
                    color="salmon",
                    label="负权重"
                )
            
            # 添加权重标签
            for i, w in enumerate(weights):
                if w != 0:
                    ax.text(
                        i, 
                        w + (0.01 if w > 0 else -0.03), 
                        f"{w:.4f}",
                        ha="center", 
                        va="bottom" if w > 0 else "top",
                        fontsize=8
                    )
            
            # 设置标签
            ax.set_xticks(index)
            labels = weights.index if hasattr(weights, "index") else [f"Signal_{i+1}" for i in range(len(weights))]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("权重")
            ax.set_title("信号组合权重分布")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            
            if np.any(neg_mask):
                ax.legend()
            
            plt.tight_layout()
            
            return fig 