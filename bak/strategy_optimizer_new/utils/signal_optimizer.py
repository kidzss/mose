#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号优化器模块

提供信号组合优化的各种方法，用于确定最优信号权重
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
from scipy import stats

from strategy_optimizer.utils.normalization import normalize_signals


class SignalOptimizer:
    """
    信号优化器
    
    提供各种优化方法确定最优的信号组合权重
    """
    
    def __init__(
        self, 
        method: str = "sharpe",
        normalize: bool = True,
        normalize_method: str = "zscore",
        normalize_window: Optional[int] = None,
        weights_constraint: str = "unit_sum",
        allow_short: bool = True,
        regularization: Optional[float] = None,
        random_state: int = 42
    ):
        """
        初始化信号优化器
        
        参数:
            method: 优化方法
                - "sharpe": 最大化夏普比率
                - "sortino": 最大化索提诺比率
                - "returns": 最大化收益率
                - "min_variance": 最小化方差
                - "regression": 使用回归方法
            normalize: 是否标准化信号
            normalize_method: 标准化方法
            normalize_window: 标准化窗口大小
            weights_constraint: 权重约束方式
                - "unit_sum": 权重和为1
                - "unit_norm": 权重的L2范数为1
                - "simplex": 权重和为1且非负
                - None: 无约束
            allow_short: 是否允许做空（负权重）
            regularization: 正则化参数
            random_state: 随机种子
        """
        self.method = method
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.normalize_window = normalize_window
        self.weights_constraint = weights_constraint
        self.allow_short = allow_short
        self.regularization = regularization
        self.random_state = random_state
        
        # 权重和性能指标
        self.weights_ = None
        self.performance_ = None
        self.optimization_result_ = None
        
        # 设置随机种子
        np.random.seed(random_state)
        
        # 验证参数
        self._validate_params()
        
    def _validate_params(self):
        """验证参数"""
        valid_methods = ["sharpe", "sortino", "returns", "min_variance", "regression"]
        if self.method not in valid_methods:
            raise ValueError(f"method参数必须是{valid_methods}中的一个")
            
        valid_constraints = ["unit_sum", "unit_norm", "simplex", None]
        if self.weights_constraint not in valid_constraints:
            raise ValueError(f"weights_constraint参数必须是{valid_constraints}中的一个")
            
    def optimize(
        self, 
        signals: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        sample_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        constraints: Optional[List[Dict]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        initial_weights: Optional[Union[np.ndarray, List[float]]] = None,
        verbose: bool = False
    ) -> pd.Series:
        """
        优化信号权重
        
        参数:
            signals: 信号数据，形状为[n_samples, n_signals]
            targets: 目标收益率，形状为[n_samples]
            sample_weights: 样本权重，形状为[n_samples]
            constraints: 额外约束条件，传递给scipy.optimize
            bounds: 权重边界，传递给scipy.optimize
            initial_weights: 初始权重，如果不指定则使用均匀权重
            verbose: 是否显示详细信息
            
        返回:
            最优权重的Series
        """
        # 转换为numpy数组和DataFrame
        is_dataframe = isinstance(signals, pd.DataFrame)
        if is_dataframe:
            signal_names = signals.columns.tolist()
            signals_np = signals.values
        else:
            signals_np = signals
            signal_names = [f"Signal_{i+1}" for i in range(signals_np.shape[1])]
            
        if isinstance(targets, pd.Series):
            targets_np = targets.values
        else:
            targets_np = targets
            
        # 验证数据形状
        n_samples, n_signals = signals_np.shape
        if n_samples != len(targets_np):
            raise ValueError(f"signals和targets的样本数不一致: {n_samples} != {len(targets_np)}")
            
        # 处理样本权重
        if sample_weights is not None:
            if isinstance(sample_weights, pd.Series):
                weights_np = sample_weights.values
            else:
                weights_np = sample_weights
                
            if len(weights_np) != n_samples:
                raise ValueError(f"sample_weights和signals的样本数不一致: {len(weights_np)} != {n_samples}")
        else:
            weights_np = np.ones(n_samples)
            
        # 标准化信号
        if self.normalize:
            signals_normalized = normalize_signals(
                signals_np if not is_dataframe else signals,
                method=self.normalize_method,
                window=self.normalize_window
            )
            
            if is_dataframe:
                signals_np = signals_normalized.values
            else:
                signals_np = signals_normalized
        
        # 根据方法选择优化器
        if self.method == "regression":
            # 使用回归方法
            weights = self._optimize_regression(signals_np, targets_np, weights_np)
        else:
            # 初始化权重
            if initial_weights is None:
                initial_weights = np.ones(n_signals) / n_signals
            else:
                initial_weights = np.array(initial_weights)
                if len(initial_weights) != n_signals:
                    raise ValueError(f"initial_weights长度不匹配: {len(initial_weights)} != {n_signals}")
                    
            # 设置边界
            if bounds is None:
                if self.allow_short:
                    bounds = [(-1.0, 1.0) for _ in range(n_signals)]
                else:
                    bounds = [(0.0, 1.0) for _ in range(n_signals)]
                    
            # 创建约束
            all_constraints = []
            
            if self.weights_constraint == "unit_sum":
                # 权重和为1
                all_constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            elif self.weights_constraint == "unit_norm":
                # 权重的L2范数为1
                all_constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w**2) - 1.0})
            elif self.weights_constraint == "simplex":
                # 权重和为1且非负（此时bounds已设置为正）
                all_constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                
            # 添加额外约束
            if constraints is not None:
                all_constraints.extend(constraints)
                
            # 使用scipy.optimize优化
            if self.method == "sharpe":
                opt_func = lambda w: -self._sharpe_ratio(w, signals_np, targets_np, weights_np)
            elif self.method == "sortino":
                opt_func = lambda w: -self._sortino_ratio(w, signals_np, targets_np, weights_np)
            elif self.method == "returns":
                opt_func = lambda w: -self._mean_return(w, signals_np, targets_np, weights_np)
            elif self.method == "min_variance":
                opt_func = lambda w: self._portfolio_variance(w, signals_np)
            
            # 添加正则化
            if self.regularization is not None:
                original_func = opt_func
                if self.method in ["sharpe", "sortino", "returns"]:
                    # 对于最大化目标函数，添加L1/L2正则化项
                    opt_func = lambda w: original_func(w) + self.regularization * np.sum(np.abs(w))
                else:
                    # 对于最小化目标函数，添加L1/L2正则化项
                    opt_func = lambda w: original_func(w) + self.regularization * np.sum(np.abs(w))
            
            # 执行优化
            result = sco.minimize(
                opt_func, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds,
                constraints=all_constraints,
                options={'disp': verbose}
            )
            
            if not result.success:
                if verbose:
                    print(f"优化警告: {result.message}")
                    
            weights = result.x
            self.optimization_result_ = result
            
        # 标准化权重
        weights = self._normalize_weights(weights)
        
        # 保存为Series
        self.weights_ = pd.Series(weights, index=signal_names)
        
        # 计算性能指标
        self.performance_ = self._calculate_performance(signals_np, targets_np, weights)
        
        # 输出结果
        if verbose:
            print("\n优化结果:")
            print(f"目标函数值: {result.fun if hasattr(result, 'fun') else 'N/A'}")
            print("\n权重:")
            for name, weight in self.weights_.items():
                print(f"{name}: {weight:.6f}")
                
            print("\n性能指标:")
            for metric, value in self.performance_.items():
                print(f"{metric}: {value:.6f}")
                
        return self.weights_
    
    def predict(
        self, 
        signals: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series]:
        """
        使用优化的权重预测组合信号
        
        参数:
            signals: 信号数据，形状为[n_samples, n_signals]
            
        返回:
            组合信号，形状为[n_samples]
        """
        if self.weights_ is None:
            raise ValueError("尚未优化权重，请先调用optimize方法")
            
        # 保存原始格式
        is_dataframe = isinstance(signals, pd.DataFrame)
        if is_dataframe:
            index = signals.index
            
        # 标准化信号
        if self.normalize:
            signals_normalized = normalize_signals(
                signals,
                method=self.normalize_method,
                window=self.normalize_window
            )
        else:
            signals_normalized = signals
            
        # 转换为numpy数组
        if is_dataframe:
            signals_np = signals_normalized.values
            
            # 检查特征名称是否匹配
            required_columns = self.weights_.index.tolist()
            if not all(col in signals.columns for col in required_columns):
                raise ValueError("输入信号的列名与训练时不匹配")
                
            # 确保顺序一致
            weights = np.array([self.weights_[col] if col in self.weights_.index else 0 
                              for col in signals.columns])
        else:
            signals_np = signals_normalized
            weights = self.weights_.values
            
        # 计算加权和
        combined_signal = np.dot(signals_np, weights)
        
        # 返回与输入格式一致的结果
        if is_dataframe:
            return pd.Series(combined_signal, index=index, name="combined_signal")
        else:
            return combined_signal
    
    def _sharpe_ratio(
        self, 
        weights: np.ndarray, 
        signals: np.ndarray, 
        targets: np.ndarray, 
        sample_weights: np.ndarray
    ) -> float:
        """计算夏普比率"""
        # 计算组合信号
        combined_signal = np.dot(signals, weights)
        
        # 计算信号方向
        if self.allow_short:
            direction = np.sign(combined_signal)
        else:
            direction = np.maximum(np.sign(combined_signal), 0)
            
        # 计算策略收益率
        strategy_returns = targets * direction
        
        # 应用样本权重
        weighted_returns = strategy_returns * sample_weights
        
        # 计算加权平均收益率和标准差
        mean_return = np.sum(weighted_returns) / np.sum(sample_weights)
        std_return = np.sqrt(
            np.sum(sample_weights * (strategy_returns - mean_return) ** 2) / np.sum(sample_weights)
        )
        
        # 计算夏普比率
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        return sharpe
    
    def _sortino_ratio(
        self, 
        weights: np.ndarray, 
        signals: np.ndarray, 
        targets: np.ndarray, 
        sample_weights: np.ndarray
    ) -> float:
        """计算索提诺比率"""
        # 计算组合信号
        combined_signal = np.dot(signals, weights)
        
        # 计算信号方向
        if self.allow_short:
            direction = np.sign(combined_signal)
        else:
            direction = np.maximum(np.sign(combined_signal), 0)
            
        # 计算策略收益率
        strategy_returns = targets * direction
        
        # 应用样本权重
        weighted_returns = strategy_returns * sample_weights
        
        # 计算加权平均收益率
        mean_return = np.sum(weighted_returns) / np.sum(sample_weights)
        
        # 计算下行风险
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_weights = sample_weights[strategy_returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.sqrt(
                np.sum(downside_weights * downside_returns ** 2) / np.sum(downside_weights)
            )
        else:
            downside_std = 0
            
        # 计算索提诺比率
        sortino = mean_return / downside_std if downside_std > 0 else 0
        
        return sortino
    
    def _mean_return(
        self, 
        weights: np.ndarray, 
        signals: np.ndarray, 
        targets: np.ndarray, 
        sample_weights: np.ndarray
    ) -> float:
        """计算平均收益率"""
        # 计算组合信号
        combined_signal = np.dot(signals, weights)
        
        # 计算信号方向
        if self.allow_short:
            direction = np.sign(combined_signal)
        else:
            direction = np.maximum(np.sign(combined_signal), 0)
            
        # 计算策略收益率
        strategy_returns = targets * direction
        
        # 应用样本权重
        weighted_returns = strategy_returns * sample_weights
        
        # 计算加权平均收益率
        mean_return = np.sum(weighted_returns) / np.sum(sample_weights)
        
        return mean_return
    
    def _portfolio_variance(
        self, 
        weights: np.ndarray, 
        signals: np.ndarray
    ) -> float:
        """计算组合方差"""
        # 计算协方差矩阵
        cov_matrix = np.cov(signals, rowvar=False)
        
        # 计算组合方差
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        return portfolio_var
    
    def _optimize_regression(
        self, 
        signals: np.ndarray, 
        targets: np.ndarray, 
        sample_weights: np.ndarray
    ) -> np.ndarray:
        """使用回归方法优化权重"""
        # 使用有样本权重的线性回归
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(signals, targets, sample_weight=sample_weights)
            weights = lr.coef_
        except ImportError:
            # 如果没有sklearn，使用numpy直接求解
            # 带权重的线性回归公式: w = (X^T * W * X)^-1 * X^T * W * y
            X = signals
            y = targets
            W = np.diag(sample_weights)
            
            # 计算 X^T * W * X
            XtWX = X.T @ W @ X
            
            # 正则化，如果需要
            if self.regularization is not None:
                # 添加正则化项到对角线
                XtWX = XtWX + self.regularization * np.eye(XtWX.shape[0])
                
            # 计算 X^T * W * y
            XtWy = X.T @ W @ y
            
            # 求解
            weights = np.linalg.solve(XtWX, XtWy)
        
        # 如果不允许做空，将负权重设为0
        if not self.allow_short:
            weights = np.maximum(weights, 0)
            
        return weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """标准化权重"""
        if self.weights_constraint == "unit_sum":
            # 权重和为1
            weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else weights
        elif self.weights_constraint == "unit_norm":
            # 权重的L2范数为1
            norm = np.sqrt(np.sum(weights ** 2))
            weights = weights / norm if norm > 0 else weights
        elif self.weights_constraint == "simplex":
            # 权重和为1且非负
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
        return weights
    
    def _calculate_performance(
        self, 
        signals: np.ndarray, 
        targets: np.ndarray, 
        weights: np.ndarray
    ) -> Dict[str, float]:
        """计算性能指标"""
        # 计算组合信号
        combined_signal = np.dot(signals, weights)
        
        # 计算信号方向
        if self.allow_short:
            direction = np.sign(combined_signal)
        else:
            direction = np.maximum(np.sign(combined_signal), 0)
            
        # 计算策略收益率
        strategy_returns = targets * direction
        
        # 计算平均收益率
        mean_return = np.mean(strategy_returns)
        
        # 计算标准差
        std_return = np.std(strategy_returns)
        
        # 计算夏普比率
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # 计算下行标准差
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # 计算索提诺比率
        sortino = mean_return / downside_std if downside_std > 0 else 0
        
        # 计算胜率
        win_rate = np.mean(strategy_returns > 0)
        
        # 计算信息系数
        ic = np.corrcoef(combined_signal, targets)[0, 1] if len(combined_signal) > 1 else 0
        
        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "information_coefficient": ic
        }
    
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
        if self.weights_ is None:
            raise ValueError("尚未优化权重，请先调用optimize方法")
            
        weights = self.weights_.copy()
        
        if sort_weights:
            # 按绝对值排序
            weights = weights.reindex(weights.abs().sort_values(ascending=False).index)
        
        # 如果指定了top_n
        if top_n is not None and top_n < len(weights):
            weights = weights.iloc[:top_n]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 区分正负权重
        pos_mask = weights > 0
        neg_mask = weights < 0
        
        bars = ax.bar(
            weights.index[pos_mask],
            weights[pos_mask],
            color="skyblue",
            label="正权重"
        )
        
        if any(neg_mask):
            bars = ax.bar(
                weights.index[neg_mask],
                weights[neg_mask],
                color="salmon",
                label="负权重"
            )
        
        # 添加权重标签
        for i, (name, w) in enumerate(weights.items()):
            ax.text(
                i, 
                w + (0.01 if w > 0 else -0.03), 
                f"{w:.4f}",
                ha="center", 
                va="bottom" if w > 0 else "top",
                fontsize=8
            )
        
        # 设置标签
        ax.set_xticklabels(weights.index, rotation=45, ha="right")
        ax.set_ylabel("权重")
        ax.set_title("信号组合权重分布")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        if any(neg_mask):
            ax.legend()
        
        plt.tight_layout()
        
        return fig 