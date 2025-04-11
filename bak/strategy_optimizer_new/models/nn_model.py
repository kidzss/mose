#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经网络信号组合模型

使用神经网络组合多个交易信号
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from strategy_optimizer.models.base_model import BaseSignalModel


class SignalCombinerNet(nn.Module):
    """信号组合神经网络"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        初始化神经网络
        
        参数:
            input_dim: 输入维度（信号数量）
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用Batch Normalization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
                
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        # 构建网络
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x).squeeze(-1)


class NeuralCombinationModel(BaseSignalModel):
    """
    神经网络信号组合模型
    
    使用神经网络组合多个交易信号
    """
    
    def __init__(
        self, 
        model_name: str = "neural_model",
        normalize_signals: bool = True,
        normalize_method: str = "zscore",
        normalize_window: Optional[int] = None,
        allow_short: bool = True,
        hidden_dims: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping: int = 10,
        loss_fn: str = "mse",
        device: Optional[str] = None,
        random_state: int = 42
    ):
        """
        初始化神经网络组合模型
        
        参数:
            model_name: 模型名称
            normalize_signals: 是否标准化信号
            normalize_method: 标准化方法
            normalize_window: 标准化窗口大小
            allow_short: 是否允许做空
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            use_batch_norm: 是否使用Batch Normalization
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            early_stopping: 早停轮数，0表示不使用早停
            loss_fn: 损失函数，可选：'mse', 'mae', 'sharpe', 'sortino'
            device: 设备，None表示自动选择
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
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.loss_fn = loss_fn
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.optimizer = None
        self.signal_names_ = None
    
    def fit(
        self, 
        signals: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.Series],
        val_signals: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        val_targets: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = False
    ) -> 'NeuralCombinationModel':
        """
        训练神经网络组合模型
        
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
        
        # 预处理数据
        X_train = self.preprocess_signals(signals)
        
        if isinstance(targets, pd.Series):
            y_train = targets.values
        else:
            y_train = targets
            
        # 处理样本权重
        if sample_weights is not None:
            if isinstance(sample_weights, pd.Series):
                weights = sample_weights.values
            else:
                weights = sample_weights
        else:
            weights = np.ones_like(y_train)
        
        # 转换为Tensor
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # 处理验证集
        if val_signals is not None and val_targets is not None:
            X_val = self.preprocess_signals(val_signals)
            
            if isinstance(val_targets, pd.Series):
                y_val = val_targets.values
            else:
                y_val = val_targets
                
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
                
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            has_validation = True
        else:
            has_validation = False
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if has_validation:
            val_weights = np.ones_like(y_val)
            val_weights_tensor = torch.FloatTensor(val_weights).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, val_weights_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        
        # 创建模型
        input_dim = X_train.shape[1]
        self.model = SignalCombinerNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 选择损失函数
        if self.loss_fn == "mse":
            criterion = nn.MSELoss(reduction='none')
        elif self.loss_fn == "mae":
            criterion = nn.L1Loss(reduction='none')
        elif self.loss_fn in ["sharpe", "sortino"]:
            criterion = nn.MSELoss(reduction='none')  # 使用MSE作为基础
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_fn}")
        
        # 训练模型
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # 清空历史记录
        self.history = {"train_loss": [], "val_loss": []}
        
        # 进度条
        pbar = tqdm(range(self.epochs), disable=not verbose)
        
        for epoch in pbar:
            # 训练
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for inputs, targets, weights in train_loader:
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算带权重的MSE/MAE损失
                batch_loss = criterion(outputs, targets)
                weighted_loss = (batch_loss * weights).mean()
                
                # 如果使用夏普比率或索提诺比率作为损失
                if self.loss_fn == "sharpe":
                    returns = outputs * torch.sign(outputs)  # 使用信号方向
                    mean_return = returns.mean()
                    std_return = returns.std()
                    if std_return > 0:
                        sharpe = -mean_return / std_return  # 最大化，所以取负
                        total_loss = weighted_loss + sharpe
                    else:
                        total_loss = weighted_loss
                elif self.loss_fn == "sortino":
                    returns = outputs * torch.sign(outputs)  # 使用信号方向
                    mean_return = returns.mean()
                    # 计算下行标准差
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        if downside_std > 0:
                            sortino = -mean_return / downside_std  # 最大化，所以取负
                            total_loss = weighted_loss + sortino
                        else:
                            total_loss = weighted_loss
                    else:
                        total_loss = weighted_loss
                else:
                    total_loss = weighted_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += batch_loss.mean().item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.history["train_loss"].append(avg_train_loss)
            
            # 验证
            if has_validation:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for inputs, targets, weights in val_loader:
                        outputs = self.model(inputs)
                        batch_loss = criterion(outputs, targets)
                        weighted_loss = (batch_loss * weights).mean()
                        
                        val_loss += batch_loss.mean().item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.history["val_loss"].append(avg_val_loss)
                
                # 早停
                if self.early_stopping > 0:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.early_stopping:
                        if verbose:
                            print(f"\n早停! 轮数 {epoch+1}/{self.epochs}")
                        break
                
                if verbose:
                    pbar.set_description(f"训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
            else:
                self.history["val_loss"].append(None)
                if verbose:
                    pbar.set_description(f"训练损失: {avg_train_loss:.6f}")
        
        # 加载最佳模型
        if has_validation and self.early_stopping > 0 and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
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
        
        # 预处理信号
        X = self.preprocess_signals(signals)
        
        # 获取索引
        if isinstance(signals, pd.DataFrame):
            index = signals.index
            X_np = X.values
        else:
            index = None
            X_np = X
            
        # 转换为Tensor
        X_tensor = torch.FloatTensor(X_np).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 返回与输入相同的格式
        if index is not None:
            return pd.Series(predictions, index=index, name="combined_signal")
        else:
            return predictions
    
    def get_weights(
        self, 
        signals: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> pd.Series:
        """
        获取模型权重
        
        对于神经网络模型，这个方法返回的是近似线性权重
        
        参数:
            signals: 用于计算权重的信号数据，如果为None，使用随机数据
            
        返回:
            近似权重Series
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 如果没有提供信号，使用随机数据
        if signals is None:
            signals = np.random.normal(0, 1, (1000, len(self.signal_names_)))
            if self.signal_names_ is not None:
                signals = pd.DataFrame(signals, columns=self.signal_names_)
        
        # 预处理信号
        X = self.preprocess_signals(signals)
        
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
        
        # 计算近似线性权重
        n_signals = X_np.shape[1]
        weights = np.zeros(n_signals)
        
        # 对每个信号单独测试
        for i in range(n_signals):
            # 创建一个单位向量
            unit = np.zeros(n_signals)
            unit[i] = 1.0
            
            # 转换为Tensor
            unit_tensor = torch.FloatTensor(unit).view(1, -1).to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                contribution = self.model(unit_tensor).item()
            
            weights[i] = contribution
        
        # 归一化权重
        weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else weights
        
        # 创建Series
        if self.signal_names_ is not None:
            return pd.Series(weights, index=self.signal_names_)
        else:
            return pd.Series(weights)
    
    def _save_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的保存逻辑
        
        参数:
            save_data: 保存数据字典
        """
        save_data["model_params"] = {
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "loss_fn": self.loss_fn,
            "device": str(self.device)
        }
        
        save_data["signal_names"] = self.signal_names_
        
        # 保存模型状态
        if self.model is not None:
            save_data["model_state_dict"] = self.model.state_dict()
    
    def _load_model_specific(self, save_data: Dict[str, Any]) -> None:
        """
        子类特定的加载逻辑
        
        参数:
            save_data: 保存数据字典
        """
        params = save_data["model_params"]
        self.hidden_dims = params["hidden_dims"]
        self.dropout_rate = params["dropout_rate"]
        self.use_batch_norm = params["use_batch_norm"]
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.early_stopping = params["early_stopping"]
        self.loss_fn = params["loss_fn"]
        self.device = torch.device(params["device"])
        
        self.signal_names_ = save_data["signal_names"]
        
        # 重建模型
        if "model_state_dict" in save_data:
            input_dim = len(self.signal_names_)
            self.model = SignalCombinerNet(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            ).to(self.device)
            
            self.model.load_state_dict(save_data["model_state_dict"])
            self.model.eval()  # 设置为评估模式 