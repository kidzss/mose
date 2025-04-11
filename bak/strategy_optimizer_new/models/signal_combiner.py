import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

from strategy_optimizer.market_state import MarketState

logger = logging.getLogger(__name__)

class MarketRegime:
    """市场状态分析类"""
    
    def __init__(self, state_probs: np.ndarray):
        """初始化
        
        参数:
            state_probs: 市场状态概率 [n_samples, n_states]
        """
        self.state_probs = state_probs
        self.n_states = state_probs.shape[1]
        
    def detect_regimes(self) -> np.ndarray:
        """检测市场状态
        
        返回:
            市场状态分类结果 [n_samples]
        """
        # 获取每个样本最可能的状态
        regimes = np.argmax(self.state_probs, axis=1)
        return regimes
        
    def get_regime_transitions(self) -> pd.DataFrame:
        """获取市场状态转换矩阵
        
        返回:
            转换矩阵 DataFrame
        """
        regimes = self.detect_regimes()
        
        # 计算转换矩阵
        n_samples = len(regimes)
        transition_matrix = np.zeros((self.n_states, self.n_states))
        
        for i in range(n_samples - 1):
            from_state = regimes[i]
            to_state = regimes[i + 1]
            transition_matrix[from_state, to_state] += 1
            
        # 归一化
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     where=row_sums!=0, out=np.zeros_like(transition_matrix))
        
        # 转换为DataFrame
        state_names = [f"状态{i+1}" for i in range(self.n_states)]
        df = pd.DataFrame(transition_matrix, 
                          index=state_names,
                          columns=state_names)
        
        return df

@dataclass
class CombinerConfig:
    """信号组合器配置"""
    # 基本配置
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60  # 使用60天的历史数据
    
    # 模型配置
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.2
    weight_decay: float = 1e-5
    
    # 训练配置
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 市场状态配置
    n_market_states: int = 5
    market_feature_dim: int = 20
    
    # 信号组合配置
    use_market_state: bool = True
    time_varying_weights: bool = True
    regularization_strength: float = 0.01

class MarketStateClassifier(nn.Module):
    """市场状态分类器
    
    根据市场特征识别当前市场所处的状态/机制。
    """
    
    def __init__(self, config: CombinerConfig):
        """初始化市场状态分类器
        
        参数:
            config: 组合器配置
        """
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.market_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 时序模型 - LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim // 2,
            hidden_size=config.hidden_dim // 2,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.n_market_states),
            nn.Softmax(dim=1)
        )
        
    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            market_features: 市场特征 [batch_size, seq_len, feature_dim]
            
        返回:
            市场状态概率 [batch_size, n_market_states]
        """
        batch_size, seq_len, _ = market_features.shape
        
        # 提取特征
        features = []
        for t in range(seq_len):
            features.append(self.feature_extractor(market_features[:, t]))
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, hidden_dim//2]
        
        # LSTM处理
        lstm_out, (h_n, _) = self.lstm(features)
        
        # 双向LSTM输出合并
        h_n = h_n.view(self.config.n_layers, 2, batch_size, self.config.hidden_dim // 2)
        h_n = h_n[-1].transpose(0, 1).contiguous().view(batch_size, self.config.hidden_dim)
        
        # 分类
        market_state_probs = self.classifier(h_n)
        
        return market_state_probs

class AdaptiveWeightModel(nn.Module):
    """自适应权重模型
    
    根据市场状态为不同策略分配权重。
    """
    
    def __init__(self, n_strategies: int, config: CombinerConfig):
        """初始化自适应权重模型
        
        参数:
            n_strategies: 策略数量
            config: 组合器配置
        """
        super().__init__()
        self.config = config
        self.n_strategies = n_strategies
        
        # 基础权重 - 不依赖市场状态
        self.base_weights = nn.Parameter(torch.ones(n_strategies) / n_strategies)
        
        if config.use_market_state:
            # 市场状态相关的权重调整矩阵
            self.state_weight_matrix = nn.Parameter(
                torch.zeros(config.n_market_states, n_strategies)
            )
        
        # 权重调整层 - 时间自适应
        if config.time_varying_weights:
            self.time_varying_layer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, n_strategies)
            )
        
    def forward(
        self, 
        market_states: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算策略权重
        
        参数:
            market_states: 市场状态概率 [batch_size, n_market_states]
            hidden_state: 隐藏状态 [batch_size, hidden_dim]
            
        返回:
            策略权重 [batch_size, n_strategies]
        """
        batch_size = market_states.shape[0] if market_states is not None else hidden_state.shape[0]
        
        # 扩展基础权重
        weights = self.base_weights.expand(batch_size, -1)  # [batch_size, n_strategies]
        
        # 加入市场状态影响
        if self.config.use_market_state and market_states is not None:
            # [batch_size, n_market_states] @ [n_market_states, n_strategies]
            state_weights = torch.matmul(market_states, self.state_weight_matrix)
            weights = weights + state_weights
        
        # 加入时间变化影响
        if self.config.time_varying_weights and hidden_state is not None:
            time_weights = self.time_varying_layer(hidden_state)
            weights = weights + time_weights
        
        # 确保权重为正且和为1
        weights = F.softmax(weights, dim=1)
        
        return weights

class SignalCombiner(nn.Module):
    """信号组合器
    
    组合多个策略信号，根据市场状态动态调整权重。
    """
    
    def __init__(
        self, 
        n_strategies: int,
        input_dim: int,
        config: CombinerConfig
    ):
        """初始化信号组合器
        
        参数:
            n_strategies: 策略数量
            input_dim: 输入特征维度
            config: 组合器配置
        """
        super().__init__()
        self.config = config
        self.n_strategies = n_strategies
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 时序模型 - LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0
        )
        
        # 市场状态分类器
        if config.use_market_state:
            self.market_classifier = MarketStateClassifier(config)
        
        # 权重生成器
        self.weight_model = AdaptiveWeightModel(n_strategies, config)
        
    def forward(
        self, 
        features: torch.Tensor,
        strategy_signals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """前向传播
        
        参数:
            features: 市场特征 [batch_size, seq_len, feature_dim]
            strategy_signals: 各策略信号 [batch_size, n_strategies]
            
        返回:
            combined_signal: 组合信号 [batch_size, 1]
            weights: 策略权重 [batch_size, n_strategies]
            market_states: 市场状态概率 [batch_size, n_market_states] 或 None
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # 提取特征
        extracted_features = []
        for t in range(seq_len):
            extracted_features.append(self.feature_extractor(features[:, t]))
        extracted_features = torch.stack(extracted_features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM处理
        lstm_out, (h_n, _) = self.lstm(extracted_features)
        
        # 获取最后一层、最后一个时间步的隐藏状态
        hidden = h_n[-1]  # [batch_size, hidden_dim]
        
        # 市场状态分类
        market_states = None
        if self.config.use_market_state:
            market_states = self.market_classifier(features)
        
        # 生成权重
        weights = self.weight_model(market_states, hidden)
        
        # 组合信号
        combined_signal = torch.sum(strategy_signals * weights, dim=1, keepdim=True)
        
        return combined_signal, weights, market_states

class SignalCombinerModel:
    """信号组合模型
    
    包装SignalCombiner模型，提供训练、预测和评估功能。
    """
    
    def __init__(
        self,
        n_strategies: int,
        input_dim: int,
        config: Optional[CombinerConfig] = None
    ):
        """初始化信号组合模型
        
        参数:
            n_strategies: 策略数量
            input_dim: 输入特征维度
            config: 组合器配置，默认为None时使用默认配置
        """
        self.config = config if config is not None else CombinerConfig()
        self.device = torch.device(self.config.device)
        
        # 创建模型
        self.model = SignalCombiner(n_strategies, input_dim, self.config)
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 用于标准化
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(
        self,
        features: np.ndarray,
        strategy_signals: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备数据
        
        参数:
            features: 特征数据 [n_samples, seq_len, feature_dim]
            strategy_signals: 策略信号 [n_samples, n_strategies]
            targets: 目标值 [n_samples,]
            
        返回:
            train_loader, val_loader, test_loader
        """
        n_samples = features.shape[0]
        
        # 标准化特征
        features_flat = features.reshape(-1, features.shape[-1])
        self.feature_scaler.fit(features_flat)
        features_scaled = np.array([
            self.feature_scaler.transform(features[i])
            for i in range(n_samples)
        ])
        
        # 标准化目标
        targets_reshaped = targets.reshape(-1, 1)
        self.target_scaler.fit(targets_reshaped)
        targets_scaled = self.target_scaler.transform(targets_reshaped).flatten()
        
        # 划分数据集
        val_size = int(n_samples * self.config.validation_split)
        test_size = int(n_samples * self.config.test_split)
        train_size = n_samples - val_size - test_size
        
        # 创建数据集
        train_features = torch.FloatTensor(features_scaled[:train_size])
        train_signals = torch.FloatTensor(strategy_signals[:train_size])
        train_targets = torch.FloatTensor(targets_scaled[:train_size])
        
        val_features = torch.FloatTensor(features_scaled[train_size:train_size+val_size])
        val_signals = torch.FloatTensor(strategy_signals[train_size:train_size+val_size])
        val_targets = torch.FloatTensor(targets_scaled[train_size:train_size+val_size])
        
        test_features = torch.FloatTensor(features_scaled[train_size+val_size:])
        test_signals = torch.FloatTensor(strategy_signals[train_size+val_size:])
        test_targets = torch.FloatTensor(targets_scaled[train_size+val_size:])
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_features, train_signals, train_targets)
        val_dataset = TensorDataset(val_features, val_signals, val_targets)
        test_dataset = TensorDataset(test_features, test_signals, test_targets)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            verbose: 是否打印训练过程
            
        返回:
            训练历史记录
        """
        self.model.train()
        history = {
            'train_loss': [],
            'val_loss': [],
            'market_state_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # 训练
            train_loss = 0
            for batch_idx, (features, signals, targets) in enumerate(train_loader):
                features, signals, targets = (
                    features.to(self.device),
                    signals.to(self.device),
                    targets.to(self.device)
                )
                
                self.optimizer.zero_grad()
                combined_signal, weights, market_states = self.model(features, signals)
                
                # 计算损失
                loss = self.criterion(combined_signal.squeeze(), targets)
                
                # 添加权重L1正则化 - 鼓励稀疏权重
                if self.config.regularization_strength > 0:
                    l1_reg = self.config.regularization_strength * weights.abs().sum()
                    loss = loss + l1_reg
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if verbose and batch_idx % 10 == 0:
                    self.logger.info(f'Epoch: {epoch+1}/{self.config.epochs} '
                                    f'[{batch_idx}/{len(train_loader)}] '
                                    f'Loss: {loss.item():.6f}')
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            val_loss, market_acc = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['market_state_acc'].append(market_acc if market_acc is not None else 0)
            
            if verbose:
                self.logger.info(f'Epoch: {epoch+1}/{self.config.epochs} '
                              f'Train Loss: {train_loss:.6f} '
                              f'Val Loss: {val_loss:.6f}')
                if market_acc is not None:
                    self.logger.info(f'Market State Accuracy: {market_acc:.4f}')
            
            # 早停
            if val_loss < best_val_loss - self.config.early_stopping_patience:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f'早停: 验证损失在 {self.config.early_stopping_patience} 个epoch内没有改善')
                break
        
        return history
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Tuple[float, Optional[float]]:
        """评估模型
        
        参数:
            data_loader: 数据加载器
            
        返回:
            loss, market_state_accuracy (如果适用)
        """
        self.model.eval()
        total_loss = 0
        
        # 用于计算市场状态分类准确率
        correct_states = 0
        total_states = 0
        
        with torch.no_grad():
            for features, signals, targets in data_loader:
                features, signals, targets = (
                    features.to(self.device),
                    signals.to(self.device),
                    targets.to(self.device)
                )
                
                combined_signal, weights, market_states = self.model(features, signals)
                loss = self.criterion(combined_signal.squeeze(), targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        
        # 如果有市场状态数据，计算准确率
        market_acc = None
        if total_states > 0:
            market_acc = correct_states / total_states
            
        return avg_loss, market_acc
    
    def predict(
        self,
        features: np.ndarray,
        strategy_signals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """预测
        
        参数:
            features: 特征数据 [n_samples, seq_len, feature_dim]
            strategy_signals: 策略信号 [n_samples, n_strategies]
            
        返回:
            combined_signals, weights, market_states
        """
        self.model.eval()
        
        # 标准化特征
        features_flat = features.reshape(-1, features.shape[-1])
        features_scaled = np.array([
            self.feature_scaler.transform(features[i])
            for i in range(features.shape[0])
        ])
        
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        signals_tensor = torch.FloatTensor(strategy_signals).to(self.device)
        
        with torch.no_grad():
            combined_signal, weights, market_states = self.model(features_tensor, signals_tensor)
            
            # 反标准化信号
            combined_signal_np = combined_signal.cpu().numpy()
            combined_signal_np = self.target_scaler.inverse_transform(combined_signal_np)
            
            weights_np = weights.cpu().numpy()
            
            market_states_np = None
            if market_states is not None:
                market_states_np = market_states.cpu().numpy()
            
        return combined_signal_np, weights_np, market_states_np
    
    def fit(
        self,
        features: np.ndarray,
        strategy_signals: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """训练模型
        
        参数:
            features: 特征数据 [n_samples, seq_len, feature_dim]
            strategy_signals: 策略信号 [n_samples, n_strategies]
            targets: 目标值 [n_samples,]
            verbose: 是否打印训练过程
            
        返回:
            train_losses, val_losses
        """
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data(
            features, strategy_signals, targets
        )
        
        # 训练模型
        history = self.train(train_loader, val_loader, verbose)
        
        return history['train_loss'], history['val_loss']
    
    def save(self, path: str):
        """保存模型
        
        参数:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config
        }, path)
        
    def load(self, path: str):
        """加载模型
        
        参数:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
    def plot_weights(self, weights: np.ndarray, strategy_names: List[str], figsize: Tuple[int, int] = (10, 6)):
        """绘制权重分布
        
        参数:
            weights: 权重数据 [n_samples, n_strategies]
            strategy_names: 策略名称列表
            figsize: 图形大小
        """
        plt.figure(figsize=figsize)
        
        weight_df = pd.DataFrame(weights, columns=strategy_names)
        weight_df_melted = weight_df.reset_index().melt(id_vars=['index'], value_vars=strategy_names, 
                                                     var_name='Strategy', value_name='Weight')
        
        sns.boxplot(x='Strategy', y='Weight', data=weight_df_melted)
        plt.title('策略权重分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_market_states(self, market_states: np.ndarray, figsize: Tuple[int, int] = (10, 6)):
        """绘制市场状态分布
        
        参数:
            market_states: 市场状态数据 [n_samples, n_market_states]
            figsize: 图形大小
        """
        plt.figure(figsize=figsize)
        
        state_names = [state.name for state in MarketRegime]
        states_df = pd.DataFrame(market_states, columns=state_names)
        
        # 计算每个样本的主导市场状态
        dominant_states = states_df.idxmax(axis=1).value_counts().sort_index()
        
        plt.bar(dominant_states.index, dominant_states.values)
        plt.title('主导市场状态分布')
        plt.xlabel('市场状态')
        plt.ylabel('样本数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf() 