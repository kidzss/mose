import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class MarketStateType(Enum):
    """市场状态类型"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"

@dataclass
class MarketState:
    """市场状态类"""
    state_type: MarketStateType
    confidence: float
    features: Dict[str, float]
    timestamp: pd.Timestamp
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketState':
        """从字典创建市场状态对象"""
        return cls(
            state_type=MarketStateType(data['state_type']),
            confidence=float(data['confidence']),
            features=data['features'],
            timestamp=pd.Timestamp(data['timestamp'])
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'state_type': self.state_type.value,
            'confidence': float(self.confidence),
            'features': self.features,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"MarketState(type={self.state_type.value}, confidence={self.confidence:.2f})"

class MarketStateAnalyzer:
    """市场状态分析器"""
    
    def __init__(self, window_size: int = 20):
        """初始化市场状态分析器
        
        参数:
            window_size: 分析窗口大小
        """
        self.window_size = window_size
        
    def analyze(self, data: pd.DataFrame) -> List[MarketState]:
        """分析市场状态
        
        参数:
            data: 市场数据DataFrame，包含OHLCV等数据
            
        返回:
            市场状态列表
        """
        states = []
        
        # 计算技术指标
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=self.window_size).std()
        trend = data['Close'].rolling(window=self.window_size).mean()
        
        # 分析每个时间点的市场状态
        for i in range(self.window_size, len(data)):
            current_returns = returns.iloc[i-self.window_size:i]
            current_volatility = volatility.iloc[i]
            current_trend = trend.iloc[i]
            current_price = data['Close'].iloc[i]
            
            # 计算状态特征
            features = {
                'returns_mean': current_returns.mean(),
                'returns_std': current_returns.std(),
                'trend_strength': abs(current_price - current_trend) / current_trend,
                'volatility': current_volatility
            }
            
            # 判断市场状态
            if current_price > current_trend and current_volatility < current_volatility.mean():
                state_type = MarketStateType.TRENDING_UP
                confidence = min(1.0, (current_price - current_trend) / current_trend)
            elif current_price < current_trend and current_volatility < current_volatility.mean():
                state_type = MarketStateType.TRENDING_DOWN
                confidence = min(1.0, (current_trend - current_price) / current_trend)
            elif current_volatility > current_volatility.mean() * 1.5:
                state_type = MarketStateType.VOLATILE
                confidence = min(1.0, current_volatility / current_volatility.mean())
            elif abs(current_price - current_trend) / current_trend < 0.01:
                state_type = MarketStateType.RANGING
                confidence = 1.0 - abs(current_price - current_trend) / current_trend
            else:
                state_type = MarketStateType.BREAKOUT
                confidence = min(1.0, abs(current_price - current_trend) / current_trend)
            
            # 创建市场状态对象
            state = MarketState(
                state_type=state_type,
                confidence=confidence,
                features=features,
                timestamp=data.index[i]
            )
            states.append(state)
        
        return states
    
    def get_state_transitions(self, states: List[MarketState]) -> pd.DataFrame:
        """获取市场状态转换矩阵
        
        参数:
            states: 市场状态列表
            
        返回:
            转换矩阵DataFrame
        """
        # 创建状态转换矩阵
        n_states = len(MarketStateType)
        transition_matrix = np.zeros((n_states, n_states))
        
        # 统计状态转换
        for i in range(len(states) - 1):
            from_state = states[i].state_type
            to_state = states[i + 1].state_type
            transition_matrix[from_state.value][to_state.value] += 1
        
        # 归一化
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     where=row_sums!=0, out=np.zeros_like(transition_matrix))
        
        # 创建DataFrame
        state_names = [state.value for state in MarketStateType]
        df = pd.DataFrame(transition_matrix, 
                         index=state_names,
                         columns=state_names)
        
        return df 

def create_market_features(data: pd.DataFrame, window_size: int = 20) -> Dict[str, float]:
    """创建市场特征
    
    参数:
        data: 市场数据DataFrame，包含OHLCV等数据
        window_size: 分析窗口大小
        
    返回:
        市场特征字典
    """
    # 计算技术指标
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window_size).std()
    trend = data['Close'].rolling(window=window_size).mean()
    
    # 计算特征
    features = {
        'returns_mean': returns.mean(),
        'returns_std': returns.std(),
        'trend_strength': abs(data['Close'].iloc[-1] - trend.iloc[-1]) / trend.iloc[-1],
        'volatility': volatility.iloc[-1],
        'volume_mean': data['Volume'].mean(),
        'volume_std': data['Volume'].std(),
        'price_range': (data['High'].max() - data['Low'].min()) / data['Close'].mean(),
        'price_momentum': (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
    }
    
    return features 