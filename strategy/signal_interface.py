import enum
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime


class SignalType(enum.Enum):
    """信号类型枚举"""
    UNKNOWN = "unknown"        # 未知类型
    TREND = "trend"            # 趋势类型
    REVERSAL = "reversal"      # 反转类型
    MOMENTUM = "momentum"      # 动量类型
    VOLATILITY = "volatility"  # 波动类型
    SUPPORT = "support"        # 支撑位类型
    RESISTANCE = "resistance"  # 阻力位类型
    BREAKOUT = "breakout"      # 突破类型
    MEAN_REVERSION = "mean_reversion"  # 均值回归类型
    PATTERN = "pattern"        # 形态类型


class SignalTimeframe(enum.Enum):
    """信号时间框架枚举"""
    UNKNOWN = "unknown"        # 未知时间框架
    INTRADAY = "intraday"      # 日内
    DAILY = "daily"            # 日线
    WEEKLY = "weekly"          # 周线
    MONTHLY = "monthly"        # 月线
    QUARTERLY = "quarterly"    # 季线
    YEARLY = "yearly"          # 年线


class SignalStrength(enum.Enum):
    """信号强度枚举"""
    UNKNOWN = "unknown"        # 未知强度
    WEAK = "weak"              # 弱信号
    MODERATE = "moderate"      # 中等信号
    STRONG = "strong"          # 强信号
    EXTREME = "extreme"        # 极强信号


class SignalMetadata:
    """信号元数据类，用于描述信号的属性"""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 signal_type: SignalType = SignalType.UNKNOWN,
                 timeframe: SignalTimeframe = SignalTimeframe.UNKNOWN,
                 weight: float = 1.0,
                 version: str = "1.0.0",
                 normalization: str = "minmax",
                 normalization_params: Dict[str, Any] = None,
                 additional_info: Dict[str, Any] = None):
        """
        初始化信号元数据
        
        参数:
            name: 信号名称
            description: 信号的描述
            signal_type: 信号类型枚举
            timeframe: 信号时间框架
            weight: 信号权重
            version: 信号版本
            normalization: 规范化方法 ("minmax", "zscore", "none")
            normalization_params: 规范化参数
            additional_info: 额外信息
        """
        self.name = name
        self.description = description
        self.signal_type = signal_type
        self.timeframe = timeframe
        self.weight = weight
        self.version = version
        self.normalization = normalization
        self.normalization_params = normalization_params or {}
        self.additional_info = additional_info or {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            'name': self.name,
            'description': self.description,
            'signal_type': self.signal_type.value,
            'timeframe': self.timeframe.value,
            'weight': self.weight,
            'version': self.version,
            'normalization': self.normalization,
            'normalization_params': self.normalization_params,
            'additional_info': self.additional_info,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalMetadata':
        """从字典创建元数据对象"""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            signal_type=SignalType(data.get('signal_type', SignalType.UNKNOWN.value)),
            timeframe=SignalTimeframe(data.get('timeframe', SignalTimeframe.UNKNOWN.value)),
            weight=data.get('weight', 1.0),
            version=data.get('version', '1.0.0'),
            normalization=data.get('normalization', 'minmax'),
            normalization_params=data.get('normalization_params', {}),
            additional_info=data.get('additional_info', {})
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SignalMetadata':
        """从JSON字符串创建元数据对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class SignalComponent:
    """信号组件类，表示单一信号分量"""
    
    def __init__(self, 
                 series: pd.Series,
                 metadata: SignalMetadata):
        """
        初始化信号组件
        
        参数:
            series: 信号数据序列
            metadata: 信号元数据
        """
        self.series = series
        self.metadata = metadata
        self._normalized_series = None
    
    @property
    def normalized(self) -> pd.Series:
        """获取规范化后的信号序列"""
        if self._normalized_series is not None:
            return self._normalized_series
        
        # 进行规范化处理
        self._normalized_series = self._normalize_series()
        return self._normalized_series
    
    def _normalize_series(self) -> pd.Series:
        """对信号序列进行规范化处理"""
        series = self.series
        method = self.metadata.normalization
        params = self.metadata.normalization_params
        
        if method == 'none':
            return series
            
        elif method == 'minmax':
            # 获取最小值和最大值
            min_val = params.get('min', series.min())
            max_val = params.get('max', series.max())
            
            # 避免除以零
            range_val = max_val - min_val
            if range_val == 0:
                return pd.Series(0.5, index=series.index)
                
            # 归一化到 [0, 1]
            normalized = (series - min_val) / range_val
            
            # 将值限制在 [0, 1] 范围内
            normalized = normalized.clip(0, 1)
            
            return normalized
            
        elif method == 'zscore':
            # 获取均值和标准差
            mean = params.get('mean', series.mean())
            std = params.get('std', series.std())
            
            # 避免除以零
            if std == 0:
                return pd.Series(0, index=series.index)
                
            # Z-score 标准化
            normalized = (series - mean) / std
            
            # 将标准化后的值转换为 [-1, 1] 范围内的值
            normalized = normalized.clip(-3, 3) / 3
            
            return normalized
            
        else:
            # 默认不做处理
            return series
    
    def get_strength(self) -> SignalStrength:
        """获取当前信号强度"""
        # 获取最新的规范化值
        latest_value = self.normalized.iloc[-1] if not self.normalized.empty else 0
        
        # 根据规范化值判断强度
        if abs(latest_value) > 0.8:
            return SignalStrength.EXTREME
        elif abs(latest_value) > 0.6:
            return SignalStrength.STRONG
        elif abs(latest_value) > 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def get_direction(self) -> int:
        """
        获取信号方向
        
        返回:
            1: 正向（看涨）
            0: 中性
            -1: 负向（看跌）
        """
        # 获取最新的规范化值
        latest_value = self.normalized.iloc[-1] if not self.normalized.empty else 0
        
        # 根据规范化值判断方向
        if latest_value > 0.55:  # 添加了一点阈值以减少噪声
            return 1
        elif latest_value < 0.45:
            return -1
        else:
            return 0
            
    def __str__(self) -> str:
        """字符串表示"""
        return f"SignalComponent({self.metadata.name}, type={self.metadata.signal_type.value}, strength={self.get_strength().value})"


class SignalCombiner:
    """信号组合器，用于将多个信号组件组合成一个最终信号"""
    
    def __init__(self, components: Dict[str, SignalComponent]):
        """
        初始化信号组合器
        
        参数:
            components: 信号组件字典，键为组件名称
        """
        self.components = components
    
    def combine(self, method: str = 'weighted_average') -> pd.Series:
        """
        组合所有信号组件
        
        参数:
            method: 组合方法，支持 'weighted_average', 'majority_vote', 'min', 'max'
            
        返回:
            组合后的信号序列，值范围 [-1, 1]
        """
        if not self.components:
            return pd.Series()
            
        # 获取所有组件的规范化序列
        normalized_series = {name: comp.normalized for name, comp in self.components.items()}
        
        # 获取所有索引的并集
        all_indices = sorted(set().union(*[s.index for s in normalized_series.values()]))
        
        # 创建一个包含所有时间点的DataFrame
        df = pd.DataFrame(index=all_indices)
        
        # 填充数据
        for name, series in normalized_series.items():
            # 对于MinMax归一化的序列，将[0,1]映射到[-1,1]
            if self.components[name].metadata.normalization == 'minmax':
                df[name] = (series * 2 - 1).reindex(all_indices)
            else:
                df[name] = series.reindex(all_indices)
                
        # 根据方法组合信号
        if method == 'weighted_average':
            # 获取权重
            weights = {name: comp.metadata.weight for name, comp in self.components.items()}
            weight_sum = sum(weights.values())
            
            # 计算加权平均
            if weight_sum > 0:
                result = sum(df[name] * weights[name] for name in normalized_series.keys()) / weight_sum
            else:
                result = pd.Series(0, index=all_indices)
                
        elif method == 'majority_vote':
            # 将连续值转换为离散方向
            directions = pd.DataFrame(index=all_indices)
            for name, series in normalized_series.items():
                directions[name] = np.sign(df[name])  # 获取-1, 0, 1
                
            # 计算多数投票结果
            result = directions.mode(axis=1)[0]  # 获取每行的众数
            
        elif method == 'min':
            # 取最小值
            result = df.min(axis=1)
            
        elif method == 'max':
            # 取最大值
            result = df.max(axis=1)
            
        else:
            # 默认简单平均
            result = df.mean(axis=1)
            
        # 确保结果范围在 [-1, 1] 之间
        result = result.clip(-1, 1)
            
        return result
        
    def get_combined_signal(self) -> float:
        """
        获取组合后的最新信号值
        
        返回:
            信号值，范围 [-1, 1]
        """
        combined = self.combine()
        return combined.iloc[-1] if not combined.empty else 0.0
    
    def get_discrete_signal(self) -> int:
        """
        获取离散化的信号方向
        
        返回:
            1: 买入信号
            0: 无信号
            -1: 卖出信号
        """
        signal_value = self.get_combined_signal()
        
        # 设置阈值将连续信号转换为离散信号
        if signal_value > 0.3:  # 买入阈值
            return 1
        elif signal_value < -0.3:  # 卖出阈值
            return -1
        else:
            return 0  # 无信号


def normalize_signal(signal: pd.Series, method: str = 'minmax', 
                    params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    规范化信号序列的辅助函数
    
    参数:
        signal: 原始信号序列
        method: 规范化方法 ('minmax', 'zscore', 'none')
        params: 规范化参数
        
    返回:
        规范化后的信号序列
    """
    params = params or {}
    
    if method == 'minmax':
        # MinMax规范化
        min_val = params.get('min', signal.min())
        max_val = params.get('max', signal.max())
        
        range_val = max_val - min_val
        if range_val == 0:
            return pd.Series(0.5, index=signal.index)
            
        normalized = (signal - min_val) / range_val
        return normalized.clip(0, 1)
        
    elif method == 'zscore':
        # Z-score规范化
        mean = params.get('mean', signal.mean())
        std = params.get('std', signal.std())
        
        if std == 0:
            return pd.Series(0, index=signal.index)
            
        normalized = (signal - mean) / std
        normalized = normalized.clip(-3, 3) / 3  # 映射到[-1,1]范围
        
        # 将[-1,1]映射到[0,1]范围
        normalized = (normalized + 1) / 2
        return normalized
        
    else:
        # 不做处理
        return signal 