# strategy_optimizer/extractors/strategy_signal_extractor.py
"""
策略信号提取器模块

这个模块负责从不同的交易策略中提取和标准化信号，为后续的机器学习模型提供输入特征。
它能够：
1. 从单个策略中提取信号和核心组件
2. 从多个策略中批量提取信号
3. 根据信号的元数据对信号重要性进行排序
4. 对不同类型的信号进行适当的标准化处理（如RSI归一化到0-1之间）

主要类:
- StrategySignalExtractor: 策略信号提取器类，提供从策略中提取标准化信号的功能

使用示例:
```python
# 创建策略实例
strategies = [RSIStrategy(), MACDStrategy(), BollingerBandsStrategy()]

# 创建信号提取器
extractor = StrategySignalExtractor()

# 提取信号
signals_df = extractor.extract_signals_from_strategies(strategies, price_data)

# 获取信号重要性排名
importance_scores = extractor.rank_signals_by_importance()
```
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from strategy.strategy_base import Strategy

class StrategySignalExtractor:
    """
    策略信号提取器
    
    从各种交易策略中提取标准化信号并对其进行处理。该类提供了一套完整的工具，
    用于从不同类型的交易策略中提取、处理和标准化信号，使其适合作为机器学习模型的输入特征。
    
    主要功能：
    - 从单个或多个策略中提取信号
    - 对不同类型的信号进行合适的标准化处理
    - 记录和管理信号的元数据（信号来源、类别、描述等）
    - 根据元数据对信号进行重要性排序
    
    属性:
        strategy_signals (Dict): 存储所有策略的信号字典，键为策略名称，值为信号DataFrame
        signal_metadata (Dict): 存储信号的元数据字典，包含信号的描述、来源、重要性等信息
    """
    
    def __init__(self):
        """
        初始化策略信号提取器
        
        初始化两个主要的存储容器：
        1. strategy_signals: 用于存储从各个策略中提取的信号
        2. signal_metadata: 用于存储每个信号的元数据
        """
        self.strategy_signals = {}  # 存储所有策略的信号
        self.signal_metadata = {}   # 存储信号的元数据
    
    def extract_signals_from_strategy(self, 
                                     strategy: Strategy, 
                                     data: pd.DataFrame,
                                     prefix: Optional[str] = None) -> pd.DataFrame:
        """
        从单个策略中提取信号并进行标准化处理
        
        该方法执行以下步骤：
        1. 调用策略的generate_signals方法生成原始信号
        2. 使用策略的extract_signal_components方法提取关键信号组件
        3. 获取信号相关的元数据信息
        4. 对不同类型的信号进行适当的标准化处理
        5. 保存提取的信号和元数据到类的存储容器中
        
        参数:
            strategy (Strategy): 策略实例，必须实现Strategy基类接口
            data (pd.DataFrame): 包含OHLCV数据的DataFrame，用于生成信号
            prefix (Optional[str]): 信号列名的前缀，用于区分不同策略的信号，默认为策略名称
            
        返回:
            pd.DataFrame: 包含标准化处理后的信号的DataFrame，索引与输入数据相同
            
        注意:
            不同类型的信号会进行不同的标准化处理：
            - RSI类指标会归一化到0-1之间
            - MACD等指标会相对于价格进行缩放
            - 其他指标保持原样
        """
        # 生成信号
        strategy_signals = strategy.generate_signals(data)
        
        # 提取核心组件
        components = strategy.extract_signal_components(strategy_signals)
        
        # 获取信号元数据
        metadata = strategy.get_signal_metadata()
        
        # 标准化信号
        prefix = prefix or strategy.name + "_"
        normalized_signals = pd.DataFrame(index=data.index)
        
        # 处理策略信号
        normalized_signals[f"{prefix}signal"] = strategy_signals["signal"]
        
        # 处理核心组件
        for comp_name, comp_data in components.items():
            if not isinstance(comp_data, pd.Series):
                continue
                
            # 跳过已经处理过的信号列
            if comp_name == "signal":
                continue
                
            # 归一化处理
            if comp_name in ["rsi"]:
                # RSI类指标通常在0-100之间，归一化到0-1
                normalized_signals[f"{prefix}{comp_name}"] = comp_data / 100.0 if comp_data.max() > 1 else comp_data
            elif comp_name in ["macd", "histogram"]:
                # MACD类指标需要相对于价格进行缩放
                normalized_signals[f"{prefix}{comp_name}"] = comp_data / data["close"].mean()
            else:
                # 其他类型的组件，直接添加
                normalized_signals[f"{prefix}{comp_name}"] = comp_data
        
        # 保存信号元数据
        for col in normalized_signals.columns:
            signal_name = col.replace(prefix, "")
            
            if signal_name == "signal":
                self.signal_metadata[col] = {
                    "source": strategy.name,
                    "category": "trading_signal",
                    "description": f"{strategy.name} 策略生成的交易信号",
                    "importance": "high",
                    "params": strategy.parameters
                }
            elif signal_name in metadata:
                self.signal_metadata[col] = metadata[signal_name]
                self.signal_metadata[col]["source"] = strategy.name
        
        # 保存策略信号
        self.strategy_signals[strategy.name] = normalized_signals
        
        return normalized_signals
    
    def extract_signals_from_strategies(self, 
                                       strategies: List[Strategy], 
                                       data: pd.DataFrame) -> pd.DataFrame:
        """
        从多个策略中批量提取信号
        
        该方法循环处理多个策略，从每个策略中提取信号，并将所有信号合并到一个DataFrame中。
        
        参数:
            strategies (List[Strategy]): 策略实例列表，每个实例必须实现Strategy基类接口
            data (pd.DataFrame): 包含OHLCV数据的DataFrame，用于生成信号
            
        返回:
            pd.DataFrame: 包含所有策略标准化信号的DataFrame，索引与输入数据相同
            
        注意:
            此方法会在内部调用extract_signals_from_strategy方法处理每个策略
        """
        all_signals = pd.DataFrame(index=data.index)
        
        for strategy in strategies:
            signals = self.extract_signals_from_strategy(strategy, data)
            all_signals = pd.concat([all_signals, signals], axis=1)
        
        return all_signals
    
    def rank_signals_by_importance(self) -> Dict[str, float]:
        """
        根据元数据中的重要性对信号进行排序
        
        该方法基于每个信号在元数据中定义的重要性级别（high/medium/low）计算重要性分数，
        并返回按重要性降序排列的信号列表。
        
        重要性映射:
        - high: 3.0
        - medium: 2.0
        - low: 1.0
        
        返回:
            Dict[str, float]: 字典，键为信号名称，值为重要性分数，按重要性降序排序
            
        注意:
            如果某信号在元数据中未指定重要性，则默认为"medium"（2.0）
        """
        importance_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
        importance_scores = {}
        
        for signal, metadata in self.signal_metadata.items():
            importance = metadata.get("importance", "medium")
            importance_scores[signal] = importance_map.get(importance, 1.0)
        
        # 按重要性降序排序
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def get_metadata(self, signal_name: Optional[str] = None) -> Dict:
        """
        获取信号元数据
        
        参数:
            signal_name (Optional[str]): 信号名称，为None时返回所有元数据
            
        返回:
            Dict: 信号元数据字典，包含信号的详细信息（来源、类别、描述、重要性等）
            
        示例:
            >>> extractor.get_metadata("rsi_strategy_rsi")
            {
                "source": "RSIStrategy",
                "category": "oscillator",
                "description": "相对强弱指标",
                "importance": "high",
                "params": {"period": 14}
            }
        """
        if signal_name:
            return self.signal_metadata.get(signal_name, {})
        return self.signal_metadata