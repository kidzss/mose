import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

from .strategy_base import Strategy

class MomentumStrategy(Strategy):
    """
    Momentum策略 - 动量策略
    
    Momentum策略基于价格动量，通过跟踪价格与其滞后周期的差值以及动量的变化
    来生成交易信号。
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化Momentum策略
        
        参数:
            parameters: 策略参数字典
        """
        # 设置默认参数
        default_params = {
            'length': 10,  # 动量计算周期
            'threshold': 0,  # 动量阈值
            'ma_length': 5  # 动量移动平均周期
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
        
        # 初始化基类
        super().__init__('MomentumStrategy', default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        length = self.parameters['length']
        ma_length = self.parameters['ma_length']
        
        # 计算动量 (当前价格与n周期前价格的差)
        df['momentum'] = df['close'] - df['close'].shift(length)
        
        # 计算动量的移动平均
        df['momentum_ma'] = df['momentum'].rolling(window=ma_length).mean()
        
        # 计算动量指标的变化
        df['momentum_change'] = df['momentum'] - df['momentum'].shift(1)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame
        """
        # 计算技术指标
        df = self.calculate_indicators(data)
        threshold = self.parameters['threshold']
        
        # 初始化信号列
        df['signal'] = 0
        
        # 计算Momentum买入/卖出条件
        # 买入条件：动量大于阈值且动量变化为正
        buy_condition = (df['momentum'] > threshold) & (df['momentum_change'] > 0)
        # 卖出条件：动量小于负阈值且动量变化为负
        sell_condition = (df['momentum'] < -threshold) & (df['momentum_change'] < 0)
        
        # 生成信号
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # 填充NaN值
        df = df.fillna(0)
        
        return df
        
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        result = self.calculate_indicators(data)
        
        # 提取关键组件
        components = {
            'momentum': result.get('momentum', pd.Series()),
            'momentum_ma': result.get('momentum_ma', pd.Series()),
            'momentum_change': result.get('momentum_change', pd.Series()),
            'price': result.get('close', pd.Series())
        }
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            'momentum': {
                'name': '动量',
                'description': f"当前价格与{self.parameters['length']}周期前价格的差值",
                'color': 'blue',
                'line_style': 'solid',
                'importance': 'high'
            },
            'momentum_ma': {
                'name': '动量平均线',
                'description': f"动量的{self.parameters['ma_length']}周期移动平均",
                'color': 'red',
                'line_style': 'solid',
                'importance': 'medium'
            },
            'momentum_change': {
                'name': '动量变化',
                'description': '动量的一阶导数，衡量动量变化速度',
                'color': 'green',
                'line_style': 'dotted',
                'importance': 'high'
            },
            'price': {
                'name': '价格',
                'description': '资产收盘价',
                'color': 'black',
                'line_style': 'solid',
                'importance': 'high'
            }
        } 