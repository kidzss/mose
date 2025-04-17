import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

from .strategy_base import Strategy

class CustomStrategy(Strategy):
    """
    自定义策略示例
    
    这个策略结合了RSI和移动平均线来生成交易信号。
    它作为用户创建自定义策略的模板。
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化自定义策略
        
        参数:
            parameters: 策略参数字典
        """
        # 设置默认参数
        default_params = {
            'ma_length': 20,     # 移动平均线周期
            'rsi_length': 14,    # RSI计算周期
            'rsi_overbought': 70,  # RSI超买阈值
            'rsi_oversold': 30     # RSI超卖阈值
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
        
        # 初始化基类
        super().__init__('CustomStrategy', default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        ma_length = self.parameters['ma_length']
        rsi_length = self.parameters['rsi_length']
        
        # 计算移动平均线
        df['ma'] = df['close'].rolling(window=ma_length).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
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
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        
        # 初始化信号列
        df['signal'] = 0
        
        # 买入条件: RSI从超卖区域上升且价格在均线上方
        buy_condition = (df['rsi'] > rsi_oversold) & \
                        (df['rsi'].shift(1) <= rsi_oversold) & \
                        (df['close'] > df['ma'])
        
        # 卖出条件: RSI从超买区域下降且价格在均线下方
        sell_condition = (df['rsi'] < rsi_overbought) & \
                         (df['rsi'].shift(1) >= rsi_overbought) & \
                         (df['close'] < df['ma'])
        
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
            'ma': result.get('ma', pd.Series()),
            'rsi': result.get('rsi', pd.Series()),
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
            'ma': {
                'name': '移动平均线',
                'description': f"{self.parameters['ma_length']}周期移动平均",
                'color': 'blue',
                'line_style': 'solid',
                'importance': 'high'
            },
            'rsi': {
                'name': 'RSI',
                'description': f"{self.parameters['rsi_length']}周期RSI指标",
                'color': 'red',
                'line_style': 'solid',
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