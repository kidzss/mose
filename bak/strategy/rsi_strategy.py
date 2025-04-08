import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy

class RSIStrategy(Strategy):
    """
    RSI（相对强弱指数）交易策略
    
    策略说明:
    1. 买入条件: 
       - RSI从超卖区域（默认30以下）回升
       - 如果ma_filter=True，则还需要价格大于移动平均线（均线过滤）
       
    2. 卖出条件:
       - RSI从超买区域（默认70以上）回落
       - 如果ma_filter=True，则还需要价格小于移动平均线（均线过滤）
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化RSI策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - rsi_period: RSI计算周期，默认14
                - ma_period: 移动平均线周期，默认50
                - overbought: 超买阈值，默认70
                - oversold: 超卖阈值，默认30
                - rsi_reversal_threshold: RSI反转确认阈值，默认1.0
                - ma_filter: 是否使用均线过滤，默认True
        """
        default_params = {
            'rsi_period': 14,
            'ma_period': 50,
            'overbought': 70,
            'oversold': 30,
            'rsi_reversal_threshold': 1.0,
            'ma_filter': True
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('RSIStrategy', default_params)
        self.logger.info(f"初始化RSI策略，参数: {default_params}")
        self.version = '1.0.0'
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        try:
            if data is None or data.empty:
                self.logger.warning("数据为空，无法计算指标")
                return pd.DataFrame()
            
            # 复制数据以避免修改原始数据
            df = data.copy()
            
            # 获取参数
            rsi_period = self.parameters['rsi_period']
            ma_period = self.parameters['ma_period']
            overbought = self.parameters['overbought']
            oversold = self.parameters['oversold']
            rsi_reversal_threshold = self.parameters['rsi_reversal_threshold']
            
            # 计算移动平均线
            df['ma'] = df['close'].rolling(window=ma_period).mean()
            
            # 计算价格相对于均线的位置
            df['price_above_ma'] = df['close'] > df['ma']
            df['price_below_ma'] = df['close'] < df['ma']
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算RSI超买超卖区域
            df['overbought'] = df['rsi'] > overbought
            df['oversold'] = df['rsi'] < oversold
            
            # 计算RSI从超买区域回落和从超卖区域回升的条件
            df['rsi_was_overbought'] = df['overbought'].shift(1)
            df['rsi_was_oversold'] = df['oversold'].shift(1)
            
            # RSI方向
            df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1) - rsi_reversal_threshold
            df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1) + rsi_reversal_threshold
            
            # 计算RSI从超买区域回落和从超卖区域回升的信号
            df['oversold_reversal'] = df['rsi_was_oversold'] & df['rsi_rising']
            df['overbought_reversal'] = df['rsi_was_overbought'] & df['rsi_falling']
            
            # 填充NaN值
            df = df.bfill().ffill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame，其中:
            1 = 买入信号
            0 = 持有/无信号
            -1 = 卖出信号
        """
        try:
            # 计算技术指标
            df = self.calculate_indicators(data)
            
            # 初始化信号列
            df['signal'] = 0
            
            # 获取参数
            ma_filter = self.parameters['ma_filter']
            
            # 买入条件
            if ma_filter:
                # 带均线过滤的买入条件
                buy_condition = df['oversold_reversal'] & df['price_above_ma']
            else:
                # 不带均线过滤的买入条件
                buy_condition = df['oversold_reversal']
            
            # 卖出条件
            if ma_filter:
                # 带均线过滤的卖出条件
                sell_condition = df['overbought_reversal'] & df['price_below_ma']
            else:
                # 不带均线过滤的卖出条件
                sell_condition = df['overbought_reversal']
            
            # 生成信号
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data.assign(signal=0)
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        df = self.calculate_indicators(data)
        
        # RSI值，标准化到[-1, 1]区间
        # 0表示50值，-1表示0值，1表示100值
        rsi_norm = (df['rsi'] - 50) / 50
        
        # 超买超卖状态
        overbought = (df['rsi'] - self.parameters['overbought']) / (100 - self.parameters['overbought'])
        overbought = overbought.clip(0, 1)
        
        oversold = (self.parameters['oversold'] - df['rsi']) / self.parameters['oversold']
        oversold = oversold.clip(0, 1)
        
        # RSI信号反转
        oversold_reversal = df['oversold_reversal'].astype(float)
        overbought_reversal = -df['overbought_reversal'].astype(float)
        
        # RSI方向
        rsi_direction = df['rsi_rising'].astype(float) - df['rsi_falling'].astype(float)
        
        # 价格相对于均线位置
        price_ma_relation = df['price_above_ma'].astype(float) - df['price_below_ma'].astype(float)
        
        return {
            "rsi": rsi_norm,
            "overbought": overbought,
            "oversold": oversold,
            "rsi_reversal": oversold_reversal + overbought_reversal,
            "rsi_direction": rsi_direction,
            "price_ma_relation": price_ma_relation,
            "composite": df['signal']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "rsi": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI值（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "overbought": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI超买程度",
                "min_value": 0,
                "max_value": 1
            },
            "oversold": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI超卖程度",
                "min_value": 0,
                "max_value": 1
            },
            "rsi_reversal": {
                "type": "signal",
                "time_scale": "short",
                "description": "RSI反转信号",
                "min_value": -1,
                "max_value": 1
            },
            "rsi_direction": {
                "type": "momentum",
                "time_scale": "short",
                "description": "RSI方向",
                "min_value": -1,
                "max_value": 1
            },
            "price_ma_relation": {
                "type": "trend",
                "time_scale": "medium",
                "description": "价格相对于均线的位置",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "short",
                "description": "RSI策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 