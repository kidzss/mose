import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy

class BollingerBandsStrategy(Strategy):
    """
    布林带交易策略
    
    策略说明:
    1. 买入条件: 
       - 价格触及或跌破下轨后反弹
       - RSI值小于超卖阈值后开始回升
       
    2. 卖出条件:
       - 价格触及或突破上轨后回落
       - RSI值大于超买阈值后开始下降
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化布林带策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - bb_length: 布林带周期，默认20
                - bb_std: 布林带标准差倍数，默认2.0
                - rsi_length: RSI周期，默认14
                - rsi_overbought: RSI超买阈值，默认70
                - rsi_oversold: RSI超卖阈值，默认30
                - price_pct_trigger: 价格反转触发百分比，默认0.01 (1%)
        """
        default_params = {
            'bb_length': 20,
            'bb_std': 2.0,
            'rsi_length': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'price_pct_trigger': 0.01
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('BollingerBandsStrategy', default_params)
        self.logger.info(f"初始化布林带策略，参数: {default_params}")
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
            
            # 计算布林带
            bb_length = self.parameters['bb_length']
            bb_std = self.parameters['bb_std']
            
            df['bb_middle'] = df['close'].rolling(window=bb_length).mean()
            price_std = df['close'].rolling(window=bb_length).std()
            df['bb_upper'] = df['bb_middle'] + bb_std * price_std
            df['bb_lower'] = df['bb_middle'] - bb_std * price_std
            
            # 计算RSI
            rsi_length = self.parameters['rsi_length']
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_length).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_length).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算价格位置（布林带百分比）
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 价格触及/突破布林带的条件
            df['price_below_lower'] = df['close'] <= df['bb_lower']
            df['price_above_upper'] = df['close'] >= df['bb_upper']
            
            # 计算价格反转信号
            price_pct = self.parameters['price_pct_trigger']
            df['price_reversal_up'] = df['close'] > df['close'].shift(1) * (1 + price_pct)
            df['price_reversal_down'] = df['close'] < df['close'].shift(1) * (1 - price_pct)
            
            # RSI超买超卖状态
            df['rsi_oversold'] = df['rsi'] < self.parameters['rsi_oversold']
            df['rsi_overbought'] = df['rsi'] > self.parameters['rsi_overbought']
            
            # RSI反转信号
            df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
            df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1)
            
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
            
            # 买入条件：价格触及或跌破下轨后反弹 + RSI超卖后回升
            buy_condition = (
                (df['price_below_lower'].shift(1)) &  # 之前价格触及或跌破下轨
                (df['price_reversal_up']) &  # 价格出现反弹
                (df['rsi_oversold'].shift(1)) &  # 之前RSI处于超卖状态
                (df['rsi_rising'])  # RSI开始回升
            )
            
            # 卖出条件：价格触及或突破上轨后回落 + RSI超买后下降
            sell_condition = (
                (df['price_above_upper'].shift(1)) &  # 之前价格触及或突破上轨
                (df['price_reversal_down']) &  # 价格出现回落
                (df['rsi_overbought'].shift(1)) &  # 之前RSI处于超买状态
                (df['rsi_falling'])  # RSI开始下降
            )
            
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
        
        # 布林带位置组件，标准化到[-1, 1]
        # 0.5表示在中轨，0表示在下轨，1表示在上轨
        bb_position = (df['bb_pct'] - 0.5) * 2
        
        # 价格反转组件
        reversal_up = df['price_reversal_up'].astype(float)
        reversal_down = -df['price_reversal_down'].astype(float)
        price_reversal = reversal_up + reversal_down
        
        # RSI组件，标准化到[-1, 1]
        # 0表示50值，-1表示超卖区域，1表示超买区域
        rsi_norm = (df['rsi'] - 50) / 50
        
        # RSI反转组件
        rsi_direction = df['rsi_rising'].astype(float) - df['rsi_falling'].astype(float)
        
        return {
            "bb_position": bb_position,
            "price_reversal": price_reversal,
            "rsi": rsi_norm,
            "rsi_direction": rsi_direction,
            "composite": df['signal']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "bb_position": {
                "type": "mean_reversion",
                "time_scale": "short",
                "description": "价格在布林带中的位置",
                "min_value": -1,
                "max_value": 1
            },
            "price_reversal": {
                "type": "momentum",
                "time_scale": "short",
                "description": "价格反转信号",
                "min_value": -1,
                "max_value": 1
            },
            "rsi": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI值（标准化到[-1, 1]）",
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
            "composite": {
                "type": "composite",
                "time_scale": "short",
                "description": "布林带策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 