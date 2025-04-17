import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .strategy_base import Strategy

class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    
    策略说明:
    1. 使用多个技术指标识别超买超卖区域
    2. 结合价格行为和成交量确认反转信号
    3. 使用动态阈值适应不同的市场环境
    
    买入条件:
    - RSI和随机指标同时处于超卖区域
    - 价格出现反转形态（如十字星、锤子线等）
    - 成交量放大确认反转
    
    卖出条件:
    - RSI和随机指标同时处于超买区域
    - 价格出现反转形态
    - 成交量放大确认反转
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化均值回归策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - rsi_length: RSI周期，默认14
                - stoch_length: 随机指标周期，默认14
                - stoch_smooth: 随机指标平滑周期，默认3
                - volume_ma_length: 成交量均线周期，默认20
                - rsi_oversold: RSI超卖阈值，默认30
                - rsi_overbought: RSI超买阈值，默认70
                - stoch_oversold: 随机指标超卖阈值，默认20
                - stoch_overbought: 随机指标超买阈值，默认80
                - volume_threshold: 成交量放大阈值，默认1.5
        """
        default_params = {
            'rsi_length': 14,
            'stoch_length': 14,
            'stoch_smooth': 3,
            'volume_ma_length': 20,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'volume_threshold': 1.5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MeanReversionStrategy', default_params)
        self.logger.info(f"初始化均值回归策略，参数: {default_params}")
        self.version = '1.0.0'
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        """
        try:
            if data is None or data.empty:
                self.logger.warning("数据为空，无法计算指标")
                return pd.DataFrame()
            
            df = data.copy()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.parameters['rsi_length']).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.parameters['rsi_length']).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算随机指标
            low_min = df['low'].rolling(window=self.parameters['stoch_length']).min()
            high_max = df['high'].rolling(window=self.parameters['stoch_length']).max()
            k = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_k'] = k.rolling(window=self.parameters['stoch_smooth']).mean()
            df['stoch_d'] = df['stoch_k'].rolling(window=self.parameters['stoch_smooth']).mean()
            
            # 计算成交量指标
            df['volume_ma'] = df['volume'].rolling(window=self.parameters['volume_ma_length']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 计算价格形态
            df['body'] = df['close'] - df['open']
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['body_ratio'] = abs(df['body']) / (df['high'] - df['low'])
            
            # 识别反转形态
            df['doji'] = (abs(df['body']) <= (df['high'] - df['low']) * 0.1)  # 十字星
            df['hammer'] = ((df['lower_shadow'] > abs(df['body']) * 2) & (df['upper_shadow'] <= abs(df['body']) * 0.5))  # 锤子线
            df['shooting_star'] = ((df['upper_shadow'] > abs(df['body']) * 2) & (df['lower_shadow'] <= abs(df['body']) * 0.5))  # 流星线
            
            # 超买超卖状态
            df['rsi_oversold'] = df['rsi'] < self.parameters['rsi_oversold']
            df['rsi_overbought'] = df['rsi'] > self.parameters['rsi_overbought']
            df['stoch_oversold'] = df['stoch_k'] < self.parameters['stoch_oversold']
            df['stoch_overbought'] = df['stoch_k'] > self.parameters['stoch_overbought']
            
            # 成交量确认
            df['volume_confirm'] = df['volume_ratio'] > self.parameters['volume_threshold']
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
            
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        try:
            df = self.calculate_indicators(data)
            
            # 初始化信号列
            df['signal'] = 0
            
            # 买入条件
            buy_condition = (
                (df['rsi_oversold']) &  # RSI超卖
                (df['stoch_oversold']) &  # 随机指标超卖
                ((df['doji']) | (df['hammer'])) &  # 出现反转形态
                (df['volume_confirm'])  # 成交量确认
            )
            
            # 卖出条件
            sell_condition = (
                (df['rsi_overbought']) &  # RSI超买
                (df['stoch_overbought']) &  # 随机指标超买
                ((df['doji']) | (df['shooting_star'])) &  # 出现反转形态
                (df['volume_confirm'])  # 成交量确认
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
        """
        df = self.calculate_indicators(data)
        
        # RSI组件，标准化到[-1, 1]
        rsi_norm = (df['rsi'] - 50) / 50
        
        # 随机指标组件，标准化到[-1, 1]
        stoch_norm = (df['stoch_k'] - 50) / 50
        
        # 价格形态组件
        reversal_pattern = (
            df['doji'].astype(float) +
            df['hammer'].astype(float) -
            df['shooting_star'].astype(float)
        )
        
        # 成交量组件，标准化到[-1, 1]
        volume_norm = (df['volume_ratio'] - 1) / self.parameters['volume_threshold']
        
        return {
            "rsi": rsi_norm,
            "stochastic": stoch_norm,
            "reversal_pattern": reversal_pattern,
            "volume": volume_norm,
            "composite": df['signal']
        }
        
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        """
        return {
            "rsi": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI指标（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "stochastic": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "随机指标（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "reversal_pattern": {
                "type": "pattern",
                "time_scale": "short",
                "description": "价格反转形态",
                "min_value": -1,
                "max_value": 1
            },
            "volume": {
                "type": "volume",
                "time_scale": "short",
                "description": "成交量确认指标",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "short",
                "description": "均值回归策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 