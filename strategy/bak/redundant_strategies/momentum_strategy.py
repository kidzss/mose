import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .strategy_base import Strategy

class MomentumStrategy(Strategy):
    """
    短期动量策略
    
    策略说明:
    1. 使用MACD捕捉短期趋势
    2. 使用ADX确认趋势强度
    3. 使用成交量确认趋势
    4. 使用价格动量指标（ROC）确认趋势
    
    买入条件:
    - MACD金叉
    - ADX显示强趋势
    - 成交量放大
    - ROC为正且增加
    
    卖出条件:
    - MACD死叉
    - ADX显示趋势减弱
    - 成交量萎缩
    - ROC为负且减少
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化动量策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - macd_fast: MACD快线周期，默认12
                - macd_slow: MACD慢线周期，默认26
                - macd_signal: MACD信号线周期，默认9
                - adx_length: ADX周期，默认14
                - adx_threshold: ADX趋势强度阈值，默认25
                - volume_ma_length: 成交量均线周期，默认20
                - volume_threshold: 成交量放大阈值，默认1.5
                - roc_length: ROC周期，默认10
                - roc_ma_length: ROC均线周期，默认5
        """
        default_params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_length': 14,
            'adx_threshold': 25,
            'volume_ma_length': 20,
            'volume_threshold': 1.5,
            'roc_length': 10,
            'roc_ma_length': 5
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MomentumStrategy', default_params)
        self.logger.info(f"初始化动量策略，参数: {default_params}")
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
            
            # 计算MACD
            exp1 = df['close'].ewm(span=self.parameters['macd_fast'], adjust=False).mean()
            exp2 = df['close'].ewm(span=self.parameters['macd_slow'], adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=self.parameters['macd_signal'], adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal_line']
            
            # 计算ADX
            tr1 = abs(df['high'] - df['low'])
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_dm = df['high'] - df['high'].shift(1)
            minus_dm = df['low'].shift(1) - df['low']
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
            
            tr_ma = df['tr'].rolling(window=self.parameters['adx_length']).mean()
            plus_di = 100 * plus_dm.rolling(window=self.parameters['adx_length']).mean() / tr_ma
            minus_di = 100 * minus_dm.rolling(window=self.parameters['adx_length']).mean() / tr_ma
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=self.parameters['adx_length']).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # 计算成交量指标
            df['volume_ma'] = df['volume'].rolling(window=self.parameters['volume_ma_length']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 计算ROC
            df['roc'] = (df['close'] - df['close'].shift(self.parameters['roc_length'])) / df['close'].shift(self.parameters['roc_length']) * 100
            df['roc_ma'] = df['roc'].rolling(window=self.parameters['roc_ma_length']).mean()
            
            # 生成信号条件
            df['macd_cross_up'] = (df['macd'] > df['signal_line']) & (df['macd'].shift(1) <= df['signal_line'].shift(1))
            df['macd_cross_down'] = (df['macd'] < df['signal_line']) & (df['macd'].shift(1) >= df['signal_line'].shift(1))
            
            df['strong_trend'] = df['adx'] > self.parameters['adx_threshold']
            df['volume_confirm'] = df['volume_ratio'] > self.parameters['volume_threshold']
            
            df['roc_rising'] = (df['roc'] > df['roc_ma']) & (df['roc'] > 0)
            df['roc_falling'] = (df['roc'] < df['roc_ma']) & (df['roc'] < 0)
            
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
                (df['macd_cross_up']) &  # MACD金叉
                (df['strong_trend']) &  # ADX显示强趋势
                (df['volume_confirm']) &  # 成交量确认
                (df['roc_rising'])  # ROC上升趋势
            )
            
            # 卖出条件
            sell_condition = (
                (df['macd_cross_down']) &  # MACD死叉
                (df['strong_trend']) &  # ADX显示强趋势
                (df['volume_confirm']) &  # 成交量确认
                (df['roc_falling'])  # ROC下降趋势
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
        提取信号组件
        """
        # 计算指标
        df = self.calculate_indicators(data)
        
        # 生成信号
        df = self.generate_signals(df)
        
        # MACD组件，标准化到[-1, 1]
        macd_max = max(abs(df['macd'].max()), abs(df['macd'].min()))
        macd_norm = df['macd'] / macd_max if macd_max != 0 else df['macd']
        
        # ADX组件，标准化到[0, 1]
        adx_norm = df['adx'] / 100
        
        # 成交量组件
        volume_norm = (df['volume'] / df['volume_ma'] - 1).clip(-1, 1)
        
        # ROC组件，标准化到[-1, 1]
        roc_max = max(abs(df['roc'].max()), abs(df['roc'].min()))
        roc_norm = df['roc'] / roc_max if roc_max != 0 else df['roc']
        
        return {
            "macd": macd_norm,
            "adx": adx_norm,
            "volume": volume_norm,
            "roc": roc_norm,
            "composite": df['signal']
        }
        
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        """
        return {
            "macd": {
                "type": "trend",
                "time_scale": "short",
                "description": "MACD指标（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "adx": {
                "type": "strength",
                "time_scale": "short",
                "description": "ADX趋势强度指标（标准化到[0, 1]）",
                "min_value": 0,
                "max_value": 1
            },
            "volume": {
                "type": "volume",
                "time_scale": "short",
                "description": "成交量确认指标",
                "min_value": -1,
                "max_value": 1
            },
            "roc": {
                "type": "momentum",
                "time_scale": "short",
                "description": "ROC动量指标（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "short",
                "description": "动量策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 