import pandas as pd
import numpy as np
from strategy.strategy_base import Strategy
import logging

class BreakoutStrategy(Strategy):
    """
    突破策略
    使用价格突破和成交量确认
    """
    def __init__(self, 
                 lookback_period=20,    # 回看周期
                 breakout_threshold=0.02, # 突破阈值
                 volume_threshold=1.5,   # 成交量阈值
                 confirmation_period=3): # 确认周期
        super().__init__(name="Breakout")
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_threshold = volume_threshold
        self.confirmation_period = confirmation_period
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, data):
        """
        计算技术指标
        """
        try:
            # 计算最高价和最低价
            data['high_20'] = data['high'].rolling(window=self.lookback_period).max()
            data['low_20'] = data['low'].rolling(window=self.lookback_period).min()
            
            # 计算ATR
            data['tr'] = pd.DataFrame({
                'hl': data['high'] - data['low'],
                'hc': abs(data['high'] - data['close'].shift()),
                'lc': abs(data['low'] - data['close'].shift())
            }).max(axis=1)
            data['atr'] = data['tr'].rolling(window=self.lookback_period).mean()
            
            # 计算成交量移动平均
            data['volume_ma'] = data['volume'].rolling(window=self.lookback_period).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # 计算突破信号
            data['high_break'] = (data['high'] > data['high_20'].shift(1) * (1 + self.breakout_threshold)).astype(int)
            data['low_break'] = (data['low'] < data['low_20'].shift(1) * (1 - self.breakout_threshold)).astype(int)
            
            # 计算趋势强度
            data['trend_strength'] = (data['close'] - data['close'].rolling(window=self.lookback_period).mean()) / data['atr']
            
            return data
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {str(e)}")
            return data
            
    def generate_signals(self, data):
        """
        生成交易信号
        """
        try:
            signals = pd.Series(0, index=data.index)
            
            # 突破信号
            breakout_signal = np.where(
                (data['high_break'] == 1) & 
                (data['volume_ratio'] > self.volume_threshold) & 
                (data['trend_strength'] > 0),
                1,
                np.where(
                    (data['low_break'] == 1) & 
                    (data['volume_ratio'] > self.volume_threshold) & 
                    (data['trend_strength'] < 0),
                    -1,
                    0
                )
            )
            
            # 确认信号
            confirmation = pd.Series(breakout_signal).rolling(window=self.confirmation_period).mean()
            
            # 最终信号
            signals = np.where(
                confirmation > 0.5,
                1,
                np.where(
                    confirmation < -0.5,
                    -1,
                    0
                )
            )
            
            return pd.Series(signals, index=data.index)
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            return pd.Series(0, index=data.index)
            
    def extract_signal_components(self, data):
        """
        提取信号组件
        """
        components = {
            'breakout_signal': data['high_break'] - data['low_break'],
            'volume_ratio': data['volume_ratio'],
            'trend_strength': data['trend_strength'],
            'atr': data['atr']
        }
        return components
        
    def get_signal_metadata(self):
        """
        获取信号元数据
        """
        return {
            'signal_type': 'breakout',
            'time_scale': '5min',
            'description': 'Breakout strategy with volume confirmation'
        } 