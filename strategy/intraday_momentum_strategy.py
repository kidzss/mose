import pandas as pd
import numpy as np
from strategy.strategy_base import Strategy
import logging

class IntradayMomentumStrategy(Strategy):
    """
    日内动量策略
    使用5分钟和15分钟数据计算动量
    """
    def __init__(self, 
                 short_window=12,    # 5分钟数据，12个周期=1小时
                 long_window=48,     # 5分钟数据，48个周期=4小时
                 momentum_window=6,  # 动量计算窗口
                 volume_window=12,   # 成交量计算窗口
                 threshold=0.001):   # 动量阈值
        super().__init__(name="IntradayMomentum")
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_window = momentum_window
        self.volume_window = volume_window
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, data):
        """
        计算技术指标
        """
        try:
            # 计算短期和长期移动平均线
            data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
            data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
            
            # 计算动量
            data['momentum'] = data['close'].pct_change(periods=self.momentum_window)
            
            # 计算成交量趋势
            data['volume_ma'] = data['volume'].rolling(window=self.volume_window).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # 计算波动率
            data['volatility'] = data['close'].rolling(window=self.short_window).std() / data['close'].rolling(window=self.short_window).mean()
            
            # 计算价格突破
            data['high_break'] = (data['high'] > data['high'].rolling(window=self.short_window).max().shift(1)).astype(int)
            data['low_break'] = (data['low'] < data['low'].rolling(window=self.short_window).min().shift(1)).astype(int)
            
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
            
            # 动量信号
            momentum_signal = np.where(
                (data['momentum'] > self.threshold) & 
                (data['volume_ratio'] > 1.2) & 
                (data['volatility'] < 0.02),
                1,
                np.where(
                    (data['momentum'] < -self.threshold) & 
                    (data['volume_ratio'] > 1.2) & 
                    (data['volatility'] < 0.02),
                    -1,
                    0
                )
            )
            
            # 突破信号
            break_signal = np.where(
                data['high_break'] == 1,
                1,
                np.where(
                    data['low_break'] == 1,
                    -1,
                    0
                )
            )
            
            # 趋势确认
            trend_signal = np.where(
                (data['short_ma'] > data['long_ma']) & 
                (data['momentum'] > 0),
                1,
                np.where(
                    (data['short_ma'] < data['long_ma']) & 
                    (data['momentum'] < 0),
                    -1,
                    0
                )
            )
            
            # 综合信号
            signals = np.where(
                (momentum_signal == 1) & (break_signal == 1) & (trend_signal == 1),
                1,
                np.where(
                    (momentum_signal == -1) & (break_signal == -1) & (trend_signal == -1),
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
            'momentum': data['momentum'],
            'volume_ratio': data['volume_ratio'],
            'volatility': data['volatility'],
            'trend': (data['short_ma'] - data['long_ma']) / data['long_ma'],
            'break_signal': data['high_break'] - data['low_break']
        }
        return components
        
    def get_signal_metadata(self):
        """
        获取信号元数据
        """
        return {
            'signal_type': 'intraday_momentum',
            'time_scale': '5min',
            'description': 'Intraday momentum strategy using 5min and 15min data'
        } 