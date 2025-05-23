import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import talib
from datetime import datetime, timedelta

class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict:
        """
        计算布林带
        :param df: 股票数据
        :param period: 周期
        :param std_dev: 标准差倍数
        :return: 布林带数据
        """
        try:
            close = df['close'].values
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            
            return {
                'upper': upper[-1],
                'middle': middle[-1],
                'lower': lower[-1],
                'bandwidth': (upper[-1] - lower[-1]) / middle[-1],
                'position': (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
            }
            
        except Exception as e:
            self.logger.error(f"计算布林带失败: {e}")
            return {}
            
    def calculate_kdj(self, df: pd.DataFrame, k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict:
        """
        计算KDJ指标
        :param df: 股票数据
        :param k_period: K值周期
        :param d_period: D值周期
        :param j_period: J值周期
        :return: KDJ数据
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # 计算RSV
            lowest_low = talib.MIN(low, k_period)
            highest_high = talib.MAX(high, k_period)
            rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
            
            # 计算K值
            k = talib.EMA(rsv, d_period)
            
            # 计算D值
            d = talib.EMA(k, d_period)
            
            # 计算J值
            j = 3 * k - 2 * d
            
            return {
                'k': k[-1],
                'd': d[-1],
                'j': j[-1],
                'signal': self._get_kdj_signal(k[-1], d[-1], j[-1])
            }
            
        except Exception as e:
            self.logger.error(f"计算KDJ失败: {e}")
            return {}
            
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算成交量指标
        :param df: 股票数据
        :return: 成交量指标数据
        """
        try:
            # 确保数据类型是 double
            close = df['close'].astype(float).values
            volume = df['volume'].astype(float).values
            
            # 计算OBV
            obv = talib.OBV(close, volume)
            
            # 计算VWAP
            vwap = (df['close'].astype(float) * df['volume'].astype(float)).cumsum() / df['volume'].astype(float).cumsum()
            
            # 计算成交量移动平均
            volume_ma5 = talib.MA(volume, timeperiod=5)
            volume_ma20 = talib.MA(volume, timeperiod=20)
            
            return {
                'obv': float(obv[-1]),
                'vwap': float(vwap.iloc[-1]),
                'volume_ma5': float(volume_ma5[-1]),
                'volume_ma20': float(volume_ma20[-1]),
                'volume_ratio': float(volume[-1] / volume_ma20[-1])
            }
            
        except Exception as e:
            self.logger.error(f"计算成交量指标失败: {e}")
            return {}
            
    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback_period: int = 60) -> Dict:
        """
        计算斐波那契回调位
        :param df: 股票数据
        :param lookback_period: 回溯周期
        :return: 斐波那契回调位数据
        """
        try:
            if len(df) < lookback_period:
                return {}
                
            # 获取最高价和最低价
            high = df['high'].max()
            low = df['low'].min()
            range_size = high - low
            
            # 计算斐波那契回调位
            levels = {
                '0.0': low,
                '0.236': low + range_size * 0.236,
                '0.382': low + range_size * 0.382,
                '0.5': low + range_size * 0.5,
                '0.618': low + range_size * 0.618,
                '0.786': low + range_size * 0.786,
                '1.0': high
            }
            
            # 计算当前价格所在位置
            current_price = df['close'].iloc[-1]
            position = (current_price - low) / range_size
            
            return {
                'levels': levels,
                'current_position': position,
                'nearest_level': self._find_nearest_level(current_price, levels)
            }
            
        except Exception as e:
            self.logger.error(f"计算斐波那契回调位失败: {e}")
            return {}
            
    def _get_kdj_signal(self, k: float, d: float, j: float) -> str:
        """
        获取KDJ信号
        """
        if k < 20 and d < 20 and j < 20:
            return 'oversold'
        elif k > 80 and d > 80 and j > 80:
            return 'overbought'
        elif k > d and j > k:
            return 'bullish'
        elif k < d and j < k:
            return 'bearish'
        else:
            return 'neutral'
            
    def _find_nearest_level(self, price: float, levels: Dict) -> Tuple[str, float]:
        """
        找到最近的斐波那契回调位
        """
        nearest_level = None
        min_distance = float('inf')
        
        for level, value in levels.items():
            distance = abs(price - value)
            if distance < min_distance:
                min_distance = distance
                nearest_level = (level, value)
                
        return nearest_level 