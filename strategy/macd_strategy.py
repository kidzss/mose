import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .strategy_base import Strategy

class MACDStrategy(Strategy):
    """
    MACD交易策略
    
    策略说明:
    1. 买入条件: 
       - MACD线上穿信号线（金叉）
       - 如果trend_filter=True，则还需要MACD值大于0（多头趋势）
       
    2. 卖出条件:
       - MACD线下穿信号线（死叉）
       - 如果trend_filter=True，则还需要MACD值小于0（空头趋势）
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化MACD策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - fast_period: 快线周期，默认12
                - slow_period: 慢线周期，默认26
                - signal_period: 信号线周期，默认9
                - trend_filter: 是否使用趋势过滤，默认True
                - histogram_threshold: 柱状图阈值，默认0.0
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'trend_filter': True,
            'histogram_threshold': 0.0
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MACDStrategy', default_params)
        self.logger.info(f"初始化MACD策略，参数: {default_params}")
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
            fast_period = self.parameters['fast_period']
            slow_period = self.parameters['slow_period']
            signal_period = self.parameters['signal_period']
            
            # 计算EMA
            df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # 计算MACD线
            df['macd_line'] = df['ema_fast'] - df['ema_slow']
            
            # 计算信号线
            df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
            
            # 计算柱状图
            df['histogram'] = df['macd_line'] - df['signal_line']
            
            # 标准化MACD值（相对于价格）
            df['macd_normalized'] = df['macd_line'] / df['close']
            df['signal_normalized'] = df['signal_line'] / df['close']
            df['histogram_normalized'] = df['histogram'] / df['close']
            
            # 计算MACD金叉、死叉
            df['golden_cross'] = (
                (df['macd_line'] > df['signal_line']) & 
                (df['macd_line'].shift(1) <= df['signal_line'].shift(1))
            )
            
            df['death_cross'] = (
                (df['macd_line'] < df['signal_line']) & 
                (df['macd_line'].shift(1) >= df['signal_line'].shift(1))
            )
            
            # 计算MACD趋势
            df['bullish_trend'] = df['macd_line'] > 0
            df['bearish_trend'] = df['macd_line'] < 0
            
            # 柱状图变化
            df['histogram_increasing'] = df['histogram'] > df['histogram'].shift(1)
            df['histogram_decreasing'] = df['histogram'] < df['histogram'].shift(1)
            
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
            trend_filter = self.parameters['trend_filter']
            histogram_threshold = self.parameters['histogram_threshold']
            
            # 买入条件
            if trend_filter:
                # 带趋势过滤的买入条件
                buy_condition = (
                    df['golden_cross'] & 
                    (df['macd_line'] > 0) &
                    (df['histogram'] > histogram_threshold)
                )
            else:
                # 不带趋势过滤的买入条件
                buy_condition = (
                    df['golden_cross'] &
                    (df['histogram'] > histogram_threshold)
                )
            
            # 卖出条件
            if trend_filter:
                # 带趋势过滤的卖出条件
                sell_condition = (
                    df['death_cross'] & 
                    (df['macd_line'] < 0) &
                    (df['histogram'] < -histogram_threshold)
                )
            else:
                # 不带趋势过滤的卖出条件
                sell_condition = (
                    df['death_cross'] &
                    (df['histogram'] < -histogram_threshold)
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
        
        # MACD值，标准化到[-1, 1]区间
        # 通常MACD值较小，需要放大来获得更好的可视化效果
        max_macd = max(abs(df['macd_normalized'].max()), abs(df['macd_normalized'].min()))
        if max_macd == 0:
            max_macd = 0.001  # 防止除零错误
            
        macd_norm = df['macd_normalized'] / max_macd
        macd_norm = macd_norm.clip(-1, 1)
        
        # 信号线，标准化到[-1, 1]区间
        signal_norm = df['signal_normalized'] / max_macd
        signal_norm = signal_norm.clip(-1, 1)
        
        # 柱状图，标准化到[-1, 1]区间
        max_hist = max(abs(df['histogram_normalized'].max()), abs(df['histogram_normalized'].min()))
        if max_hist == 0:
            max_hist = 0.001  # 防止除零错误
            
        histogram_norm = df['histogram_normalized'] / max_hist
        histogram_norm = histogram_norm.clip(-1, 1)
        
        # 金叉死叉信号
        cross_signal = df['golden_cross'].astype(float) - df['death_cross'].astype(float)
        
        # 柱状图方向
        histogram_direction = df['histogram_increasing'].astype(float) - df['histogram_decreasing'].astype(float)
        
        return {
            "macd": macd_norm,
            "signal": signal_norm,
            "histogram": histogram_norm,
            "cross": cross_signal,
            "histogram_direction": histogram_direction,
            "composite": df['signal']
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "macd": {
                "type": "trend",
                "time_scale": "medium",
                "description": "MACD线（标准化）",
                "min_value": -1,
                "max_value": 1
            },
            "signal": {
                "type": "trend",
                "time_scale": "medium",
                "description": "信号线（标准化）",
                "min_value": -1,
                "max_value": 1
            },
            "histogram": {
                "type": "momentum",
                "time_scale": "medium",
                "description": "MACD柱状图（标准化）",
                "min_value": -1,
                "max_value": 1
            },
            "cross": {
                "type": "signal",
                "time_scale": "medium",
                "description": "MACD金叉或死叉信号",
                "min_value": -1,
                "max_value": 1
            },
            "histogram_direction": {
                "type": "momentum",
                "time_scale": "short",
                "description": "柱状图方向（增加或减少）",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "MACD策略综合信号",
                "min_value": -1,
                "max_value": 1
            }
        } 