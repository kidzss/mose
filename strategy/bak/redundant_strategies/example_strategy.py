<<<<<<< HEAD
"""
示例策略

演示如何使用技术指标模块来构建交易策略。
这个策略是一个多指标结合策略，同时使用移动平均线、RSI和MACD作为信号生成依据。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from strategy.strategy_base import Strategy
from strategy.indicators.indicators import TechnicalIndicators


class MultiIndicatorStrategy(Strategy):
    """
    多指标结合策略
    
    结合移动平均线、RSI和MACD三种技术指标，生成交易信号。
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化多指标结合策略
        
        参数:
            parameters: 策略参数字典，可包含：
                - fast_ma_type: 快速移动平均线类型，默认为'ema'
                - slow_ma_type: 慢速移动平均线类型，默认为'sma'
                - fast_ma_period: 快速移动平均线周期，默认为20
                - slow_ma_period: 慢速移动平均线周期，默认为50
                - rsi_period: RSI周期，默认为14
                - rsi_overbought: RSI超买阈值，默认为70
                - rsi_oversold: RSI超卖阈值，默认为30
                - macd_fast_period: MACD快线周期，默认为12
                - macd_slow_period: MACD慢线周期，默认为26
                - macd_signal_period: MACD信号线周期，默认为9
                - rsi_weight: RSI信号权重，默认为1.0
                - ma_weight: 移动平均线信号权重，默认为1.0
                - macd_weight: MACD信号权重，默认为1.0
                - signal_threshold: 信号阈值，默认为0.5
        """
        default_params = {
            'fast_ma_type': 'ema',
            'slow_ma_type': 'sma',
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'rsi_weight': 1.0,
            'ma_weight': 1.0,
            'macd_weight': 1.0,
            'signal_threshold': 0.5
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MultiIndicatorStrategy', default_params)
        self.logger.info(f"初始化多指标结合策略，参数: {default_params}")
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
            
            # 计算移动平均线
            fast_ma = TechnicalIndicators.calculate_ma(
                df['close'], 
                self.parameters['fast_ma_type'], 
                self.parameters['fast_ma_period']
            )
            
            slow_ma = TechnicalIndicators.calculate_ma(
                df['close'], 
                self.parameters['slow_ma_type'], 
                self.parameters['slow_ma_period']
            )
            
            df['fast_ma'] = fast_ma
            df['slow_ma'] = slow_ma
            
            # 计算移动平均线交叉信号
            ma_crossover = TechnicalIndicators.get_crossover_signals(fast_ma, slow_ma)
            df['ma_cross_up'] = ma_crossover['cross_up']
            df['ma_cross_down'] = ma_crossover['cross_down']
            
            # 计算RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(
                df['close'], 
                self.parameters['rsi_period']
            )
            
            # 计算RSI超买超卖信号
            rsi_signals = TechnicalIndicators.calculate_rsi(df['close']).to_frame('rsi')
            rsi_signals['overbought'] = (rsi_signals['rsi'] > self.parameters['rsi_overbought']).astype(int)
            rsi_signals['oversold'] = (rsi_signals['rsi'] < self.parameters['rsi_oversold']).astype(int)
            rsi_signals['overbought_exit'] = ((rsi_signals['rsi'] < self.parameters['rsi_overbought']) & 
                                             (rsi_signals['rsi'].shift(1) >= self.parameters['rsi_overbought'])).astype(int)
            rsi_signals['oversold_exit'] = ((rsi_signals['rsi'] > self.parameters['rsi_oversold']) & 
                                           (rsi_signals['rsi'].shift(1) <= self.parameters['rsi_oversold'])).astype(int)
            
            df['rsi_overbought'] = rsi_signals['overbought']
            df['rsi_oversold'] = rsi_signals['oversold']
            df['rsi_overbought_exit'] = rsi_signals['overbought_exit']
            df['rsi_oversold_exit'] = rsi_signals['oversold_exit']
            
            # 计算MACD
            macd_data = TechnicalIndicators.calculate_macd(
                df['close'], 
                self.parameters['macd_fast_period'], 
                self.parameters['macd_slow_period'], 
                self.parameters['macd_signal_period']
            )
            
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_hist'] = macd_data['histogram']
            
            # 计算MACD交叉信号
            macd_crossover = TechnicalIndicators.get_crossover_signals(macd_data['macd'], macd_data['signal'])
            df['macd_cross_up'] = macd_crossover['cross_up']
            df['macd_cross_down'] = macd_crossover['cross_down']
            
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
            
            # 计算综合信号强度
            ma_signal = 0
            rsi_signal = 0
            macd_signal = 0
            
            # 移动平均线信号
            ma_signal = np.where(df['ma_cross_up'] == 1, 1, 
                                np.where(df['ma_cross_down'] == 1, -1, 0))
            
            # RSI信号
            rsi_signal = np.where(df['rsi_oversold_exit'] == 1, 1, 
                                 np.where(df['rsi_overbought_exit'] == 1, -1, 0))
            
            # MACD信号
            macd_signal = np.where(df['macd_cross_up'] == 1, 1, 
                                  np.where(df['macd_cross_down'] == 1, -1, 0))
            
            # 综合信号强度
            df['signal_strength'] = (
                ma_signal * self.parameters['ma_weight'] + 
                rsi_signal * self.parameters['rsi_weight'] + 
                macd_signal * self.parameters['macd_weight']
            ) / (self.parameters['ma_weight'] + self.parameters['rsi_weight'] + self.parameters['macd_weight'])
            
            # 根据信号强度和阈值生成最终信号
            threshold = self.parameters['signal_threshold']
            df.loc[df['signal_strength'] > threshold, 'signal'] = 1
            df.loc[df['signal_strength'] < -threshold, 'signal'] = -1
            
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
        
        # 移动平均线组件，标准化到[-1, 1]
        ma_diff = (df['fast_ma'] - df['slow_ma']) / df['slow_ma'] * 10
        ma_diff = ma_diff.clip(-1, 1)
        
        # RSI组件，标准化到[-1, 1]
        rsi_norm = (df['rsi'] - 50) / 50
        
        # MACD组件，标准化
        macd_max = max(abs(df['macd'].max()), abs(df['macd'].min()))
        macd_norm = df['macd'] / macd_max if macd_max != 0 else df['macd']
        
        # 组件融合
        ma_weight = self.parameters['ma_weight']
        rsi_weight = self.parameters['rsi_weight']
        macd_weight = self.parameters['macd_weight']
        total_weight = ma_weight + rsi_weight + macd_weight
        
        composite = (ma_diff * ma_weight + rsi_norm * rsi_weight + macd_norm * macd_weight) / total_weight
        
        return {
            "ma_component": ma_diff,
            "rsi_component": rsi_norm,
            "macd_component": macd_norm,
            "composite": composite
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "ma_component": {
                "type": "trend",
                "time_scale": "medium",
                "description": "移动平均线差异（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "rsi_component": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI值（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "macd_component": {
                "type": "momentum",
                "time_scale": "medium",
                "description": "MACD值（标准化）",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "多指标综合信号",
                "min_value": -1,
                "max_value": 1
            }
=======
"""
示例策略

演示如何使用技术指标模块来构建交易策略。
这个策略是一个多指标结合策略，同时使用移动平均线、RSI和MACD作为信号生成依据。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from strategy.strategy_base import Strategy
from strategy.indicators.indicators import TechnicalIndicators


class MultiIndicatorStrategy(Strategy):
    """
    多指标结合策略
    
    结合移动平均线、RSI和MACD三种技术指标，生成交易信号。
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化多指标结合策略
        
        参数:
            parameters: 策略参数字典，可包含：
                - fast_ma_type: 快速移动平均线类型，默认为'ema'
                - slow_ma_type: 慢速移动平均线类型，默认为'sma'
                - fast_ma_period: 快速移动平均线周期，默认为20
                - slow_ma_period: 慢速移动平均线周期，默认为50
                - rsi_period: RSI周期，默认为14
                - rsi_overbought: RSI超买阈值，默认为70
                - rsi_oversold: RSI超卖阈值，默认为30
                - macd_fast_period: MACD快线周期，默认为12
                - macd_slow_period: MACD慢线周期，默认为26
                - macd_signal_period: MACD信号线周期，默认为9
                - rsi_weight: RSI信号权重，默认为1.0
                - ma_weight: 移动平均线信号权重，默认为1.0
                - macd_weight: MACD信号权重，默认为1.0
                - signal_threshold: 信号阈值，默认为0.5
        """
        default_params = {
            'fast_ma_type': 'ema',
            'slow_ma_type': 'sma',
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            'rsi_weight': 1.0,
            'ma_weight': 1.0,
            'macd_weight': 1.0,
            'signal_threshold': 0.5
        }
        
        # 合并参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('MultiIndicatorStrategy', default_params)
        self.logger.info(f"初始化多指标结合策略，参数: {default_params}")
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
            
            # 计算移动平均线
            fast_ma = TechnicalIndicators.calculate_ma(
                df['close'], 
                self.parameters['fast_ma_type'], 
                self.parameters['fast_ma_period']
            )
            
            slow_ma = TechnicalIndicators.calculate_ma(
                df['close'], 
                self.parameters['slow_ma_type'], 
                self.parameters['slow_ma_period']
            )
            
            df['fast_ma'] = fast_ma
            df['slow_ma'] = slow_ma
            
            # 计算移动平均线交叉信号
            ma_crossover = TechnicalIndicators.get_crossover_signals(fast_ma, slow_ma)
            df['ma_cross_up'] = ma_crossover['cross_up']
            df['ma_cross_down'] = ma_crossover['cross_down']
            
            # 计算RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(
                df['close'], 
                self.parameters['rsi_period']
            )
            
            # 计算RSI超买超卖信号
            rsi_signals = TechnicalIndicators.calculate_rsi(df['close']).to_frame('rsi')
            rsi_signals['overbought'] = (rsi_signals['rsi'] > self.parameters['rsi_overbought']).astype(int)
            rsi_signals['oversold'] = (rsi_signals['rsi'] < self.parameters['rsi_oversold']).astype(int)
            rsi_signals['overbought_exit'] = ((rsi_signals['rsi'] < self.parameters['rsi_overbought']) & 
                                             (rsi_signals['rsi'].shift(1) >= self.parameters['rsi_overbought'])).astype(int)
            rsi_signals['oversold_exit'] = ((rsi_signals['rsi'] > self.parameters['rsi_oversold']) & 
                                           (rsi_signals['rsi'].shift(1) <= self.parameters['rsi_oversold'])).astype(int)
            
            df['rsi_overbought'] = rsi_signals['overbought']
            df['rsi_oversold'] = rsi_signals['oversold']
            df['rsi_overbought_exit'] = rsi_signals['overbought_exit']
            df['rsi_oversold_exit'] = rsi_signals['oversold_exit']
            
            # 计算MACD
            macd_data = TechnicalIndicators.calculate_macd(
                df['close'], 
                self.parameters['macd_fast_period'], 
                self.parameters['macd_slow_period'], 
                self.parameters['macd_signal_period']
            )
            
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_hist'] = macd_data['histogram']
            
            # 计算MACD交叉信号
            macd_crossover = TechnicalIndicators.get_crossover_signals(macd_data['macd'], macd_data['signal'])
            df['macd_cross_up'] = macd_crossover['cross_up']
            df['macd_cross_down'] = macd_crossover['cross_down']
            
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
            
            # 计算综合信号强度
            ma_signal = 0
            rsi_signal = 0
            macd_signal = 0
            
            # 移动平均线信号
            ma_signal = np.where(df['ma_cross_up'] == 1, 1, 
                                np.where(df['ma_cross_down'] == 1, -1, 0))
            
            # RSI信号
            rsi_signal = np.where(df['rsi_oversold_exit'] == 1, 1, 
                                 np.where(df['rsi_overbought_exit'] == 1, -1, 0))
            
            # MACD信号
            macd_signal = np.where(df['macd_cross_up'] == 1, 1, 
                                  np.where(df['macd_cross_down'] == 1, -1, 0))
            
            # 综合信号强度
            df['signal_strength'] = (
                ma_signal * self.parameters['ma_weight'] + 
                rsi_signal * self.parameters['rsi_weight'] + 
                macd_signal * self.parameters['macd_weight']
            ) / (self.parameters['ma_weight'] + self.parameters['rsi_weight'] + self.parameters['macd_weight'])
            
            # 根据信号强度和阈值生成最终信号
            threshold = self.parameters['signal_threshold']
            df.loc[df['signal_strength'] > threshold, 'signal'] = 1
            df.loc[df['signal_strength'] < -threshold, 'signal'] = -1
            
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
        
        # 移动平均线组件，标准化到[-1, 1]
        ma_diff = (df['fast_ma'] - df['slow_ma']) / df['slow_ma'] * 10
        ma_diff = ma_diff.clip(-1, 1)
        
        # RSI组件，标准化到[-1, 1]
        rsi_norm = (df['rsi'] - 50) / 50
        
        # MACD组件，标准化
        macd_max = max(abs(df['macd'].max()), abs(df['macd'].min()))
        macd_norm = df['macd'] / macd_max if macd_max != 0 else df['macd']
        
        # 组件融合
        ma_weight = self.parameters['ma_weight']
        rsi_weight = self.parameters['rsi_weight']
        macd_weight = self.parameters['macd_weight']
        total_weight = ma_weight + rsi_weight + macd_weight
        
        composite = (ma_diff * ma_weight + rsi_norm * rsi_weight + macd_norm * macd_weight) / total_weight
        
        return {
            "ma_component": ma_diff,
            "rsi_component": rsi_norm,
            "macd_component": macd_norm,
            "composite": composite
        }
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            "ma_component": {
                "type": "trend",
                "time_scale": "medium",
                "description": "移动平均线差异（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "rsi_component": {
                "type": "oscillator",
                "time_scale": "short",
                "description": "RSI值（标准化到[-1, 1]）",
                "min_value": -1,
                "max_value": 1
            },
            "macd_component": {
                "type": "momentum",
                "time_scale": "medium",
                "description": "MACD值（标准化）",
                "min_value": -1,
                "max_value": 1
            },
            "composite": {
                "type": "composite",
                "time_scale": "medium",
                "description": "多指标综合信号",
                "min_value": -1,
                "max_value": 1
            }
>>>>>>> 3d7330be7ea0ecb409ac485e1c8391bc6d56a2de
        } 