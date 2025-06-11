"""
技术指标综合模块

提供一个统一的接口，集成所有已实现的技术指标，方便在策略中使用。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, List, Tuple, Any

# 导入所有指标模块
from .moving_averages import sma, ema, wma, smma, tema, kama, hull_ma
from .bollinger_bands import bollinger_bands, bollinger_band_squeeze, bollinger_breakout, bollinger_reversal
from .rsi import rsi, rsi_divergence, rsi_overbought_oversold, rsi_reversal, stochastic_rsi
from .macd import macd, macd_crossover, macd_zero_crossover, macd_divergence, macd_histogram_reversal, ppo
from .adx import adx, adx_trend_strength, adx_trend_direction, adx_crossover, adx_reversal, dmi_oscillator

# 导入新添加的指标模块
from .volatility import atr, historical_volatility, bollinger_bandwidth, keltner_channel_width, volatility_ratio
from .volume import volume_sma, volume_ratio, on_balance_volume, chaikin_money_flow, money_flow_index
from .oscillators import stochastic_oscillator, williams_r, roc, cci, awesome_oscillator
from .trend_strength import adx as ts_adx, aroon, vortex_indicator, directional_movement_index, supertrend
from .support_resistance import pivot_points_traditional, price_channels, donchian_channels, fibonacci_retracement, keltner_channel


class TechnicalIndicators:
    """
    技术指标计算类
    
    提供统一的接口计算各种技术指标，并可以批量计算多个指标。
    """
    
    @staticmethod
    def calculate_ma(data: pd.Series, 
                     ma_type: str = 'sma', 
                     window: int = 20, 
                     **kwargs) -> pd.Series:
        """
        计算移动平均线
        
        参数:
            data: 价格数据
            ma_type: 移动平均线类型，可选 'sma', 'ema', 'wma', 'smma', 'tema', 'kama', 'hull'
            window: 窗口大小
            **kwargs: 其他特定于各种MA的参数
            
        返回:
            移动平均线数据
        """
        ma_functions = {
            'sma': sma,
            'ema': ema,
            'wma': wma,
            'smma': smma,
            'tema': tema,
            'kama': kama,
            'hull': hull_ma
        }
        
        if ma_type not in ma_functions:
            raise ValueError(f"不支持的移动平均线类型: {ma_type}，可选: {', '.join(ma_functions.keys())}")
        
        return ma_functions[ma_type](data, window, **kwargs)
    
    @staticmethod
    def calculate_bb(data: pd.Series, 
                     window: int = 20, 
                     num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带指标
        
        参数:
            data: 价格数据
            window: 窗口大小
            num_std: 标准差的倍数
            
        返回:
            包含中轨、上轨、下轨、带宽和百分比b的字典
        """
        return bollinger_bands(data, window, num_std)
    
    @staticmethod
    def calculate_rsi(data: pd.Series, 
                      window: int = 14, 
                      method: str = 'wilders') -> pd.Series:
        """
        计算RSI指标
        
        参数:
            data: 价格数据
            window: 窗口大小
            method: 计算方法，可选 'wilders' 或 'ema'
            
        返回:
            RSI值
        """
        return rsi(data, window, method)
    
    @staticmethod
    def calculate_macd(data: pd.Series, 
                       fast_period: int = 12, 
                       slow_period: int = 26, 
                       signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        参数:
            data: 价格数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        返回:
            包含MACD线、信号线和柱状图的字典
        """
        return macd(data, fast_period, slow_period, signal_period)
    
    @staticmethod
    def calculate_adx(high: pd.Series, 
                      low: pd.Series, 
                      close: pd.Series, 
                      window: int = 14) -> Dict[str, pd.Series]:
        """
        计算ADX指标
        
        参数:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口大小
            
        返回:
            包含ADX、+DI、-DI的字典
        """
        return ts_adx(high, low, close, window)
    
    @staticmethod
    def calculate_atr(high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series, 
                     window: int = 14) -> pd.Series:
        """
        计算ATR指标
        
        参数:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            window: 窗口大小
            
        返回:
            ATR值
        """
        return atr(high, low, close, window)
    
    @staticmethod
    def calculate_volatility(close: pd.Series, 
                            window: int = 20, 
                            annualize: bool = True) -> pd.Series:
        """
        计算历史波动率
        
        参数:
            close: 收盘价数据
            window: 窗口大小
            annualize: 是否年化
            
        返回:
            波动率值
        """
        return historical_volatility(close, window, annualize)
    
    @staticmethod
    def calculate_volume_indicators(close: pd.Series, 
                                  volume: pd.Series, 
                                  high: Optional[pd.Series] = None, 
                                  low: Optional[pd.Series] = None, 
                                  indicator_type: str = 'obv') -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        计算成交量指标
        
        参数:
            close: 收盘价数据
            volume: 成交量数据
            high: 最高价数据（某些指标需要）
            low: 最低价数据（某些指标需要）
            indicator_type: 指标类型，可选 'obv', 'cmf', 'mfi'
            
        返回:
            成交量指标值
        """
        if indicator_type == 'obv':
            return on_balance_volume(close, volume)
        elif indicator_type == 'cmf' and high is not None and low is not None:
            return chaikin_money_flow(high, low, close, volume)
        elif indicator_type == 'mfi' and high is not None and low is not None:
            return money_flow_index(high, low, close, volume)
        elif indicator_type == 'volume_ratio':
            return volume_ratio(volume)
        else:
            raise ValueError(f"不支持的成交量指标类型: {indicator_type}")
    
    @staticmethod
    def calculate_oscillators(high: pd.Series, 
                             low: pd.Series, 
                             close: pd.Series, 
                             oscillator_type: str = 'stoch') -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        计算震荡指标
        
        参数:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            oscillator_type: 指标类型，可选 'stoch', 'willr', 'roc', 'cci', 'ao'
            
        返回:
            震荡指标值
        """
        if oscillator_type == 'stoch':
            return stochastic_oscillator(high, low, close)
        elif oscillator_type == 'willr':
            return williams_r(high, low, close)
        elif oscillator_type == 'roc':
            return roc(close)
        elif oscillator_type == 'cci':
            return cci(high, low, close)
        elif oscillator_type == 'ao':
            return awesome_oscillator(high, low)
        else:
            raise ValueError(f"不支持的震荡指标类型: {oscillator_type}")
    
    @staticmethod
    def calculate_trend_strength(high: pd.Series, 
                               low: pd.Series, 
                               close: pd.Series, 
                               indicator_type: str = 'adx') -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        计算趋势强度指标
        
        参数:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            indicator_type: 指标类型，可选 'adx', 'aroon', 'vortex', 'supertrend'
            
        返回:
            趋势强度指标值
        """
        if indicator_type == 'adx':
            return ts_adx(high, low, close)
        elif indicator_type == 'aroon':
            return aroon(high, low)
        elif indicator_type == 'vortex':
            return vortex_indicator(high, low, close)
        elif indicator_type == 'supertrend':
            return supertrend(high, low, close)
        else:
            raise ValueError(f"不支持的趋势强度指标类型: {indicator_type}")
    
    @staticmethod
    def calculate_support_resistance(high: pd.Series, 
                                   low: pd.Series, 
                                   close: pd.Series, 
                                   indicator_type: str = 'pivot') -> Dict[str, pd.Series]:
        """
        计算支撑与阻力指标
        
        参数:
            high: 最高价数据
            low: 最低价数据
            close: 收盘价数据
            indicator_type: 指标类型，可选 'pivot', 'price_channels', 'donchian', 'keltner'
            
        返回:
            支撑与阻力指标值
        """
        if indicator_type == 'pivot':
            return pivot_points_traditional(high, low, close)
        elif indicator_type == 'price_channels':
            return price_channels(high, low)
        elif indicator_type == 'donchian':
            return donchian_channels(high, low)
        elif indicator_type == 'keltner':
            return keltner_channel(high, low, close)
        else:
            raise ValueError(f"不支持的支撑与阻力指标类型: {indicator_type}")
    
    @staticmethod
    def calculate_divergence(price: pd.Series, 
                             indicator: pd.Series, 
                             window: int = 5, 
                             threshold: float = 2.0) -> pd.DataFrame:
        """
        计算价格与指标的背离
        
        参数:
            price: 价格数据
            indicator: 指标数据，可以是RSI、MACD等
            window: 检测局部极值的窗口大小
            threshold: 用于确定显著背离的阈值
            
        返回:
            包含看涨和看跌背离信号的DataFrame
        """
        # 根据指标类型选择合适的背离检测函数
        if len(indicator) == len(price):  # 确保指标与价格长度相同
            # 默认使用MACD背离检测
            return macd_divergence(price, indicator, window, threshold)
        else:
            raise ValueError("指标和价格数据长度不匹配")
    
    @classmethod
    def calculate_all_indicators(cls, 
                                df: pd.DataFrame, 
                                selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量计算多个指标
        
        参数:
            df: 包含OHLCV数据的DataFrame
            selected_indicators: 要计算的指标列表，如果为None则计算所有指标
            
        返回:
            添加了各种指标的DataFrame
        """
        result_df = df.copy()
        
        # 可用的指标及其计算函数
        available_indicators = {
            'sma': lambda: cls.calculate_ma(df['close'], 'sma', 20),
            'ema': lambda: cls.calculate_ma(df['close'], 'ema', 20),
            'wma': lambda: cls.calculate_ma(df['close'], 'wma', 20),
            'bb': lambda: cls.calculate_bb(df['close']),
            'rsi': lambda: cls.calculate_rsi(df['close']),
            'macd': lambda: cls.calculate_macd(df['close']),
            'adx': lambda: cls.calculate_adx(df['high'], df['low'], df['close']),
            'atr': lambda: cls.calculate_atr(df['high'], df['low'], df['close']),
            'volatility': lambda: cls.calculate_volatility(df['close']),
            'obv': lambda: cls.calculate_volume_indicators(df['close'], df['volume'], indicator_type='obv'),
            'cmf': lambda: cls.calculate_volume_indicators(df['close'], df['volume'], df['high'], df['low'], 'cmf'),
            'stoch': lambda: cls.calculate_oscillators(df['high'], df['low'], df['close'], 'stoch'),
            'willr': lambda: cls.calculate_oscillators(df['high'], df['low'], df['close'], 'willr'),
            'aroon': lambda: cls.calculate_trend_strength(df['high'], df['low'], df['close'], 'aroon'),
            'supertrend': lambda: cls.calculate_trend_strength(df['high'], df['low'], df['close'], 'supertrend'),
            'pivot': lambda: cls.calculate_support_resistance(df['high'], df['low'], df['close'], 'pivot'),
            'donchian': lambda: cls.calculate_support_resistance(df['high'], df['low'], df['close'], 'donchian')
        }
        
        # 如果没有指定，计算所有指标
        if selected_indicators is None:
            selected_indicators = list(available_indicators.keys())
        
        # 计算选定的指标
        for indicator in selected_indicators:
            if indicator in available_indicators:
                if indicator == 'bb':
                    bb_data = available_indicators[indicator]()
                    result_df['bb_middle'] = bb_data['middle']
                    result_df['bb_upper'] = bb_data['upper']
                    result_df['bb_lower'] = bb_data['lower']
                    result_df['bb_bandwidth'] = bb_data['bandwidth']
                    result_df['bb_percent'] = bb_data['b_percent']
                elif indicator == 'macd':
                    macd_data = available_indicators[indicator]()
                    result_df['macd'] = macd_data['macd']
                    result_df['macd_signal'] = macd_data['signal']
                    result_df['macd_hist'] = macd_data['histogram']
                elif indicator == 'adx':
                    adx_data = available_indicators[indicator]()
                    result_df['adx'] = adx_data['adx']
                    result_df['plus_di'] = adx_data['plus_di']
                    result_df['minus_di'] = adx_data['minus_di']
                elif indicator == 'stoch':
                    stoch_data = available_indicators[indicator]()
                    result_df['stoch_k'] = stoch_data['k']
                    result_df['stoch_d'] = stoch_data['d']
                elif indicator == 'aroon':
                    aroon_data = available_indicators[indicator]()
                    result_df['aroon_up'] = aroon_data['aroon_up']
                    result_df['aroon_down'] = aroon_data['aroon_down']
                    result_df['aroon_osc'] = aroon_data['aroon_osc']
                elif indicator == 'supertrend':
                    supertrend_data = available_indicators[indicator]()
                    result_df['supertrend'] = supertrend_data['supertrend']
                    result_df['supertrend_trend'] = supertrend_data['trend']
                elif indicator == 'pivot':
                    pivot_data = available_indicators[indicator]()
                    result_df['pivot'] = pivot_data['pivot']
                    result_df['support1'] = pivot_data['support1']
                    result_df['resistance1'] = pivot_data['resistance1']
                elif indicator == 'donchian':
                    donchian_data = available_indicators[indicator]()
                    result_df['donchian_upper'] = donchian_data['upper']
                    result_df['donchian_middle'] = donchian_data['middle']
                    result_df['donchian_lower'] = donchian_data['lower']
                else:
                    result_df[indicator] = available_indicators[indicator]()
            else:
                print(f"警告: 不支持的指标 '{indicator}'")
        
        return result_df
    
    @staticmethod
    def get_crossover_signals(series1: pd.Series, 
                              series2: pd.Series) -> pd.DataFrame:
        """
        计算两个序列的交叉信号
        
        参数:
            series1: 第一个序列
            series2: 第二个序列
            
        返回:
            包含上穿和下穿信号的DataFrame
        """
        # 上穿信号 (series1从下方穿过series2)
        cross_up = ((series1 > series2) & (series1.shift(1) <= series2.shift(1))).astype(int)
        
        # 下穿信号 (series1从上方穿过series2)
        cross_down = ((series1 < series2) & (series1.shift(1) >= series2.shift(1))).astype(int)
        
        return pd.DataFrame({
            'cross_up': cross_up,
            'cross_down': cross_down
        })


# 便捷函数，直接从模块级别调用
def calculate_indicators(df: pd.DataFrame, 
                        selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    批量计算多个指标的便捷函数
    
    参数:
        df: 包含OHLCV数据的DataFrame
        selected_indicators: 要计算的指标列表
        
    返回:
        添加了各种指标的DataFrame
    """
    return TechnicalIndicators.calculate_all_indicators(df, selected_indicators) 