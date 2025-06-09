"""
支撑与阻力指标模块

提供各种支撑和阻力位的计算方法，包括枢轴点、价格通道、斐波那契回调等。
这些指标有助于识别潜在的价格反转点和关键价格水平。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple, List


def pivot_points_traditional(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    """
    计算传统的枢轴点及支撑/阻力位
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        
    返回:
        包含枢轴点(PP)、支撑位(S1,S2,S3)和阻力位(R1,R2,R3)的字典
    """
    # 初始化结果序列
    pp = pd.Series(0.0, index=close.index)
    s1 = pd.Series(0.0, index=close.index)
    s2 = pd.Series(0.0, index=close.index)
    s3 = pd.Series(0.0, index=close.index)
    r1 = pd.Series(0.0, index=close.index)
    r2 = pd.Series(0.0, index=close.index)
    r3 = pd.Series(0.0, index=close.index)
    
    # 计算枢轴点和支撑/阻力位
    for i in range(1, len(close)):
        # 枢轴点
        pp.iloc[i] = (high.iloc[i-1] + low.iloc[i-1] + close.iloc[i-1]) / 3
        
        # 第一支撑位和阻力位
        s1.iloc[i] = (2 * pp.iloc[i]) - high.iloc[i-1]
        r1.iloc[i] = (2 * pp.iloc[i]) - low.iloc[i-1]
        
        # 第二支撑位和阻力位
        s2.iloc[i] = pp.iloc[i] - (high.iloc[i-1] - low.iloc[i-1])
        r2.iloc[i] = pp.iloc[i] + (high.iloc[i-1] - low.iloc[i-1])
        
        # 第三支撑位和阻力位
        s3.iloc[i] = low.iloc[i-1] - 2 * (high.iloc[i-1] - pp.iloc[i])
        r3.iloc[i] = high.iloc[i-1] + 2 * (pp.iloc[i] - low.iloc[i-1])
    
    return {
        'pivot': pp,
        'support1': s1,
        'support2': s2,
        'support3': s3,
        'resistance1': r1,
        'resistance2': r2,
        'resistance3': r3
    }


def pivot_points_fibonacci(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    """
    计算基于斐波那契的枢轴点及支撑/阻力位
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        
    返回:
        包含枢轴点(PP)、支撑位(S1,S2,S3)和阻力位(R1,R2,R3)的字典
    """
    # 初始化结果序列
    pp = pd.Series(0.0, index=close.index)
    s1 = pd.Series(0.0, index=close.index)
    s2 = pd.Series(0.0, index=close.index)
    s3 = pd.Series(0.0, index=close.index)
    r1 = pd.Series(0.0, index=close.index)
    r2 = pd.Series(0.0, index=close.index)
    r3 = pd.Series(0.0, index=close.index)
    
    # 计算枢轴点和支撑/阻力位
    for i in range(1, len(close)):
        # 枢轴点
        pp.iloc[i] = (high.iloc[i-1] + low.iloc[i-1] + close.iloc[i-1]) / 3
        
        # 计算范围
        range_val = high.iloc[i-1] - low.iloc[i-1]
        
        # 斐波那契支撑位和阻力位
        r1.iloc[i] = pp.iloc[i] + 0.382 * range_val
        r2.iloc[i] = pp.iloc[i] + 0.618 * range_val
        r3.iloc[i] = pp.iloc[i] + 1.000 * range_val
        
        s1.iloc[i] = pp.iloc[i] - 0.382 * range_val
        s2.iloc[i] = pp.iloc[i] - 0.618 * range_val
        s3.iloc[i] = pp.iloc[i] - 1.000 * range_val
    
    return {
        'pivot': pp,
        'support1': s1,
        'support2': s2,
        'support3': s3,
        'resistance1': r1,
        'resistance2': r2,
        'resistance3': r3
    }


def pivot_points_woodie(high: pd.Series, low: pd.Series, close: pd.Series, open_price: pd.Series) -> Dict[str, pd.Series]:
    """
    计算Woodie枢轴点及支撑/阻力位
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        open_price: 开盘价序列
        
    返回:
        包含枢轴点(PP)、支撑位(S1,S2)和阻力位(R1,R2)的字典
    """
    # 初始化结果序列
    pp = pd.Series(0.0, index=close.index)
    s1 = pd.Series(0.0, index=close.index)
    s2 = pd.Series(0.0, index=close.index)
    r1 = pd.Series(0.0, index=close.index)
    r2 = pd.Series(0.0, index=close.index)
    
    # 计算Woodie枢轴点和支撑/阻力位
    for i in range(1, len(close)):
        # Woodie枢轴点
        pp.iloc[i] = (high.iloc[i-1] + low.iloc[i-1] + 2 * close.iloc[i-1]) / 4
        
        # 支撑位和阻力位
        r1.iloc[i] = (2 * pp.iloc[i]) - low.iloc[i-1]
        s1.iloc[i] = (2 * pp.iloc[i]) - high.iloc[i-1]
        
        r2.iloc[i] = pp.iloc[i] + (high.iloc[i-1] - low.iloc[i-1])
        s2.iloc[i] = pp.iloc[i] - (high.iloc[i-1] - low.iloc[i-1])
    
    return {
        'pivot': pp,
        'support1': s1,
        'support2': s2,
        'resistance1': r1,
        'resistance2': r2
    }


def price_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    计算价格通道
    
    参数:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
        
    返回:
        包含上通道、下通道和中线的字典
    """
    # 计算上通道、下通道和中线
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return {
        'upper': upper,
        'lower': lower,
        'middle': middle
    }


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    计算唐奇安通道
    
    参数:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
        
    返回:
        包含上通道、下通道和中线的字典
    """
    # 计算上通道、下通道和中线
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return {
        'upper': upper,
        'lower': lower,
        'middle': middle
    }


def fibonacci_retracement(high: pd.Series, low: pd.Series, is_uptrend: bool = True) -> Dict[str, pd.Series]:
    """
    计算斐波那契回调位
    
    参数:
        high: 最高价序列
        low: 最低价序列
        is_uptrend: 是否为上升趋势，默认为True
        
    返回:
        包含各回调位的字典
    """
    # 初始化结果序列
    fib_0 = pd.Series(0.0, index=high.index)
    fib_236 = pd.Series(0.0, index=high.index)
    fib_382 = pd.Series(0.0, index=high.index)
    fib_500 = pd.Series(0.0, index=high.index)
    fib_618 = pd.Series(0.0, index=high.index)
    fib_786 = pd.Series(0.0, index=high.index)
    fib_1000 = pd.Series(0.0, index=high.index)
    
    # 计算回调位
    for i in range(1, len(high)):
        # 确定高点和低点
        if is_uptrend:
            swing_high = high.iloc[:i].max()
            swing_low = low.iloc[:i].min()
        else:
            swing_high = low.iloc[:i].max()
            swing_low = high.iloc[:i].min()
        
        # 计算价格差值
        price_range = swing_high - swing_low
        
        # 计算各回调位
        if is_uptrend:
            fib_0.iloc[i] = swing_high
            fib_236.iloc[i] = swing_high - 0.236 * price_range
            fib_382.iloc[i] = swing_high - 0.382 * price_range
            fib_500.iloc[i] = swing_high - 0.500 * price_range
            fib_618.iloc[i] = swing_high - 0.618 * price_range
            fib_786.iloc[i] = swing_high - 0.786 * price_range
            fib_1000.iloc[i] = swing_low
        else:
            fib_0.iloc[i] = swing_low
            fib_236.iloc[i] = swing_low + 0.236 * price_range
            fib_382.iloc[i] = swing_low + 0.382 * price_range
            fib_500.iloc[i] = swing_low + 0.500 * price_range
            fib_618.iloc[i] = swing_low + 0.618 * price_range
            fib_786.iloc[i] = swing_low + 0.786 * price_range
            fib_1000.iloc[i] = swing_high
    
    return {
        'fib_0': fib_0,
        'fib_236': fib_236,
        'fib_382': fib_382,
        'fib_500': fib_500,
        'fib_618': fib_618,
        'fib_786': fib_786,
        'fib_1000': fib_1000
    }


def support_resistance_levels(close: pd.Series, high: pd.Series, low: pd.Series, 
                              window: int = 20, threshold: float = 0.02) -> Dict[str, List[float]]:
    """
    识别支撑和阻力水平
    
    参数:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        window: 窗口大小
        threshold: 价格聚集阈值
        
    返回:
        包含支撑位和阻力位的字典
    """
    if len(close) < window:
        return {'support': [], 'resistance': []}
    
    # 获取最近的数据点
    recent_high = high[-window:].max()
    recent_low = low[-window:].min()
    
    # 计算价格范围
    price_range = recent_high - recent_low
    
    # 设置区间数（太多的区间会导致计算复杂）
    num_bins = 20
    
    # 创建价格区间
    bins = np.linspace(recent_low, recent_high, num_bins)
    
    # 创建直方图
    hist, bin_edges = np.histogram(close[-window:], bins=bins)
    
    # 寻找频率高峰
    peak_indices = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peak_indices.append(i)
    
    # 识别支撑位和阻力位
    support_levels = []
    resistance_levels = []
    
    for idx in peak_indices:
        price_level = (bin_edges[idx] + bin_edges[idx+1]) / 2
        
        # 根据当前价格判断是支撑还是阻力
        if price_level < close.iloc[-1]:
            support_levels.append(price_level)
        else:
            resistance_levels.append(price_level)
    
    # 按距离当前价格排序
    current_price = close.iloc[-1]
    support_levels.sort(key=lambda x: abs(current_price - x))
    resistance_levels.sort(key=lambda x: abs(current_price - x))
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, 
                   ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    计算Keltner通道
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        ema_period: EMA周期
        atr_period: ATR周期
        multiplier: ATR乘数
        
    返回:
        包含上通道、中线和下通道的字典
    """
    # 计算中线 (EMA)
    middle = close.ewm(span=ema_period, adjust=False).mean()
    
    # 计算ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    
    # 计算上下轨
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def ichimoku_cloud(high: pd.Series, low: pd.Series, 
                  tenkan_period: int = 9, kijun_period: int = 26, 
                  senkou_span_b_period: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
    """
    计算一目均衡表(云图)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        tenkan_period: 转换线周期
        kijun_period: 基准线周期
        senkou_span_b_period: 先行带B周期
        displacement: 延迟周期
        
    返回:
        包含转换线、基准线、先行带A、先行带B和延迟线的字典
    """
    # 转换线 (Tenkan-sen)
    tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    
    # 基准线 (Kijun-sen)
    kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    
    # 先行带A (Senkou Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # 先行带B (Senkou Span B)
    senkou_span_b = ((high.rolling(window=senkou_span_b_period).max() + 
                     low.rolling(window=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # 延迟线 (Chikou Span)
    chikou_span = high.shift(-displacement)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    } 