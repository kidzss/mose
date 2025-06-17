"""
布林带模块

提供布林带相关指标的计算函数，用于技术分析和交易策略。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict


def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                    window: int = 20, 
                    num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    计算布林带指标

    参数:
        data: 原始价格数据，通常是收盘价
        window: 窗口大小，默认为20
        num_std: 标准差的倍数，默认为2

    返回:
        包含中轨、上轨、下轨和带宽的字典
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # 计算中轨(简单移动平均线)
    middle_band = data.rolling(window=window).mean()
    
    # 计算标准差
    std = data.rolling(window=window).std()
    
    # 计算上轨和下轨
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # 计算带宽
    bandwidth = (upper_band - lower_band) / middle_band * 100
    
    # 计算百分比b
    b_percent = (data - lower_band) / (upper_band - lower_band)
    
    return {
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band,
        'bandwidth': bandwidth,
        'b_percent': b_percent
    }


def bollinger_band_squeeze(data: Union[pd.Series, np.ndarray], 
                           window: int = 20, 
                           num_std: float = 2.0,
                           bandwidth_ma_window: int = 20) -> pd.Series:
    """
    计算布林带挤压指标
    布林带挤压发生在带宽缩小到相对低位时，通常预示着即将发生的大幅波动

    参数:
        data: 原始价格数据
        window: 布林带窗口大小
        num_std: 标准差的倍数
        bandwidth_ma_window: 带宽移动平均窗口大小

    返回:
        布林带挤压指标，值越小表示挤压越严重
    """
    bb = bollinger_bands(data, window, num_std)
    
    # 计算带宽的移动平均
    bandwidth_ma = bb['bandwidth'].rolling(window=bandwidth_ma_window).mean()
    
    # 计算挤压指标 (当前带宽相对于移动平均)
    squeeze = bb['bandwidth'] / bandwidth_ma
    
    return squeeze


def bollinger_breakout(data: Union[pd.Series, np.ndarray], 
                       window: int = 20, 
                       num_std: float = 2.0) -> pd.DataFrame:
    """
    计算布林带突破信号

    参数:
        data: 原始价格数据
        window: 布林带窗口大小
        num_std: 标准差的倍数

    返回:
        包含上轨突破和下轨突破信号的DataFrame
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    bb = bollinger_bands(data, window, num_std)
    
    # 上轨突破信号
    upper_breakout = (data > bb['upper']).astype(int)
    
    # 下轨突破信号
    lower_breakout = (data < bb['lower']).astype(int)
    
    # 连续突破的天数
    upper_streak = upper_breakout.groupby((upper_breakout != upper_breakout.shift(1)).cumsum()).cumsum()
    lower_streak = lower_breakout.groupby((lower_breakout != lower_breakout.shift(1)).cumsum()).cumsum()
    
    result = pd.DataFrame({
        'upper_breakout': upper_breakout,
        'lower_breakout': lower_breakout,
        'upper_streak': upper_streak,
        'lower_streak': lower_streak
    })
    
    return result


def bollinger_reversal(data: Union[pd.Series, np.ndarray], 
                       window: int = 20, 
                       num_std: float = 2.0,
                       threshold: float = 0.05) -> pd.DataFrame:
    """
    计算布林带反转信号
    当价格触及或突破布林带后立即回落/反弹的情况

    参数:
        data: 原始价格数据
        window: 布林带窗口大小
        num_std: 标准差的倍数
        threshold: 反转阈值，价格回撤的百分比

    返回:
        包含看涨和看跌反转信号的DataFrame
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    bb = bollinger_bands(data, window, num_std)
    
    # 价格位置
    price_position = bb['b_percent']
    
    # 看涨反转 (价格跌破下轨后反弹)
    bullish_reversal = ((price_position.shift(1) <= 0) & 
                         (price_position > 0) & 
                         (data > data.shift(1) * (1 + threshold)))
    
    # 看跌反转 (价格突破上轨后回落)
    bearish_reversal = ((price_position.shift(1) >= 1) & 
                         (price_position < 1) & 
                         (data < data.shift(1) * (1 - threshold)))
    
    result = pd.DataFrame({
        'bullish_reversal': bullish_reversal.astype(int),
        'bearish_reversal': bearish_reversal.astype(int)
    })
    
    return result 