"""
移动平均线模块

提供各种移动平均线计算函数:
- 简单移动平均线 (SMA)
- 指数移动平均线 (EMA)
- 加权移动平均线 (WMA)
- 自适应移动平均线 (AMA)
- 平滑移动平均线 (SMMA)
- 三重指数平滑移动平均线 (TEMA)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def sma(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    计算简单移动平均线 (SMA)

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        window: 窗口大小

    返回:
        pandas Series包含SMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    return data.rolling(window=window).mean()


def ema(data: Union[pd.Series, np.ndarray], span: int, adjust: bool = False) -> pd.Series:
    """
    计算指数移动平均线 (EMA)

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        span: 时间跨度，用于确定alpha值
        adjust: 是否调整权重，默认为False

    返回:
        pandas Series包含EMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    return data.ewm(span=span, adjust=adjust).mean()


def wma(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    计算加权移动平均线 (WMA)
    权重与数据点的位置成线性关系，越近的数据点权重越大

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        window: 窗口大小

    返回:
        pandas Series包含WMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    weights = np.arange(1, window+1)
    return data.rolling(window=window).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)


def smma(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    计算平滑移动平均线 (SMMA)，也称为Modified Moving Average

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        window: 窗口大小

    返回:
        pandas Series包含SMMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # SMMA可以用EMA来近似，参数alpha=1/N
    return data.ewm(alpha=1/window, adjust=False).mean()


def tema(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    计算三重指数平滑移动平均线 (TEMA)
    TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
    其中EMA1是原始数据的EMA，EMA2是EMA1的EMA，EMA3是EMA2的EMA

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        window: 窗口大小

    返回:
        pandas Series包含TEMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    ema1 = data.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    return 3 * ema1 - 3 * ema2 + ema3


def kama(data: Union[pd.Series, np.ndarray], 
         er_window: int = 10, 
         fast_span: int = 2, 
         slow_span: int = 30) -> pd.Series:
    """
    计算考夫曼自适应移动平均线 (KAMA)

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        er_window: 效率比率的窗口大小
        fast_span: 快速EMA的跨度
        slow_span: 慢速EMA的跨度

    返回:
        pandas Series包含KAMA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # 计算方向变化
    change = abs(data - data.shift(er_window))
    
    # 计算波动
    volatility = data.diff().abs().rolling(window=er_window).sum()
    
    # 计算效率比率 (ER)
    er = pd.Series(np.zeros(len(data)), index=data.index)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # 计算平滑常数
    fast_alpha = 2 / (fast_span + 1)
    slow_alpha = 2 / (slow_span + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    # 计算KAMA
    kama = pd.Series(np.zeros(len(data)), index=data.index)
    kama.iloc[0] = data.iloc[0]  # 初始值
    
    for i in range(1, len(data)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
    
    return kama


def hull_ma(data: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
    """
    计算赫尔移动平均线 (Hull Moving Average)
    Hull MA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))

    参数:
        data: 原始数据，可以是pandas Series或numpy数组
        window: 窗口大小

    返回:
        pandas Series包含Hull MA值
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))
    
    wma1 = wma(data, half_window)
    wma2 = wma(data, window)
    wma_diff = 2 * wma1 - wma2
    hull = wma(wma_diff, sqrt_window)
    
    return hull 