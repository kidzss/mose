"""
成交量指标模块

提供各种成交量相关的技术指标计算，包括成交量均线、OBV、CMF等。
这些指标有助于确认价格趋势的有效性和潜在的反转信号。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    计算成交量简单移动平均
    
    参数:
        volume: 成交量序列
        window: 计算周期
        
    返回:
        成交量SMA序列
    """
    return volume.rolling(window=window).mean()


def volume_ema(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    计算成交量指数移动平均
    
    参数:
        volume: 成交量序列
        window: 计算周期
        
    返回:
        成交量EMA序列
    """
    return volume.ewm(span=window, adjust=False).mean()


def volume_ratio(volume: pd.Series, ma_window: int = 20) -> pd.Series:
    """
    计算成交量比率（当前成交量/平均成交量）
    
    参数:
        volume: 成交量序列
        ma_window: 移动平均周期
        
    返回:
        成交量比率序列
    """
    vol_ma = volume_sma(volume, ma_window)
    return volume / vol_ma


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算能量潮指标(OBV)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
        
    返回:
        OBV序列
    """
    obv = pd.Series(0, index=close.index)
    
    # 第一个值设为成交量
    if not obv.empty:
        obv.iloc[0] = volume.iloc[0]
    
    # 计算后续值
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, window: int = 20) -> pd.Series:
    """
    计算蔡金资金流量指标(CMF)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        window: 计算周期
        
    返回:
        CMF序列
    """
    # 计算资金流量乘数
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0)  # 处理除以0的情况
    
    # 计算资金流量量
    mfv = mfm * volume
    
    # 计算CMF
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    
    return cmf


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                     volume: pd.Series, window: int = 14) -> pd.Series:
    """
    计算资金流量指标(MFI)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        window: 计算周期
        
    返回:
        MFI序列
    """
    # 计算典型价格
    tp = (high + low + close) / 3
    
    # 计算原始资金流
    raw_money_flow = tp * volume
    
    # 计算正向和负向资金流
    money_flow_pos = pd.Series(0, index=tp.index)
    money_flow_neg = pd.Series(0, index=tp.index)
    
    for i in range(1, len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:  # 价格上涨
            money_flow_pos.iloc[i] = raw_money_flow.iloc[i]
        else:  # 价格下跌或不变
            money_flow_neg.iloc[i] = raw_money_flow.iloc[i]
    
    # 计算资金比率
    positive_flow = money_flow_pos.rolling(window=window).sum()
    negative_flow = money_flow_neg.rolling(window=window).sum()
    
    # 处理除以0的情况
    money_ratio = pd.Series(1, index=positive_flow.index)
    non_zero_mask = negative_flow != 0
    money_ratio[non_zero_mask] = positive_flow[non_zero_mask] / negative_flow[non_zero_mask]
    
    # 计算MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算成交量价格趋势指标(VPT)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
        
    返回:
        VPT序列
    """
    # 计算价格变化率
    pct_change = close.pct_change()
    
    # 计算VPT
    vpt = pd.Series(0, index=close.index)
    
    if not vpt.empty:
        vpt.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * pct_change.iloc[i]
    
    return vpt


def negative_volume_index(close: pd.Series, volume: pd.Series, base_value: float = 1000) -> pd.Series:
    """
    计算负成交量指标(NVI)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
        base_value: 初始值
        
    返回:
        NVI序列
    """
    # 初始化NVI
    nvi = pd.Series(base_value, index=close.index)
    
    # 计算NVI
    for i in range(1, len(volume)):
        if volume.iloc[i] < volume.iloc[i-1]:  # 只在成交量下降时更新
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + close.pct_change().iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]
    
    return nvi


def positive_volume_index(close: pd.Series, volume: pd.Series, base_value: float = 1000) -> pd.Series:
    """
    计算正成交量指标(PVI)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
        base_value: 初始值
        
    返回:
        PVI序列
    """
    # 初始化PVI
    pvi = pd.Series(base_value, index=close.index)
    
    # 计算PVI
    for i in range(1, len(volume)):
        if volume.iloc[i] > volume.iloc[i-1]:  # 只在成交量上升时更新
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + close.pct_change().iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]
    
    return pvi


def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, divisor: float = 10000) -> pd.Series:
    """
    计算简易波动指标(EMV)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        divisor: 除数（调整比例）
        
    返回:
        EMV序列
    """
    # 计算中间价的移动
    mid_point_move = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    
    # 计算高低价差
    box_ratio = (volume / divisor) / (high - low)
    
    # 处理除以0的情况
    box_ratio = box_ratio.replace([np.inf, -np.inf], 0)
    
    # 计算EMV
    emv = mid_point_move / box_ratio
    
    return emv


def volume_oscillator(volume: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """
    计算成交量振荡器
    
    参数:
        volume: 成交量序列
        short_window: 短期移动平均周期
        long_window: 长期移动平均周期
        
    返回:
        成交量振荡器序列
    """
    # 计算短期和长期移动平均
    short_ma = volume.rolling(window=short_window).mean()
    long_ma = volume.rolling(window=long_window).mean()
    
    # 计算振荡器（百分比形式）
    oscillator = ((short_ma - long_ma) / long_ma) * 100
    
    return oscillator


def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    计算积累/分配线(A/D Line)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        
    返回:
        A/D Line序列
    """
    # 计算资金流量乘数
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0)  # 处理除以0的情况
    
    # 计算资金流量量
    mfv = mfm * volume
    
    # 计算A/D线
    ad_line = mfv.cumsum()
    
    return ad_line 