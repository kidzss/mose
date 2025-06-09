"""
波动率指标模块

提供各种波动率相关的技术指标计算，包括ATR、Bollinger带宽、历史波动率等。
这些指标有助于识别市场的波动性和潜在的趋势变化点。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    计算平均真实范围(ATR)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期
        
    返回:
        ATR值的序列
    """
    # 计算三种真实范围
    tr1 = high - low  # 当日高点 - 当日低点
    tr2 = abs(high - close.shift())  # 当日高点 - 前一日收盘价
    tr3 = abs(low - close.shift())  # 当日低点 - 前一日收盘价
    
    # 取三者中的最大值作为真实范围
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # 计算ATR
    atr_values = tr.rolling(window=window).mean()
    
    return atr_values


def average_range(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """
    计算平均波动范围(Average Range)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        window: 计算周期
        
    返回:
        平均波动范围的序列
    """
    # 计算日内波动范围
    daily_range = high - low
    
    # 计算平均波动范围
    avg_range = daily_range.rolling(window=window).mean()
    
    return avg_range


def historical_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    计算历史波动率
    
    参数:
        close: 收盘价序列
        window: 计算周期
        annualize: 是否年化（默认为True）
        
    返回:
        历史波动率序列
    """
    # 计算对数收益率
    log_returns = np.log(close / close.shift(1))
    
    # 计算标准差
    std = log_returns.rolling(window=window).std()
    
    # 如果需要年化，乘以sqrt(252)（假设一年有252个交易日）
    if annualize:
        std = std * np.sqrt(252)
        
    return std


def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    计算布林带宽度
    
    参数:
        close: 收盘价序列
        window: 移动平均周期
        num_std: 标准差倍数
        
    返回:
        布林带宽度序列
    """
    # 计算简单移动平均
    middle = close.rolling(window=window).mean()
    
    # 计算标准差
    std = close.rolling(window=window).std()
    
    # 计算上轨和下轨
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # 计算带宽
    bandwidth = (upper - lower) / middle * 100
    
    return bandwidth


def keltner_channel_width(high: pd.Series, 
                          low: pd.Series, 
                          close: pd.Series, 
                          window: int = 20, 
                          atr_mult: float = 2.0) -> pd.Series:
    """
    计算Keltner通道宽度
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 移动平均周期
        atr_mult: ATR乘数
        
    返回:
        Keltner通道宽度序列
    """
    # 计算EMA
    middle = close.ewm(span=window, adjust=False).mean()
    
    # 计算ATR
    atr_val = atr(high, low, close, window)
    
    # 计算上轨和下轨
    upper = middle + (atr_val * atr_mult)
    lower = middle - (atr_val * atr_mult)
    
    # 计算通道宽度
    channel_width = (upper - lower) / middle * 100
    
    return channel_width


def volatility_ratio(close: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """
    计算短期/长期波动率比率
    
    参数:
        close: 收盘价序列
        short_window: 短期窗口
        long_window: 长期窗口
        
    返回:
        波动率比率序列
    """
    # 计算短期波动率
    short_vol = historical_volatility(close, short_window, False)
    
    # 计算长期波动率
    long_vol = historical_volatility(close, long_window, False)
    
    # 计算比率
    vol_ratio = short_vol / long_vol
    
    return vol_ratio


def chaikin_volatility(high: pd.Series, 
                       low: pd.Series, 
                       ema_period: int = 10, 
                       change_period: int = 10) -> pd.Series:
    """
    计算Chaikin波动率指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        ema_period: EMA周期
        change_period: 变化率计算周期
        
    返回:
        Chaikin波动率指标序列
    """
    # 计算高低价差
    hl_range = high - low
    
    # 计算高低价差的EMA
    ema_hl = hl_range.ewm(span=ema_period, adjust=False).mean()
    
    # 计算波动率变化率
    chaikin_vol = ((ema_hl - ema_hl.shift(change_period)) / ema_hl.shift(change_period)) * 100
    
    return chaikin_vol


def garman_klass_volatility(open_price: pd.Series, 
                           high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series, 
                           window: int = 20, 
                           annualize: bool = True) -> pd.Series:
    """
    计算Garman-Klass波动率
    
    参数:
        open_price: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期
        annualize: 是否年化
        
    返回:
        Garman-Klass波动率序列
    """
    # 计算对数高低价比率
    log_hl = np.log(high / low) ** 2
    
    # 计算对数收盘开盘价比率
    log_co = np.log(close / open_price) ** 2
    
    # Garman-Klass公式
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    
    # 滚动窗口计算波动率
    gk_vol = np.sqrt(gk.rolling(window=window).mean())
    
    # 年化
    if annualize:
        gk_vol = gk_vol * np.sqrt(252)
        
    return gk_vol


def ulcer_index(close: pd.Series, window: int = 14) -> pd.Series:
    """
    计算Ulcer指数(UI)，用于衡量下行波动风险
    
    参数:
        close: 收盘价序列
        window: 计算周期
        
    返回:
        Ulcer指数序列
    """
    # 计算周期内的最高价
    roll_max = close.rolling(window=window).max()
    
    # 计算当前价格与周期内最高价的百分比回撤
    pct_drawdown = ((close - roll_max) / roll_max) * 100
    
    # 平方回撤
    squared_dd = pct_drawdown ** 2
    
    # 计算Ulcer指数
    ui = np.sqrt(squared_dd.rolling(window=window).mean())
    
    return ui 