"""
震荡指标模块

提供各种震荡指标的计算，包括随机指标、威廉指标、CCI等。
这些指标主要用于识别超买超卖区域和可能的价格反转点。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                          k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Dict[str, pd.Series]:
    """
    计算随机指标(%K和%D)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: %K周期
        d_period: %D周期
        smooth_k: %K平滑周期
        
    返回:
        包含%K和%D的字典
    """
    # 计算最低价的最低值
    low_min = low.rolling(window=k_period).min()
    
    # 计算最高价的最高值
    high_max = high.rolling(window=k_period).max()
    
    # 计算原始%K
    k_raw = 100 * ((close - low_min) / (high_max - low_min))
    
    # 平滑%K
    k = k_raw.rolling(window=smooth_k).mean() if smooth_k > 1 else k_raw
    
    # 计算%D
    d = k.rolling(window=d_period).mean()
    
    return {'k': k, 'd': d}


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算威廉指标(%R)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
        
    返回:
        %R值的序列
    """
    # 计算最高价的最高值
    high_max = high.rolling(window=period).max()
    
    # 计算最低价的最低值
    low_min = low.rolling(window=period).min()
    
    # 计算威廉%R
    wr = -100 * ((high_max - close) / (high_max - low_min))
    
    return wr


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """
    计算变动率指标(ROC)
    
    参数:
        close: 收盘价序列
        period: 计算周期
        
    返回:
        ROC值的序列
    """
    # 计算ROC
    roc_values = ((close / close.shift(period)) - 1) * 100
    
    return roc_values


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, constant: float = 0.015) -> pd.Series:
    """
    计算顺势指标(CCI)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
        constant: 常数（通常为0.015）
        
    返回:
        CCI值的序列
    """
    # 计算典型价格
    tp = (high + low + close) / 3
    
    # 计算移动平均
    tp_ma = tp.rolling(window=period).mean()
    
    # 计算平均偏差
    tp_md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    # 计算CCI
    cci_values = (tp - tp_ma) / (constant * tp_md)
    
    return cci_values


def awesome_oscillator(high: pd.Series, low: pd.Series, short_period: int = 5, long_period: int = 34) -> pd.Series:
    """
    计算动量震荡指标(AO)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        short_period: 短期周期
        long_period: 长期周期
        
    返回:
        AO值的序列
    """
    # 计算中点价格
    median_price = (high + low) / 2
    
    # 计算短期SMA
    short_sma = median_price.rolling(window=short_period).mean()
    
    # 计算长期SMA
    long_sma = median_price.rolling(window=long_period).mean()
    
    # 计算AO
    ao = short_sma - long_sma
    
    return ao


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period1: int = 7, period2: int = 14, period3: int = 28, 
                        weight1: float = 4, weight2: float = 2, weight3: float = 1) -> pd.Series:
    """
    计算终极震荡器(UO)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period1: 第一周期
        period2: 第二周期
        period3: 第三周期
        weight1: 第一周期权重
        weight2: 第二周期权重
        weight3: 第三周期权重
        
    返回:
        UO值的序列
    """
    # 计算买入压力
    buying_pressure = close - pd.DataFrame({
        'close_shift': close.shift(1),
        'low': low
    }).min(axis=1)
    
    # 计算真实范围
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    # 计算各周期的平均值
    avg1 = buying_pressure.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = buying_pressure.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = buying_pressure.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    # 处理除以0的情况
    avg1.replace([np.inf, -np.inf], 0, inplace=True)
    avg2.replace([np.inf, -np.inf], 0, inplace=True)
    avg3.replace([np.inf, -np.inf], 0, inplace=True)
    
    # 计算UO
    total_weight = weight1 + weight2 + weight3
    uo = 100 * ((weight1 * avg1 + weight2 * avg2 + weight3 * avg3) / total_weight)
    
    return uo


def stochastic_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, 
                  k_period: int = 3, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    计算随机RSI指标
    
    参数:
        close: 收盘价序列
        rsi_period: RSI计算周期
        stoch_period: 随机指标计算周期
        k_period: %K平滑周期
        d_period: %D平滑周期
        
    返回:
        包含随机RSI的%K和%D的字典
    """
    # 计算RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # 计算随机RSI
    stoch_rsi_k = 100 * (rsi - rsi.rolling(window=stoch_period).min()) / \
                 (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
    
    # 平滑%K
    stoch_rsi_k = stoch_rsi_k.rolling(window=k_period).mean()
    
    # 计算%D
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
    
    return {'k': stoch_rsi_k, 'd': stoch_rsi_d}


def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """
    计算三重指数平滑平均(TRIX)指标
    
    参数:
        close: 收盘价序列
        period: 计算周期
        
    返回:
        TRIX值的序列
    """
    # 第一次EMA
    ema1 = close.ewm(span=period, adjust=False).mean()
    
    # 第二次EMA
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    
    # 第三次EMA
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # 计算TRIX
    trix_values = (ema3 / ema3.shift(1) - 1) * 100
    
    return trix_values


def true_strength_index(close: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
    """
    计算真实强度指数(TSI)
    
    参数:
        close: 收盘价序列
        long_period: 长期EMA周期
        short_period: 短期EMA周期
        
    返回:
        TSI值的序列
    """
    # 计算价格变化
    momentum = close.diff()
    
    # 第一次平滑 - 长期EMA
    long_ema_momentum = momentum.ewm(span=long_period, adjust=False).mean()
    long_ema_abs_momentum = abs(momentum).ewm(span=long_period, adjust=False).mean()
    
    # 第二次平滑 - 短期EMA
    double_ema_momentum = long_ema_momentum.ewm(span=short_period, adjust=False).mean()
    double_ema_abs_momentum = long_ema_abs_momentum.ewm(span=short_period, adjust=False).mean()
    
    # 计算TSI
    tsi = (double_ema_momentum / double_ema_abs_momentum) * 100
    
    return tsi


def ppo(close: pd.Series, fast_period: int = 12, slow_period: int = 26, 
        signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    计算百分比价格震荡指标(PPO)
    
    参数:
        close: 收盘价序列
        fast_period: 快速EMA周期
        slow_period: 慢速EMA周期
        signal_period: 信号线EMA周期
        
    返回:
        包含PPO线、信号线和柱状图的字典
    """
    # 计算快速EMA和慢速EMA
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    # 计算PPO线
    ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100
    
    # 计算信号线
    signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
    
    # 计算柱状图
    histogram = ppo_line - signal_line
    
    return {
        'ppo': ppo_line,
        'signal': signal_line,
        'histogram': histogram
    } 