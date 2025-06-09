"""
ADX方向指标模块

提供ADX(平均方向指数)及相关方向指标的计算函数，用于技术分析和交易策略。
ADX指标用于衡量趋势的强度，而不是趋势的方向。
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional


def adx(high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        window: int = 14) -> Dict[str, pd.Series]:
    """
    计算平均方向指数(ADX)及相关方向指标

    参数:
        high: 最高价数据
        low: 最低价数据
        close: 收盘价数据
        window: 窗口大小，默认为14

    返回:
        包含ADX、+DI、-DI的字典
    """
    # 计算方向变动
    pos_dm = high.diff()
    neg_dm = low.diff()
    
    # 正方向变动: 当前最高价高于前一天的最高价，且高于前一天的最高价与当前最低价之差
    pos_dm = pd.Series(np.where((pos_dm > 0) & (pos_dm > neg_dm.abs()), pos_dm, 0), index=pos_dm.index)
    
    # 负方向变动: 当前最低价低于前一天的最低价，且低于前一天的最低价与当前最高价之差
    neg_dm = pd.Series(np.where((neg_dm < 0) & (neg_dm.abs() > pos_dm), neg_dm.abs(), 0), index=neg_dm.index)
    
    # 计算真实波幅(TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算平滑的方向变动和真实波幅
    smooth_pos_dm = pos_dm.rolling(window=window).sum()
    smooth_neg_dm = neg_dm.rolling(window=window).sum()
    smooth_tr = tr.rolling(window=window).sum()
    
    # 计算方向指标
    pos_di = 100 * smooth_pos_dm / smooth_tr
    neg_di = 100 * smooth_neg_dm / smooth_tr
    
    # 计算方向指标差异
    di_diff = abs(pos_di - neg_di)
    di_sum = pos_di + neg_di
    
    # 计算方向指数
    dx = 100 * di_diff / di_sum
    
    # 计算平均方向指数
    adx_value = dx.rolling(window=window).mean()
    
    return {
        'adx': adx_value,
        'plus_di': pos_di,
        'minus_di': neg_di
    }


def adx_trend_strength(adx_value: pd.Series) -> pd.Series:
    """
    根据ADX值判断趋势强度

    参数:
        adx_value: ADX值

    返回:
        趋势强度分类: 
        0 = 无趋势或非常弱的趋势 (ADX < 20)
        1 = 弱趋势 (20 <= ADX < 25)
        2 = 强趋势 (25 <= ADX < 40)
        3 = 非常强趋势 (40 <= ADX < 50)
        4 = 极端强趋势 (ADX >= 50)
    """
    strength = pd.Series(np.zeros(len(adx_value)), index=adx_value.index)
    
    strength[(adx_value >= 20) & (adx_value < 25)] = 1
    strength[(adx_value >= 25) & (adx_value < 40)] = 2
    strength[(adx_value >= 40) & (adx_value < 50)] = 3
    strength[adx_value >= 50] = 4
    
    return strength


def adx_trend_direction(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """
    根据+DI和-DI判断趋势方向

    参数:
        plus_di: +DI值
        minus_di: -DI值

    返回:
        趋势方向: 
        1 = 上升趋势 (+DI > -DI)
        -1 = 下降趋势 (-DI > +DI)
        0 = 无明确趋势 (+DI ≈ -DI)
    """
    direction = pd.Series(np.zeros(len(plus_di)), index=plus_di.index)
    
    # 上升趋势
    direction[plus_di > minus_di] = 1
    
    # 下降趋势
    direction[minus_di > plus_di] = -1
    
    return direction


def adx_crossover(plus_di: pd.Series, minus_di: pd.Series) -> pd.DataFrame:
    """
    检测+DI和-DI的交叉信号

    参数:
        plus_di: +DI值
        minus_di: -DI值

    返回:
        包含看涨和看跌交叉信号的DataFrame
    """
    # 看涨交叉 (+DI从下方穿过-DI)
    bullish_cross = ((plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))).astype(int)
    
    # 看跌交叉 (-DI从下方穿过+DI)
    bearish_cross = ((minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))).astype(int)
    
    return pd.DataFrame({
        'bullish_cross': bullish_cross,
        'bearish_cross': bearish_cross
    })


def adx_reversal(adx_value: pd.Series, threshold: float = 3.0, window: int = 3) -> pd.DataFrame:
    """
    检测ADX反转信号，通常指示趋势可能即将改变

    参数:
        adx_value: ADX值
        threshold: ADX反转的最小变化阈值
        window: 用于判断ADX趋势的窗口大小

    返回:
        包含ADX反转信号的DataFrame
    """
    # 计算ADX的变化率
    adx_change = adx_value.diff(window)
    
    # ADX从上升转为下降
    adx_peak = ((adx_value.diff() < 0) & (adx_value.diff().shift(1) >= 0) & (adx_change < -threshold)).astype(int)
    
    # ADX从下降转为上升
    adx_bottom = ((adx_value.diff() > 0) & (adx_value.diff().shift(1) <= 0) & (adx_change > threshold)).astype(int)
    
    return pd.DataFrame({
        'adx_peak': adx_peak,
        'adx_bottom': adx_bottom
    })


def dmi_oscillator(plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    """
    计算方向运动指标振荡器(DMI Oscillator)
    
    参数:
        plus_di: +DI值
        minus_di: -DI值
        
    返回:
        DMI振荡器值 (+DI - -DI)
    """
    return plus_di - minus_di 