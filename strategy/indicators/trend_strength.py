"""
趋势强度指标模块

提供各种趋势强度相关的技术指标计算，包括ADX、Aroon、DPO等。
这些指标有助于识别市场趋势的强度、持续性和可能的变化。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Tuple


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
    """
    计算平均方向指标(ADX)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算周期
        
    返回:
        包含ADX、+DI和-DI的字典
    """
    # 计算真实范围
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    
    # 计算+DM和-DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # 计算平滑后的TR、+DM和-DM
    tr_smooth = pd.Series(tr).rolling(window=window).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=window).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=window).mean()
    
    # 计算方向指标
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # 计算方向指数差异
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0)  # 处理除以0的情况
    
    # 计算ADX
    adx_value = dx.rolling(window=window).mean()
    
    return {
        'adx': adx_value,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Dict[str, pd.Series]:
    """
    计算Aroon指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
        
    返回:
        包含Aroon上升、下降和振荡器的字典
    """
    aroon_up = pd.Series(0, index=high.index)
    aroon_down = pd.Series(0, index=low.index)
    
    for i in range(period, len(high)):
        # 计算最高价出现的位置
        high_window = high[i-period+1:i+1]
        high_idx = high_window.argmax()
        
        # 计算最低价出现的位置
        low_window = low[i-period+1:i+1]
        low_idx = low_window.argmin()
        
        # 计算Aroon上升和下降
        aroon_up.iloc[i] = ((period - high_idx) / period) * 100
        aroon_down.iloc[i] = ((period - low_idx) / period) * 100
    
    # 计算Aroon振荡器
    aroon_osc = aroon_up - aroon_down
    
    return {
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'aroon_osc': aroon_osc
    }


def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    计算Vortex指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
        
    返回:
        包含VI+和VI-的字典
    """
    # 计算价格波动
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    
    # 计算真实范围
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    # 计算滚动和
    vm_plus_sum = vm_plus.rolling(window=period).sum()
    vm_minus_sum = vm_minus.rolling(window=period).sum()
    tr_sum = tr.rolling(window=period).sum()
    
    # 计算Vortex指标
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    return {
        'vi_plus': vi_plus,
        'vi_minus': vi_minus
    }


def dmi_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算DMI振荡器
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
        
    返回:
        DMI振荡器序列
    """
    # 计算ADX、+DI和-DI
    adx_result = adx(high, low, close, period)
    
    # 计算DMI振荡器
    dmi_osc = adx_result['plus_di'] - adx_result['minus_di']
    
    return dmi_osc


def directional_movement_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                              period: int = 14, adx_smoothing: int = 14) -> Dict[str, pd.Series]:
    """
    计算方向运动指数(DMI)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
        adx_smoothing: ADX平滑周期
        
    返回:
        包含DX、ADX和ADXR的字典
    """
    # 计算ADX和DI
    adx_data = adx(high, low, close, period)
    
    # 计算方向指数
    dx = 100 * abs(adx_data['plus_di'] - adx_data['minus_di']) / (adx_data['plus_di'] + adx_data['minus_di'])
    dx = dx.replace([np.inf, -np.inf], 0)  # 处理除以0的情况
    
    # 计算ADX
    adx_value = dx.rolling(window=adx_smoothing).mean()
    
    # 计算ADXR（ADX的平滑）
    adxr = (adx_value + adx_value.shift(adx_smoothing)) / 2
    
    return {
        'dx': dx,
        'adx': adx_value,
        'adxr': adxr
    }


def detrended_price_oscillator(close: pd.Series, period: int = 20) -> pd.Series:
    """
    计算去趋势价格振荡器(DPO)
    
    参数:
        close: 收盘价序列
        period: 计算周期
        
    返回:
        DPO序列
    """
    # 计算移动平均
    ma = close.rolling(window=period).mean()
    
    # 计算DPO
    shift_period = period // 2 + 1
    dpo = close - ma.shift(shift_period)
    
    return dpo


def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, 
                 af_max: float = 0.2) -> pd.Series:
    """
    计算抛物线转向指标(SAR)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        af_start: 初始加速因子
        af_increment: 加速因子增量
        af_max: 最大加速因子
        
    返回:
        SAR序列
    """
    # 初始化SAR序列
    sar = pd.Series(0.0, index=high.index)
    
    # 如果数据不足，返回空序列
    if len(high) < 2:
        return sar
    
    # 确定初始趋势
    trend = 1 if high.iloc[1] > high.iloc[0] else -1
    
    # 设置初始值
    extreme_point = high.iloc[1] if trend == 1 else low.iloc[1]
    sar.iloc[1] = low.iloc[0] if trend == 1 else high.iloc[0]
    
    # 当前加速因子
    af = af_start
    
    # 计算SAR
    for i in range(2, len(high)):
        # 更新SAR
        sar.iloc[i] = sar.iloc[i-1] + af * (extreme_point - sar.iloc[i-1])
        
        # 确保SAR不会突破价格
        if trend == 1:  # 上升趋势
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
            # 检查是否翻转
            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = extreme_point
                extreme_point = low.iloc[i]
                af = af_start
            else:
                # 更新极值点和加速因子
                if high.iloc[i] > extreme_point:
                    extreme_point = high.iloc[i]
                    af = min(af + af_increment, af_max)
        else:  # 下降趋势
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
            # 检查是否翻转
            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = extreme_point
                extreme_point = high.iloc[i]
                af = af_start
            else:
                # 更新极值点和加速因子
                if low.iloc[i] < extreme_point:
                    extreme_point = low.iloc[i]
                    af = min(af + af_increment, af_max)
    
    return sar


def trend_intensity_index(close: pd.Series, period: int = 20) -> pd.Series:
    """
    计算趋势强度指数(TII)
    
    参数:
        close: 收盘价序列
        period: 计算周期
        
    返回:
        TII序列
    """
    # 计算移动平均
    ma = close.rolling(window=period).mean()
    
    # 计算价格与均线差值
    diff = close - ma
    
    # 计算正差值和负差值的平方和
    pos_sq_sum = (diff[diff > 0] ** 2).rolling(window=period).sum().fillna(0)
    neg_sq_sum = (diff[diff < 0] ** 2).rolling(window=period).sum().fillna(0)
    
    # 计算总平方和
    total_sq_sum = pos_sq_sum + neg_sq_sum
    
    # 计算TII
    tii = 100 * pos_sq_sum / total_sq_sum
    tii = tii.replace([np.inf, -np.inf], 50)  # 处理除以0的情况
    
    return tii


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
              period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
    """
    计算Supertrend指标
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR计算周期
        multiplier: ATR乘数
        
    返回:
        包含Supertrend线、方向和趋势的字典
    """
    # 计算ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # 计算基础上下轨
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    
    # 初始化最终上下轨和SuperTrend
    final_upper = pd.Series(0.0, index=close.index)
    final_lower = pd.Series(0.0, index=close.index)
    supertrend = pd.Series(0.0, index=close.index)
    trend = pd.Series(1, index=close.index)  # 1表示上升趋势，-1表示下降趋势
    
    # 设置初始值
    if len(close) >= period:
        final_upper.iloc[period-1] = basic_upper.iloc[period-1]
        final_lower.iloc[period-1] = basic_lower.iloc[period-1]
        supertrend.iloc[period-1] = final_upper.iloc[period-1] if close.iloc[period-1] <= final_upper.iloc[period-1] else final_lower.iloc[period-1]
        trend.iloc[period-1] = 1 if close.iloc[period-1] > final_upper.iloc[period-1] else -1
    
    # 计算SuperTrend
    for i in range(period, len(close)):
        # 更新最终上轨
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        # 更新最终下轨
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # 更新趋势
        if supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        
        # 更新SuperTrend
        if trend.iloc[i] == 1:
            supertrend.iloc[i] = final_lower.iloc[i]
        else:
            supertrend.iloc[i] = final_upper.iloc[i]
    
    return {
        'supertrend': supertrend,
        'trend': trend,
        'upper': final_upper,
        'lower': final_lower
    } 