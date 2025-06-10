"""
MACD指标模块

提供MACD及相关指标的计算函数，用于技术分析和交易策略。
MACD(Moving Average Convergence Divergence)是一种趋势跟踪动量指标，显示两条移动平均线之间的关系。
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional


def macd(data: Union[pd.Series, np.ndarray], 
         fast_period: int = 12, 
         slow_period: int = 26, 
         signal_period: int = 9, 
         adjust: bool = False) -> Dict[str, pd.Series]:
    """
    计算MACD指标

    参数:
        data: 原始价格数据，通常是收盘价
        fast_period: 快线周期，默认为12
        slow_period: 慢线周期，默认为26
        signal_period: 信号线周期，默认为9
        adjust: 是否调整EMA计算，默认为False

    返回:
        包含MACD线、信号线和柱状图的字典
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # 计算快线和慢线
    ema_fast = data.ewm(span=fast_period, adjust=adjust).mean()
    ema_slow = data.ewm(span=slow_period, adjust=adjust).mean()
    
    # 计算MACD线 (快线 - 慢线)
    macd_line = ema_fast - ema_slow
    
    # 计算信号线 (MACD的EMA)
    signal_line = macd_line.ewm(span=signal_period, adjust=adjust).mean()
    
    # 计算柱状图 (MACD线 - 信号线)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def macd_crossover(macd_line: pd.Series, signal_line: pd.Series) -> pd.DataFrame:
    """
    检测MACD与信号线的交叉

    参数:
        macd_line: MACD线
        signal_line: 信号线
        
    返回:
        包含金叉和死叉信号的DataFrame
    """
    # 金叉信号 (MACD线从下方穿过信号线)
    golden_cross = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    
    # 死叉信号 (MACD线从上方穿过信号线)
    death_cross = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    
    return pd.DataFrame({
        'golden_cross': golden_cross,
        'death_cross': death_cross
    })


def macd_zero_crossover(macd_line: pd.Series) -> pd.DataFrame:
    """
    检测MACD线与零轴的交叉

    参数:
        macd_line: MACD线
        
    返回:
        包含零轴上穿和下穿信号的DataFrame
    """
    # 零轴上穿信号 (MACD线从下方穿过零轴)
    zero_cross_up = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(int)
    
    # 零轴下穿信号 (MACD线从上方穿过零轴)
    zero_cross_down = ((macd_line < 0) & (macd_line.shift(1) >= 0)).astype(int)
    
    return pd.DataFrame({
        'zero_cross_up': zero_cross_up,
        'zero_cross_down': zero_cross_down
    })


def macd_divergence(price: pd.Series, 
                    macd_line: pd.Series, 
                    window: int = 5,
                    threshold: float = 2.0) -> pd.DataFrame:
    """
    检测MACD与价格之间的背离

    参数:
        price: 价格数据
        macd_line: MACD线
        window: 检测局部极值的窗口大小
        threshold: 用于确定显著背离的阈值
        
    返回:
        包含看涨和看跌背离信号的DataFrame
    """
    # 找出价格的局部极值
    price_highs = pd.Series(np.zeros_like(price), index=price.index)
    price_lows = pd.Series(np.zeros_like(price), index=price.index)
    
    for i in range(window, len(price) - window):
        # 如果当前价格是窗口内的最高点
        if price.iloc[i] == price.iloc[i-window:i+window+1].max():
            price_highs.iloc[i] = 1
        
        # 如果当前价格是窗口内的最低点
        if price.iloc[i] == price.iloc[i-window:i+window+1].min():
            price_lows.iloc[i] = 1
    
    # 找出MACD的局部极值
    macd_highs = pd.Series(np.zeros_like(macd_line), index=macd_line.index)
    macd_lows = pd.Series(np.zeros_like(macd_line), index=macd_line.index)
    
    for i in range(window, len(macd_line) - window):
        # 如果当前MACD是窗口内的最高点
        if macd_line.iloc[i] == macd_line.iloc[i-window:i+window+1].max():
            macd_highs.iloc[i] = 1
        
        # 如果当前MACD是窗口内的最低点
        if macd_line.iloc[i] == macd_line.iloc[i-window:i+window+1].min():
            macd_lows.iloc[i] = 1
    
    # 初始化背离信号
    bullish_divergence = pd.Series(np.zeros_like(price), index=price.index)
    bearish_divergence = pd.Series(np.zeros_like(price), index=price.index)
    
    # 查找看涨背离: 价格创新低但MACD不创新低
    for i in range(window*2, len(price)):
        if price_lows.iloc[i] == 1:
            # 寻找前一个价格低点
            for j in range(i-window, max(0, i-window*5), -1):
                if price_lows.iloc[j] == 1:
                    # 如果当前价格更低但MACD更高，则是看涨背离
                    if (price.iloc[i] < price.iloc[j]) and (macd_line.iloc[i] > macd_line.iloc[j]):
                        # 计算背离强度
                        price_change = (price.iloc[i] / price.iloc[j] - 1) * 100
                        macd_change = macd_line.iloc[i] - macd_line.iloc[j]
                        
                        # 只有显著背离才记录
                        if abs(price_change) + abs(macd_change) > threshold:
                            bullish_divergence.iloc[i] = 1
                    break
    
    # 查找看跌背离: 价格创新高但MACD不创新高
    for i in range(window*2, len(price)):
        if price_highs.iloc[i] == 1:
            # 寻找前一个价格高点
            for j in range(i-window, max(0, i-window*5), -1):
                if price_highs.iloc[j] == 1:
                    # 如果当前价格更高但MACD更低，则是看跌背离
                    if (price.iloc[i] > price.iloc[j]) and (macd_line.iloc[i] < macd_line.iloc[j]):
                        # 计算背离强度
                        price_change = (price.iloc[i] / price.iloc[j] - 1) * 100
                        macd_change = macd_line.iloc[j] - macd_line.iloc[i]
                        
                        # 只有显著背离才记录
                        if abs(price_change) + abs(macd_change) > threshold:
                            bearish_divergence.iloc[i] = 1
                    break
    
    return pd.DataFrame({
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence
    })


def macd_histogram_reversal(histogram: pd.Series, window: int = 3) -> pd.DataFrame:
    """
    检测MACD柱状图反转信号
    
    参数:
        histogram: MACD柱状图
        window: 连续柱状图趋势的最小长度
        
    返回:
        包含柱状图反转信号的DataFrame
    """
    # 计算柱状图的变化
    hist_change = histogram.diff()
    
    # 连续增长的柱状图转为下降
    bullish_exhaustion = pd.Series(np.zeros_like(histogram), index=histogram.index)
    
    # 连续下降的柱状图转为增长
    bearish_exhaustion = pd.Series(np.zeros_like(histogram), index=histogram.index)
    
    for i in range(window + 1, len(histogram)):
        # 检查之前的window个柱状图是否连续增长
        prev_increases = True
        for j in range(i-window, i):
            if hist_change.iloc[j] <= 0:
                prev_increases = False
                break
        
        # 如果之前连续增长且当前下降，则是上升动能耗尽
        if prev_increases and hist_change.iloc[i] < 0:
            bullish_exhaustion.iloc[i] = 1
        
        # 检查之前的window个柱状图是否连续下降
        prev_decreases = True
        for j in range(i-window, i):
            if hist_change.iloc[j] >= 0:
                prev_decreases = False
                break
        
        # 如果之前连续下降且当前增长，则是下降动能耗尽
        if prev_decreases and hist_change.iloc[i] > 0:
            bearish_exhaustion.iloc[i] = 1
    
    return pd.DataFrame({
        'bullish_exhaustion': bullish_exhaustion,
        'bearish_exhaustion': bearish_exhaustion
    })


def ppo(data: Union[pd.Series, np.ndarray], 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9, 
        adjust: bool = False) -> Dict[str, pd.Series]:
    """
    计算百分比价格震荡指标(PPO)
    PPO类似于MACD，但以百分比表示，使其在不同价格区间更具可比性

    参数:
        data: 原始价格数据，通常是收盘价
        fast_period: 快线周期，默认为12
        slow_period: 慢线周期，默认为26
        signal_period: 信号线周期，默认为9
        adjust: 是否调整EMA计算，默认为False

    返回:
        包含PPO线、信号线和柱状图的字典
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # 计算快线和慢线
    ema_fast = data.ewm(span=fast_period, adjust=adjust).mean()
    ema_slow = data.ewm(span=slow_period, adjust=adjust).mean()
    
    # 计算PPO线 (百分比表示的差异)
    ppo_line = (ema_fast - ema_slow) / ema_slow * 100
    
    # 计算信号线 (PPO的EMA)
    signal_line = ppo_line.ewm(span=signal_period, adjust=adjust).mean()
    
    # 计算柱状图 (PPO线 - 信号线)
    histogram = ppo_line - signal_line
    
    return {
        'ppo': ppo_line,
        'signal': signal_line,
        'histogram': histogram
    } 