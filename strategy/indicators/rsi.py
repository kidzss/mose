"""
相对强弱指数(RSI)模块

提供RSI及相关指标的计算函数，用于技术分析和交易策略。
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional


def rsi(data: Union[pd.Series, np.ndarray], 
        window: int = 14, 
        method: str = 'wilders') -> pd.Series:
    """
    计算相对强弱指数(RSI)

    参数:
        data: 原始价格数据，通常是收盘价
        window: 窗口大小，默认为14
        method: 计算方法，可选 'wilders'(原始方法) 或 'ema'(使用EMA)

    返回:
        RSI值的Series
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    # 计算价格变化
    delta = data.diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均上涨和下跌
    if method == 'wilders':
        # Wilder's方法: 第一个平均值为简单平均，之后使用平滑系数
        avg_gain = pd.Series(np.zeros_like(gain), index=gain.index)
        avg_loss = pd.Series(np.zeros_like(loss), index=loss.index)
        
        # 初始化第一个值
        if len(gain) >= window:
            avg_gain.iloc[window-1] = gain.iloc[:window].mean()
            avg_loss.iloc[window-1] = loss.iloc[:window].mean()
        
            # 计算后续值
            for i in range(window, len(gain)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window-1) + gain.iloc[i]) / window
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window-1) + loss.iloc[i]) / window
    else:
        # EMA方法: 使用指数移动平均计算
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    # 计算相对强度
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def rsi_divergence(price: pd.Series, 
                   rsi_values: pd.Series, 
                   window: int = 5,
                   threshold: float = 2.0) -> pd.DataFrame:
    """
    检测RSI与价格之间的背离
    
    参数:
        price: 价格数据
        rsi_values: RSI值
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
    
    # 找出RSI的局部极值
    rsi_highs = pd.Series(np.zeros_like(rsi_values), index=rsi_values.index)
    rsi_lows = pd.Series(np.zeros_like(rsi_values), index=rsi_values.index)
    
    for i in range(window, len(rsi_values) - window):
        # 如果当前RSI是窗口内的最高点
        if rsi_values.iloc[i] == rsi_values.iloc[i-window:i+window+1].max():
            rsi_highs.iloc[i] = 1
        
        # 如果当前RSI是窗口内的最低点
        if rsi_values.iloc[i] == rsi_values.iloc[i-window:i+window+1].min():
            rsi_lows.iloc[i] = 1
    
    # 初始化背离信号
    bullish_divergence = pd.Series(np.zeros_like(price), index=price.index)
    bearish_divergence = pd.Series(np.zeros_like(price), index=price.index)
    
    # 查找看涨背离: 价格创新低但RSI不创新低
    for i in range(window*2, len(price)):
        if price_lows.iloc[i] == 1:
            # 寻找前一个价格低点
            for j in range(i-window, max(0, i-window*5), -1):
                if price_lows.iloc[j] == 1:
                    # 如果当前价格更低但RSI更高，则是看涨背离
                    if (price.iloc[i] < price.iloc[j]) and (rsi_values.iloc[i] > rsi_values.iloc[j]):
                        # 计算背离强度
                        price_change = (price.iloc[i] / price.iloc[j] - 1) * 100
                        rsi_change = rsi_values.iloc[i] - rsi_values.iloc[j]
                        
                        # 只有显著背离才记录
                        if abs(price_change) + abs(rsi_change) > threshold:
                            bullish_divergence.iloc[i] = 1
                    break
    
    # 查找看跌背离: 价格创新高但RSI不创新高
    for i in range(window*2, len(price)):
        if price_highs.iloc[i] == 1:
            # 寻找前一个价格高点
            for j in range(i-window, max(0, i-window*5), -1):
                if price_highs.iloc[j] == 1:
                    # 如果当前价格更高但RSI更低，则是看跌背离
                    if (price.iloc[i] > price.iloc[j]) and (rsi_values.iloc[i] < rsi_values.iloc[j]):
                        # 计算背离强度
                        price_change = (price.iloc[i] / price.iloc[j] - 1) * 100
                        rsi_change = rsi_values.iloc[j] - rsi_values.iloc[i]
                        
                        # 只有显著背离才记录
                        if abs(price_change) + abs(rsi_change) > threshold:
                            bearish_divergence.iloc[i] = 1
                    break
    
    return pd.DataFrame({
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence
    })


def rsi_overbought_oversold(rsi_values: pd.Series, 
                            overbought: float = 70,
                            oversold: float = 30) -> pd.DataFrame:
    """
    检测RSI超买超卖状态
    
    参数:
        rsi_values: RSI值
        overbought: 超买阈值，默认为70
        oversold: 超卖阈值，默认为30
        
    返回:
        包含超买超卖状态的DataFrame
    """
    # 检测超买超卖状态
    overbought_signal = (rsi_values > overbought).astype(int)
    oversold_signal = (rsi_values < oversold).astype(int)
    
    # 检测从超买区域回落的信号
    overbought_exit = ((rsi_values < overbought) & (rsi_values.shift(1) >= overbought)).astype(int)
    
    # 检测从超卖区域回升的信号
    oversold_exit = ((rsi_values > oversold) & (rsi_values.shift(1) <= oversold)).astype(int)
    
    return pd.DataFrame({
        'overbought': overbought_signal,
        'oversold': oversold_signal,
        'overbought_exit': overbought_exit,
        'oversold_exit': oversold_exit
    })


def rsi_reversal(rsi_values: pd.Series, 
                 window: int = 3,
                 threshold: float = 5.0) -> pd.DataFrame:
    """
    检测RSI反转信号
    
    参数:
        rsi_values: RSI值
        window: 用于判断趋势的窗口大小
        threshold: 反转所需的最小RSI变化阈值
        
    返回:
        包含RSI反转信号的DataFrame
    """
    # 计算RSI的短期趋势方向
    rsi_trend = pd.Series(np.zeros_like(rsi_values), index=rsi_values.index)
    
    for i in range(window, len(rsi_values)):
        # 计算窗口内的RSI平均变化
        changes = np.diff(rsi_values.iloc[i-window:i+1])
        avg_change = np.mean(changes)
        
        if avg_change > 0:
            rsi_trend.iloc[i] = 1  # 上升趋势
        elif avg_change < 0:
            rsi_trend.iloc[i] = -1  # 下降趋势
    
    # 检测趋势反转
    trend_change = rsi_trend.diff()
    
    # 从下降到上升的反转
    bullish_reversal = (trend_change == 2) & (abs(rsi_values - rsi_values.shift(window)) > threshold)
    
    # 从上升到下降的反转
    bearish_reversal = (trend_change == -2) & (abs(rsi_values - rsi_values.shift(window)) > threshold)
    
    return pd.DataFrame({
        'bullish_reversal': bullish_reversal.astype(int),
        'bearish_reversal': bearish_reversal.astype(int)
    })


def stochastic_rsi(rsi_values: pd.Series, 
                   k_period: int = 3, 
                   d_period: int = 3) -> Dict[str, pd.Series]:
    """
    计算随机相对强弱指标(Stochastic RSI)
    
    参数:
        rsi_values: RSI值
        k_period: K线周期
        d_period: D线周期
        
    返回:
        包含K线和D线的字典
    """
    # 计算RSI的最高和最低值
    min_rsi = rsi_values.rolling(window=k_period).min()
    max_rsi = rsi_values.rolling(window=k_period).max()
    
    # 计算随机RSI的K值
    stoch_k = 100 * (rsi_values - min_rsi) / (max_rsi - min_rsi)
    
    # 计算随机RSI的D值 (K值的简单移动平均)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return {
        'k': stoch_k,
        'd': stoch_d
    } 