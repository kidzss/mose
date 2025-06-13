"""
公用技术指标模块

提供RSI、MACD、ADX等常用技术指标的计算函数，供所有策略使用。
这些指标可以在数据预处理阶段统一计算，避免重复计算。
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional


class CommonIndicators:
    """公用技术指标计算类"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算相对强弱指数(RSI)
        
        参数:
            prices: 价格序列（通常是收盘价）
            period: 计算周期，默认14
            
        返回:
            RSI值序列
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            
            # 处理异常值
            rsi = rsi.replace([np.inf, -np.inf], np.nan)
            rsi = rsi.fillna(50)  # 用中性值填充NaN
            rsi = rsi.clip(0, 100)  # 确保在0-100范围内
            
            return rsi
        except Exception:
            return pd.Series(index=prices.index, dtype=float).fillna(50)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        参数:
            prices: 价格序列
            fast: 快线周期，默认12
            slow: 慢线周期，默认26
            signal: 信号线周期，默认9
            
        返回:
            包含MACD线、信号线和柱状图的字典
        """
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return {
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram
            }
        except Exception:
            return {
                'macd': pd.Series(index=prices.index, dtype=float),
                'macd_signal': pd.Series(index=prices.index, dtype=float),
                'macd_histogram': pd.Series(index=prices.index, dtype=float)
            }
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算平均趋向指数(ADX)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14
            
        返回:
            ADX值序列
        """
        try:
            # 计算真实波幅(TR)
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # 计算方向移动
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # 平滑处理
            tr_smooth = true_range.rolling(window=period).mean()
            plus_dm_smooth = plus_dm.rolling(window=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period).mean()
            
            # 计算方向指标
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # 计算ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            # 处理异常值
            adx = adx.replace([np.inf, -np.inf], np.nan)
            adx = adx.fillna(20)  # 用默认值填充NaN
            adx = adx.clip(0, 100)  # 确保在合理范围内
            
            return adx
        except Exception:
            return pd.Series(index=high.index, dtype=float).fillna(20)
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带
        
        参数:
            prices: 价格序列
            period: 计算周期，默认20
            std_dev: 标准差倍数，默认2.0
            
        返回:
            包含上轨、中轨、下轨的字典
        """
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return {
                'bb_upper': upper,
                'bb_middle': sma,
                'bb_lower': lower
            }
        except Exception:
            return {
                'bb_upper': pd.Series(index=prices.index, dtype=float),
                'bb_middle': pd.Series(index=prices.index, dtype=float),
                'bb_lower': pd.Series(index=prices.index, dtype=float)
            }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算随机指标
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_period: K值计算周期，默认14
            d_period: D值平滑周期，默认3
            
        返回:
            包含K值和D值的字典
        """
        try:
            low_min = low.rolling(window=k_period).min()
            high_max = high.rolling(window=k_period).max()
            
            k_percent = 100 * (close - low_min) / (high_max - low_min)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            # 处理异常值
            k_percent = k_percent.replace([np.inf, -np.inf], np.nan).fillna(50).clip(0, 100)
            d_percent = d_percent.replace([np.inf, -np.inf], np.nan).fillna(50).clip(0, 100)
            
            return {
                'stoch_k': k_percent,
                'stoch_d': d_percent
            }
        except Exception:
            return {
                'stoch_k': pd.Series(index=high.index, dtype=float).fillna(50),
                'stoch_d': pd.Series(index=high.index, dtype=float).fillna(50)
            }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算平均真实波幅(ATR)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14
            
        返回:
            ATR值序列
        """
        try:
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0)
        except Exception:
            return pd.Series(index=high.index, dtype=float).fillna(0)
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算威廉指标(%R)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14
            
        返回:
            威廉指标值序列
        """
        try:
            high_max = high.rolling(window=period).max()
            low_min = low.rolling(window=period).min()
            
            williams_r = -100 * (high_max - close) / (high_max - low_min)
            
            # 处理异常值
            williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).clip(-100, 0)
            
            return williams_r
        except Exception:
            return pd.Series(index=high.index, dtype=float).fillna(-50)
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        计算商品通道指数(CCI)
        
        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认20
            
        返回:
            CCI值序列
        """
        try:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            # 处理异常值
            cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return cci
        except Exception:
            return pd.Series(index=high.index, dtype=float).fillna(0)
    
    @classmethod
    def add_all_indicators(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        为数据添加所有常用技术指标
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了所有技术指标的DataFrame
        """
        try:
            df = data.copy()
            
            # 确保必要的列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    return df
            
            # 添加大小写兼容的列名
            df['Open'] = df['open']
            df['High'] = df['high']
            df['Low'] = df['low']
            df['Close'] = df['close']
            df['Volume'] = df['volume']
            
            # 计算所有指标
            df['RSI'] = cls.calculate_rsi(df['close'])
            df['rsi'] = df['RSI']
            
            macd_data = cls.calculate_macd(df['close'])
            df['MACD'] = macd_data['macd']
            df['macd'] = macd_data['macd']
            df['MACD_signal'] = macd_data['macd_signal']
            df['macd_signal'] = macd_data['macd_signal']
            df['MACD_histogram'] = macd_data['macd_histogram']
            df['macd_histogram'] = macd_data['macd_histogram']
            
            df['ADX'] = cls.calculate_adx(df['high'], df['low'], df['close'])
            df['adx'] = df['ADX']
            
            bb_data = cls.calculate_bollinger_bands(df['close'])
            df.update(bb_data)
            
            stoch_data = cls.calculate_stochastic(df['high'], df['low'], df['close'])
            df.update(stoch_data)
            
            df['ATR'] = cls.calculate_atr(df['high'], df['low'], df['close'])
            df['atr'] = df['ATR']
            
            df['williams_r'] = cls.calculate_williams_r(df['high'], df['low'], df['close'])
            df['cci'] = cls.calculate_cci(df['high'], df['low'], df['close'])
            
            # 移动平均线
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # 成交量指标
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 动量指标
            df['momentum'] = df['close'].pct_change(periods=10)
            df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # 波动率
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 支撑阻力
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # 处理无穷大和NaN值
            df = df.replace([np.inf, -np.inf], np.nan)
            
            return df
            
        except Exception as e:
            print(f"添加技术指标失败: {e}")
            return data


# 为了向后兼容，提供函数接口
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI计算函数（向后兼容）"""
    return CommonIndicators.calculate_rsi(prices, period)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """MACD计算函数（向后兼容）"""
    return CommonIndicators.calculate_macd(prices, fast, slow, signal)


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX计算函数（向后兼容）"""
    return CommonIndicators.calculate_adx(high, low, close, period)


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """添加所有技术指标函数（向后兼容）"""
    return CommonIndicators.add_all_indicators(data) 