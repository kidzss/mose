#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
信号提取器模块

负责从各种交易策略中提取、标准化和处理信号，并为每个信号添加元数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas_ta as ta

class SignalExtractor:
    """
    信号提取器
    
    从各种交易策略中提取信号，进行标准化处理，并添加元数据
    """
    
    def __init__(self, price_data: pd.DataFrame = None):
        """
        初始化信号提取器
        
        参数:
            price_data: 价格数据 DataFrame，包含 OHLCV 列
        """
        self.price_data = price_data
        self.signals = {}  # 存储所有提取的信号
        self.metadata = {}  # 存储信号的元数据
    
    def set_price_data(self, price_data: pd.DataFrame):
        """
        设置价格数据
        
        参数:
            price_data: 价格数据 DataFrame，包含 OHLCV 列
        """
        self.price_data = price_data
    
    def extract_trend_signals(self, 
                               short_period: int = 20, 
                               medium_period: int = 50, 
                               long_period: int = 200,
                               prefix: str = "trend_") -> pd.DataFrame:
        """
        提取趋势相关信号
        
        参数:
            short_period: 短期均线周期
            medium_period: 中期均线周期
            long_period: 长期均线周期
            prefix: 信号名称前缀
            
        返回:
            包含趋势信号的 DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        price = self.price_data["close"].copy()
        
        # 计算各种均线
        sma_short = ta.sma(price, length=short_period)
        sma_medium = ta.sma(price, length=medium_period)
        sma_long = ta.sma(price, length=long_period)
        
        # 生成均线交叉信号
        signals = pd.DataFrame(index=price.index)
        
        # 短期/中期均线交叉
        signals[f"{prefix}short_above_medium"] = (sma_short > sma_medium).astype(float)
        
        # 短期/长期均线交叉
        signals[f"{prefix}short_above_long"] = (sma_short > sma_long).astype(float)
        
        # 中期/长期均线交叉
        signals[f"{prefix}medium_above_long"] = (sma_medium > sma_long).astype(float)
        
        # 价格相对于均线的位置
        signals[f"{prefix}price_above_short"] = (price > sma_short).astype(float)
        signals[f"{prefix}price_above_medium"] = (price > sma_medium).astype(float)
        signals[f"{prefix}price_above_long"] = (price > sma_long).astype(float)
        
        # 添加元数据
        for col in signals.columns:
            self.metadata[col] = {
                "category": "trend",
                "source": "moving_average",
                "timeframe": "daily",  # 假设输入数据是日线数据
                "description": f"基于 {short_period}/{medium_period}/{long_period} 周期均线的趋势信号"
            }
        
        # 保存信号
        self.signals.update({col: signals[col] for col in signals.columns})
        
        return signals
    
    def extract_momentum_signals(self, 
                                 rsi_period: int = 14, 
                                 macd_fast: int = 12, 
                                 macd_slow: int = 26, 
                                 macd_signal: int = 9,
                                 prefix: str = "momentum_") -> pd.DataFrame:
        """
        提取动量相关信号
        
        参数:
            rsi_period: RSI 计算周期
            macd_fast: MACD 快线周期
            macd_slow: MACD 慢线周期
            macd_signal: MACD 信号线周期
            prefix: 信号名称前缀
            
        返回:
            包含动量信号的 DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        price = self.price_data["close"].copy()
        high = self.price_data["high"].copy()
        low = self.price_data["low"].copy()
        
        # 计算各种动量指标
        rsi = ta.rsi(price, length=rsi_period)
        macd = ta.macd(price, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        
        # 整合信号
        signals = pd.DataFrame(index=price.index)
        
        # RSI 信号
        signals[f"{prefix}rsi"] = rsi / 100.0  # 归一化到 0-1
        signals[f"{prefix}rsi_oversold"] = (rsi < 30).astype(float)
        signals[f"{prefix}rsi_overbought"] = (rsi > 70).astype(float)
        
        # MACD 信号
        signals[f"{prefix}macd_above_signal"] = (macd["MACD_12_26_9"] > macd["MACDs_12_26_9"]).astype(float)
        signals[f"{prefix}macd_histogram"] = macd["MACDh_12_26_9"] / price.abs().mean()  # 归一化
        
        # 动量 (比率)
        signals[f"{prefix}momentum_5d"] = price / price.shift(5) - 1
        signals[f"{prefix}momentum_10d"] = price / price.shift(10) - 1
        signals[f"{prefix}momentum_20d"] = price / price.shift(20) - 1
        
        # 添加元数据
        for col in signals.columns:
            category = "momentum"
            source = "unknown"
            desc = "动量信号"
            
            if "rsi" in col:
                source = "rsi"
                desc = f"基于 {rsi_period} 周期 RSI 的动量信号"
            elif "macd" in col:
                source = "macd"
                desc = f"基于 {macd_fast}/{macd_slow}/{macd_signal} 参数的 MACD 信号"
            elif "momentum" in col:
                source = "price_ratio"
                period = col.split("_")[-1].replace("d", "")
                desc = f"{period} 天价格变化率"
            
            self.metadata[col] = {
                "category": category,
                "source": source,
                "timeframe": "daily",  # 假设输入数据是日线数据
                "description": desc
            }
        
        # 保存信号
        self.signals.update({col: signals[col] for col in signals.columns})
        
        return signals
    
    def extract_volatility_signals(self, 
                                  atr_period: int = 14, 
                                  bollinger_period: int = 20, 
                                  bollinger_std: float = 2.0,
                                  prefix: str = "volatility_") -> pd.DataFrame:
        """
        提取波动率相关信号
        
        参数:
            atr_period: ATR 计算周期
            bollinger_period: 布林带周期
            bollinger_std: 布林带标准差倍数
            prefix: 信号名称前缀
            
        返回:
            包含波动率信号的 DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        price = self.price_data["close"].copy()
        high = self.price_data["high"].copy()
        low = self.price_data["low"].copy()
        
        # 计算波动率指标
        atr = ta.atr(high, low, price, length=atr_period)
        bbands = ta.bbands(price, length=bollinger_period, std=bollinger_std)
        
        # 整合信号
        signals = pd.DataFrame(index=price.index)
        
        # ATR 相关信号
        signals[f"{prefix}atr_percent"] = atr / price  # 相对价格的 ATR
        
        # 布林带相关信号
        signals[f"{prefix}bb_width"] = (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"]) / bbands["BBM_20_2.0"]
        signals[f"{prefix}price_bb_upper"] = (price >= bbands["BBU_20_2.0"]).astype(float)
        signals[f"{prefix}price_bb_lower"] = (price <= bbands["BBL_20_2.0"]).astype(float)
        signals[f"{prefix}price_bb_percent"] = (price - bbands["BBL_20_2.0"]) / (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"])
        
        # 历史波动率
        signals[f"{prefix}volatility_5d"] = price.pct_change().rolling(5).std() * np.sqrt(252)
        signals[f"{prefix}volatility_20d"] = price.pct_change().rolling(20).std() * np.sqrt(252)
        
        # 添加元数据
        for col in signals.columns:
            category = "volatility"
            source = "unknown"
            desc = "波动率信号"
            
            if "atr" in col:
                source = "atr"
                desc = f"基于 {atr_period} 周期 ATR 的波动率信号"
            elif "bb_" in col or "price_bb" in col:
                source = "bollinger_bands"
                desc = f"基于 {bollinger_period} 周期, {bollinger_std} 标准差的布林带信号"
            elif "volatility_" in col:
                source = "historical_volatility"
                period = col.split("_")[-1].replace("d", "")
                desc = f"{period} 天历史波动率"
            
            self.metadata[col] = {
                "category": category,
                "source": source,
                "timeframe": "daily",
                "description": desc
            }
        
        # 保存信号
        self.signals.update({col: signals[col] for col in signals.columns})
        
        return signals
    
    def extract_volume_signals(self, 
                              volume_ma_period: int = 20,
                              prefix: str = "volume_") -> pd.DataFrame:
        """
        提取交易量相关信号
        
        参数:
            volume_ma_period: 交易量均线周期
            prefix: 信号名称前缀
            
        返回:
            包含交易量信号的 DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        if "volume" not in self.price_data.columns:
            raise ValueError("价格数据缺少 volume 列")
        
        price = self.price_data["close"].copy()
        volume = self.price_data["volume"].copy()
        
        # 计算交易量指标
        volume_ma = ta.sma(volume, length=volume_ma_period)
        
        # 整合信号
        signals = pd.DataFrame(index=price.index)
        
        # 基础交易量信号
        signals[f"{prefix}relative"] = volume / volume_ma
        signals[f"{prefix}above_ma"] = (volume > volume_ma).astype(float)
        
        # 量价关系
        price_change = price.pct_change()
        signals[f"{prefix}price_up_volume_up"] = ((price_change > 0) & (volume > volume_ma)).astype(float)
        signals[f"{prefix}price_down_volume_up"] = ((price_change < 0) & (volume > volume_ma)).astype(float)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(price_change) * volume).fillna(0).cumsum()
        obv_ma = ta.sma(obv, length=volume_ma_period)
        signals[f"{prefix}obv_above_ma"] = (obv > obv_ma).astype(float)
        
        # 添加元数据
        for col in signals.columns:
            category = "volume"
            source = "unknown"
            desc = "交易量信号"
            
            if "relative" in col or "above_ma" in col:
                source = "volume_ma"
                desc = f"基于 {volume_ma_period} 周期交易量均线的信号"
            elif "price_up" in col or "price_down" in col:
                source = "volume_price"
                desc = "量价关系信号"
            elif "obv" in col:
                source = "obv"
                desc = "基于能量潮(OBV)的交易量信号"
            
            self.metadata[col] = {
                "category": category,
                "source": source,
                "timeframe": "daily",
                "description": desc
            }
        
        # 保存信号
        self.signals.update({col: signals[col] for col in signals.columns})
        
        return signals
    
    def extract_support_resistance_signals(self, 
                                          lookback_period: int = 20,
                                          prefix: str = "sr_") -> pd.DataFrame:
        """
        提取支撑和阻力位相关信号
        
        参数:
            lookback_period: 回看周期
            prefix: 信号名称前缀
            
        返回:
            包含支撑和阻力位信号的 DataFrame
        """
        if self.price_data is None:
            raise ValueError("请先设置价格数据")
        
        price = self.price_data["close"].copy()
        high = self.price_data["high"].copy()
        low = self.price_data["low"].copy()
        
        # 整合信号
        signals = pd.DataFrame(index=price.index)
        
        # 计算近期支撑位和阻力位
        for i in range(len(price)):
            if i >= lookback_period:
                # 获取一段历史数据
                high_window = high.iloc[i-lookback_period:i]
                low_window = low.iloc[i-lookback_period:i]
                current_price = price.iloc[i]
                
                # 识别阻力位 (比当前价格高的近期高点)
                resistance_candidates = high_window[high_window > current_price]
                if not resistance_candidates.empty:
                    signals.loc[price.index[i], f"{prefix}nearest_resistance"] = (resistance_candidates.min() - current_price) / current_price
                else:
                    signals.loc[price.index[i], f"{prefix}nearest_resistance"] = 0.1  # 如果没有阻力位，设置一个默认值
                
                # 识别支撑位 (比当前价格低的近期低点)
                support_candidates = low_window[low_window < current_price]
                if not support_candidates.empty:
                    signals.loc[price.index[i], f"{prefix}nearest_support"] = (current_price - support_candidates.max()) / current_price
                else:
                    signals.loc[price.index[i], f"{prefix}nearest_support"] = 0.1  # 如果没有支撑位，设置一个默认值
                
                # 计算支撑/阻力带宽度
                signals.loc[price.index[i], f"{prefix}range_width"] = signals.loc[price.index[i], f"{prefix}nearest_resistance"] + \
                                                                     signals.loc[price.index[i], f"{prefix}nearest_support"]
                
                # 计算价格在支撑阻力区间内的位置 (0=支撑位, 1=阻力位)
                range_position = signals.loc[price.index[i], f"{prefix}nearest_support"] / signals.loc[price.index[i], f"{prefix}range_width"] \
                                if signals.loc[price.index[i], f"{prefix}range_width"] > 0 else 0.5
                signals.loc[price.index[i], f"{prefix}range_position"] = range_position
            else:
                # 历史数据不足
                signals.loc[price.index[i], f"{prefix}nearest_resistance"] = 0.1
                signals.loc[price.index[i], f"{prefix}nearest_support"] = 0.1
                signals.loc[price.index[i], f"{prefix}range_width"] = 0.2
                signals.loc[price.index[i], f"{prefix}range_position"] = 0.5
        
        # 计算突破信号
        signals[f"{prefix}break_resistance"] = signals[f"{prefix}nearest_resistance"].shift(1) < 0.01
        signals[f"{prefix}break_support"] = signals[f"{prefix}nearest_support"].shift(1) < 0.01
        
        # 添加元数据
        for col in signals.columns:
            self.metadata[col] = {
                "category": "support_resistance",
                "source": "price_levels",
                "timeframe": "daily",
                "description": f"基于 {lookback_period} 天历史数据的支撑/阻力位信号"
            }
        
        # 保存信号
        self.signals.update({col: signals[col] for col in signals.columns})
        
        return signals
    
    def extract_all_signals(self) -> pd.DataFrame:
        """
        提取所有类型的信号
        
        返回:
            包含所有信号的 DataFrame
        """
        self.extract_trend_signals()
        self.extract_momentum_signals()
        self.extract_volatility_signals()
        self.extract_volume_signals()
        self.extract_support_resistance_signals()
        
        # 将所有信号合并到一个 DataFrame
        all_signals = pd.DataFrame(self.signals)
        
        return all_signals
    
    def get_signals(self, categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取指定类别的信号
        
        参数:
            categories: 要获取的信号类别列表，如果为 None，则获取所有信号
            
        返回:
            包含指定类别信号的 DataFrame
        """
        if not self.signals:
            raise ValueError("请先提取信号")
        
        if categories is None:
            return pd.DataFrame(self.signals)
        
        # 筛选指定类别的信号
        selected_signals = {}
        for signal_name, signal in self.signals.items():
            if signal_name in self.metadata and self.metadata[signal_name]["category"] in categories:
                selected_signals[signal_name] = signal
        
        return pd.DataFrame(selected_signals)
    
    def get_metadata(self, signal_name: Optional[str] = None) -> Dict:
        """
        获取信号的元数据
        
        参数:
            signal_name: 要获取元数据的信号名称，如果为 None，则获取所有信号的元数据
            
        返回:
            信号的元数据字典
        """
        if signal_name is not None:
            if signal_name in self.metadata:
                return self.metadata[signal_name]
            else:
                raise ValueError(f"信号 {signal_name} 不存在或未提取")
        else:
            return self.metadata
    
    def normalize_signals(self, method: str = "zscore") -> pd.DataFrame:
        """
        对信号进行标准化处理
        
        参数:
            method: 标准化方法，可选 "zscore", "minmax", "robust"
            
        返回:
            标准化后的信号 DataFrame
        """
        if not self.signals:
            raise ValueError("请先提取信号")
        
        signals_df = pd.DataFrame(self.signals)
        
        # 对每个信号应用标准化
        for col in signals_df.columns:
            if method == "zscore":
                # Z-score 标准化
                mean = signals_df[col].mean()
                std = signals_df[col].std()
                if std > 0:
                    signals_df[col] = (signals_df[col] - mean) / std
            elif method == "minmax":
                # Min-max 标准化到 [0, 1]
                min_val = signals_df[col].min()
                max_val = signals_df[col].max()
                if max_val > min_val:
                    signals_df[col] = (signals_df[col] - min_val) / (max_val - min_val)
            elif method == "robust":
                # 稳健标准化，基于四分位数
                q1 = signals_df[col].quantile(0.25)
                q3 = signals_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    signals_df[col] = (signals_df[col] - signals_df[col].median()) / iqr
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
        
        return signals_df
    
    def calculate_signal_correlation(self) -> pd.DataFrame:
        """
        计算信号之间的相关性
        
        返回:
            信号相关性矩阵
        """
        if not self.signals:
            raise ValueError("请先提取信号")
        
        signals_df = pd.DataFrame(self.signals)
        correlation_matrix = signals_df.corr()
        
        return correlation_matrix
    
    def rank_signals_by_correlation(self, target: pd.Series) -> pd.DataFrame:
        """
        根据与目标变量的相关性对信号进行排序
        
        参数:
            target: 目标变量序列，如收益率
            
        返回:
            按相关性排序的信号列表
        """
        if not self.signals:
            raise ValueError("请先提取信号")
        
        signals_df = pd.DataFrame(self.signals)
        
        # 计算每个信号与目标变量的相关性
        correlations = {}
        for col in signals_df.columns:
            common_index = signals_df[col].dropna().index.intersection(target.dropna().index)
            if len(common_index) > 0:
                correlation = signals_df[col].loc[common_index].corr(target.loc[common_index])
                correlations[col] = correlation
            else:
                correlations[col] = np.nan
        
        # 转换为 DataFrame 并排序
        correlation_df = pd.DataFrame({
            'signal': list(correlations.keys()),
            'correlation': list(correlations.values())
        })
        correlation_df['abs_correlation'] = correlation_df['correlation'].abs()
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        
        return correlation_df
    
    def get_top_signals(self, target: pd.Series, n: int = 10) -> List[str]:
        """
        获取与目标相关性最高的前 n 个信号
        
        参数:
            target: 目标变量序列，如收益率
            n: 要返回的信号数量
            
        返回:
            相关性最高的信号名称列表
        """
        correlation_df = self.rank_signals_by_correlation(target)
        top_signals = correlation_df['signal'].head(n).tolist()
        
        return top_signals


# 使用示例
if __name__ == "__main__":
    # 创建模拟价格数据
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 生成模拟价格数据
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    np.random.seed(42)
    
    # 模拟价格数据
    price_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': 0.0,
        'low': 0.0,
        'close': 0.0,
        'volume': np.abs(np.random.randn(len(dates)) * 1000000)
    }, index=dates)
    
    # 调整 high, low, close
    for i in range(len(price_data)):
        price_data.loc[price_data.index[i], 'close'] = price_data.loc[price_data.index[i], 'open'] * (1 + np.random.randn() * 0.01)
        price_data.loc[price_data.index[i], 'high'] = max(price_data.loc[price_data.index[i], 'open'], price_data.loc[price_data.index[i], 'close']) * (1 + abs(np.random.randn() * 0.005))
        price_data.loc[price_data.index[i], 'low'] = min(price_data.loc[price_data.index[i], 'open'], price_data.loc[price_data.index[i], 'close']) * (1 - abs(np.random.randn() * 0.005))
    
    # 创建收益率序列
    returns = price_data['close'].pct_change().shift(-1)  # 使用下一天的收益率作为目标
    
    # 初始化信号提取器
    extractor = SignalExtractor(price_data)
    
    # 提取各类信号
    trend_signals = extractor.extract_trend_signals()
    momentum_signals = extractor.extract_momentum_signals()
    volatility_signals = extractor.extract_volatility_signals()
    volume_signals = extractor.extract_volume_signals()
    
    # 打印信号信息
    print(f"提取的信号总数: {len(extractor.signals)}")
    print("\n各类别的信号数量:")
    category_counts = {}
    for signal, metadata in extractor.metadata.items():
        category = metadata["category"]
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    
    # 对信号进行标准化
    normalized_signals = extractor.normalize_signals(method="zscore")
    
    # 计算与收益率的相关性并排序
    correlation_df = extractor.rank_signals_by_correlation(returns)
    
    print("\n与收益率相关性最高的前 10 个信号:")
    for i, row in correlation_df.head(10).iterrows():
        signal = row['signal']
        correlation = row['correlation']
        metadata = extractor.get_metadata(signal)
        print(f"{signal}: 相关性 = {correlation:.4f}, 类别 = {metadata['category']}, 来源 = {metadata['source']}") 