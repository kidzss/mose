import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from .strategy_base import Strategy, MarketRegime
from .signal_interface import SignalComponent, SignalMetadata, SignalType, SignalTimeframe

class CustomCPGWStrategy(Strategy):
    """
    CPGW策略 - 操盘工位策略 (优化版)
    
    CPGW (操盘工位) 策略基于EMA的金叉死叉，使用快线和慢线的交叉以及信号线判断趋势，
    生成买入和卖出信号。
    
    优化点：
    1. 增强牛市中的买入条件
    2. 添加趋势跟踪能力
    3. 添加突破买入逻辑
    """
    
    def __init__(self, 
                 rsi_period=14,
                 rsi_oversold=30,
                 rsi_overbought=70,
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 volume_ma_period=20,
                 volume_threshold=1.5,
                 trend_ma_period=20,
                 volatility_period=20,
                 volatility_threshold=0.02,
                 market_adaptation=True):
        """
        初始化CPGW策略
        
        参数:
            rsi_period: RSI计算周期
            rsi_oversold: RSI超卖阈值
            rsi_overbought: RSI超买阈值
            macd_fast: MACD快速线周期
            macd_slow: MACD慢速线周期
            macd_signal: MACD信号线周期
            volume_ma_period: 成交量移动平均周期
            volume_threshold: 成交量突破因子
            trend_ma_period: 趋势移动平均周期
            volatility_period: 波动率计算周期
            volatility_threshold: 波动率阈值
            market_adaptation: 是否使用市场环境调整
        """
        super().__init__('CustomCPGWStrategy')
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.trend_ma_period = trend_ma_period
        self.volatility_period = volatility_period
        self.volatility_threshold = volatility_threshold
        self.market_adaptation = market_adaptation
        
        # 市场环境参数
        self.market_params = {
            MarketRegime.BULLISH: {
                'rsi_oversold': 35,  # 更宽松的RSI超卖阈值
                'rsi_overbought': 75,  # 更宽松的RSI超买阈值
                'volume_threshold': 1.2,  # 更低的成交量阈值
                'volatility_threshold': 0.025,  # 更高的波动率阈值
                'signal_threshold': 0.5  # 更低的信号阈值
            },
            MarketRegime.BEARISH: {
                'rsi_oversold': 25,  # 更严格的RSI超卖阈值
                'rsi_overbought': 65,  # 更严格的RSI超买阈值
                'volume_threshold': 1.8,  # 更高的成交量阈值
                'volatility_threshold': 0.015,  # 更低的波动率阈值
                'signal_threshold': 0.7  # 更高的信号阈值
            },
            MarketRegime.RANGING: {
                'rsi_oversold': 30,  # 标准RSI超卖阈值
                'rsi_overbought': 70,  # 标准RSI超买阈值
                'volume_threshold': 1.5,  # 标准成交量阈值
                'volatility_threshold': 0.02,  # 标准波动率阈值
                'signal_threshold': 0.6  # 标准信号阈值
            },
            MarketRegime.VOLATILE: {
                'rsi_oversold': 20,  # 更严格的RSI超卖阈值
                'rsi_overbought': 80,  # 更严格的RSI超买阈值
                'volume_threshold': 2.0,  # 更高的成交量阈值
                'volatility_threshold': 0.03,  # 更高的波动率阈值
                'signal_threshold': 0.8  # 更高的信号阈值
            }
        }
        
        # 初始化市场环境
        self.current_regime = MarketRegime.RANGING
        self.logger = logging.getLogger(__name__)
        
        # 初始化信号元数据
        self.signal_metadata = {
            'fast_ema': SignalMetadata(
                name='快线',
                description=f"{self.macd_fast}周期EMA",
                signal_type=SignalType.TREND,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='minmax'
            ),
            'slow_ema': SignalMetadata(
                name='慢线',
                description=f"{self.macd_slow}周期EMA",
                signal_type=SignalType.TREND,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='minmax'
            ),
            'diff': SignalMetadata(
                name='差值线',
                description='快线与慢线的差值',
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='zscore'
            ),
            'signal': SignalMetadata(
                name='信号线',
                description=f"差值的{self.macd_signal}周期平均",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='zscore'
            ),
            'histogram': SignalMetadata(
                name='柱状图',
                description='差值线与信号线的差',
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='zscore'
            ),
            'price': SignalMetadata(
                name='价格',
                description='资产收盘价',
                signal_type=SignalType.UNKNOWN,
                timeframe=SignalTimeframe.DAILY,
                weight=0.5,
                normalization='none'
            ),
            'trend_score': SignalMetadata(
                name='趋势得分',
                description='多周期趋势综合得分',
                signal_type=SignalType.TREND,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='minmax'
            ),
            'rsi': SignalMetadata(
                name='RSI',
                description='相对强弱指数',
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='minmax'
            ),
            'breakout': SignalMetadata(
                name='突破',
                description='价格突破阻力位',
                signal_type=SignalType.BREAKOUT,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='none'
            )
        }
    
    def _adjust_parameters_for_market(self):
        """根据市场环境调整参数"""
        if not self.market_adaptation:
            return
            
        params = self.market_params[self.current_regime]
        self.rsi_oversold = params['rsi_oversold']
        self.rsi_overbought = params['rsi_overbought']
        self.volume_threshold = params['volume_threshold']
        self.volatility_threshold = params['volatility_threshold']
        self.signal_threshold = params['signal_threshold']
        
        self.logger.info(f"CPGW策略: 市场环境{self.current_regime.value}，参数已调整")
    
    def _calculate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算交易信号"""
        # 计算技术指标
        df = self._calculate_indicators(df)
        
        # 根据市场环境调整参数
        self._adjust_parameters_for_market()
        
        # 初始化信号列
        df['signal'] = 0.0  # 确保signal列为float类型
        
        # 计算信号强度
        for i in range(len(df)):
            if i < self.trend_ma_period:
                continue
                
            # 趋势判断
            trend = 1 if df['close'].iloc[i] > df['ma'].iloc[i] else -1
            
            # RSI判断
            rsi_signal = 0
            if df['rsi'].iloc[i] < self.rsi_oversold:
                rsi_signal = 1
            elif df['rsi'].iloc[i] > self.rsi_overbought:
                rsi_signal = -1
                
            # MACD判断
            macd_signal = 0
            if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                macd_signal = 1
            elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                macd_signal = -1
                
            # 成交量判断
            volume_signal = 0
            if df['volume'].iloc[i] > df['volume_ma'].iloc[i] * self.volume_threshold:
                volume_signal = 1
            elif df['volume'].iloc[i] < df['volume_ma'].iloc[i] / self.volume_threshold:
                volume_signal = -1
                
            # 波动率判断
            volatility_signal = 0
            if df['volatility'].iloc[i] > self.volatility_threshold:
                volatility_signal = -1  # 高波动时偏向做空
                
            # 综合信号
            total_signal = (trend + rsi_signal + macd_signal + volume_signal + volatility_signal) / 5
            
            # 根据市场环境调整信号
            if self.current_regime == MarketRegime.BEARISH:
                # 熊市环境下增强卖出信号
                if total_signal < 0:
                    total_signal *= 1.2
                else:
                    total_signal *= 0.8
            elif self.current_regime == MarketRegime.BULLISH:
                # 牛市环境下增强买入信号
                if total_signal > 0:
                    total_signal *= 1.2
                else:
                    total_signal *= 0.8
                    
            # 限制信号强度在[-1.5, 1.5]范围内
            total_signal = np.clip(total_signal, -1.5, 1.5)
            
            # 如果有前一个信号，限制信号变化幅度
            if i > 0 and df['signal'].iloc[i-1] != 0:
                prev_signal = df['signal'].iloc[i-1]
                max_change = 1.5
                min_signal = max(prev_signal - max_change, -1.5)
                max_signal = min(prev_signal + max_change, 1.5)
                total_signal = np.clip(total_signal, min_signal, max_signal)
            
            # 设置信号
            if abs(total_signal) >= self.signal_threshold:
                df.loc[df.index[i], 'signal'] = total_signal
                
        return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        fast_length = self.macd_fast
        slow_length = self.macd_slow
        signal_length = self.macd_signal
        
        # 计算快线和慢线 EMA
        df['cpgw_fast_ema'] = df['close'].ewm(span=fast_length, adjust=False).mean()
        df['cpgw_slow_ema'] = df['close'].ewm(span=slow_length, adjust=False).mean()
        
        # 计算差值 (快线 - 慢线)
        df['cpgw_diff'] = df['cpgw_fast_ema'] - df['cpgw_slow_ema']
        
        # 计算信号线
        df['cpgw_signal'] = df['cpgw_diff'].ewm(span=signal_length, adjust=False).mean()
        
        # 计算柱状图
        df['cpgw_histogram'] = df['cpgw_diff'] - df['cpgw_signal']
        
        # 添加趋势跟踪功能 - 计算多周期趋势线
        trend_short = 5  # 短期趋势
        trend_medium = 20  # 中期趋势
        trend_long = 60  # 长期趋势
        
        df['trend_short'] = df['close'] > df['close'].rolling(trend_short).mean()
        df['trend_medium'] = df['close'] > df['close'].rolling(trend_medium).mean()
        df['trend_long'] = df['close'] > df['close'].rolling(trend_long).mean()
        
        # 计算趋势得分 (0-3)，给予不同权重
        df['trend_score'] = (df['trend_short'].astype(int) * 0.5 + 
                           df['trend_medium'].astype(int) * 0.8 + 
                           df['trend_long'].astype(int) * 1.0)
        
        # 计算RSI指标
        rsi_period = self.rsi_period
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 使用EMA计算平均涨跌幅
        avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
        
        # 处理除以0的情况
        rs = avg_gain / avg_loss.replace(0, float('inf'))
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 处理无穷大和NaN值
        df['rsi'] = df['rsi'].replace([np.inf, -np.inf], [100, 0])
        df['rsi'] = df['rsi'].fillna(50)  # 用中性值填充NaN
        
        # 确保RSI值在0-100范围内
        df['rsi'] = df['rsi'].clip(0, 100)
        
        # 计算阻力位和支撑位 (用于突破逻辑)
        lookback = 20
        df['resistance'] = df['high'].rolling(lookback).max()
        df['support'] = df['low'].rolling(lookback).min()
        
        # 识别有效突破
        volume_factor = self.volume_threshold
        df['volume_ma'] = df['volume'].rolling(lookback).mean()
        df['breakout'] = ((df['close'] > df['resistance'].shift(1)) & 
                          (df['volume'] > df['volume_ma'] * volume_factor)).astype(int)
        
        # 计算波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # 计算动量
        df['momentum'] = df['close'].pct_change(periods=10)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame
        """
        # 计算技术指标
        df = self.calculate_indicators(data)
        
        # 根据市场环境调整参数
        self._adjust_parameters_for_market()
        
        # 初始化信号列为float类型
        df['signal'] = pd.Series(0.0, index=df.index)
        
        # 基础信号 - CPGW金叉死叉
        self._generate_base_signals(df)
        
        # 添加趋势跟踪能力 - 根据趋势强度调整信号
        self._enhance_with_trend_following(df)
        
        # 添加RSI超卖反弹信号 - 用于增强牛市买入信号
        self._enhance_with_rsi_signals(df)
        
        # 添加突破买入信号
        self._add_breakout_signals(df)
        
        # 填充NaN值
        df['signal'] = df['signal'].fillna(0.0)
        
        # 确保信号值在合理范围内
        df['signal'] = df['signal'].clip(-1.5, 1.5)
        
        # 如果启用市场环境适配，调整信号
        if self.market_adaptation:
            df = self.adjust_for_market_regime(df, df)
            
        # 最后再次确保信号值在合理范围内
        df['signal'] = df['signal'].clip(-1.5, 1.5)
        
        return df
    
    def _generate_base_signals(self, df: pd.DataFrame) -> None:
        """
        生成基础的CPGW策略信号
        
        参数:
            df: 计算好指标的DataFrame
        """
        # 计算CPGW技术条件 - 金叉和死叉
        golden_cross = (df['cpgw_diff'] > df['cpgw_signal']) & (df['cpgw_diff'].shift(1) <= df['cpgw_signal'].shift(1))
        death_cross = (df['cpgw_diff'] < df['cpgw_signal']) & (df['cpgw_diff'].shift(1) >= df['cpgw_signal'].shift(1))
        
        # 增加信号持续性判断
        for i in range(3, len(df)):
            # 买入条件：金叉且柱状图连续3天为正
            buy_condition = (golden_cross.iloc[i] and 
                           all(df['cpgw_histogram'].iloc[i-2:i+1] > 0) and
                           df['cpgw_diff'].iloc[i] > df['cpgw_diff'].iloc[i-1])
            
            # 卖出条件：死叉且柱状图连续3天为负
            sell_condition = (death_cross.iloc[i] and 
                            all(df['cpgw_histogram'].iloc[i-2:i+1] < 0) and
                            df['cpgw_diff'].iloc[i] < df['cpgw_diff'].iloc[i-1])
            
            # 生成信号
            if buy_condition:
                df.loc[df.index[i], 'signal'] = 0.8  # 降低初始信号强度
            elif sell_condition:
                df.loc[df.index[i], 'signal'] = -0.8
    
    def _enhance_with_trend_following(self, df: pd.DataFrame) -> None:
        """
        根据趋势强度增强信号
        
        参数:
            df: 计算好指标的DataFrame
        """
        # 在强趋势中增强信号
        for i in range(len(df)):
            if i == 0:
                continue
                
            # 如果存在信号且趋势得分高，增强信号
            if df['signal'].iloc[i] > 0:
                if df['trend_score'].iloc[i] >= 1.5:  # 降低趋势得分阈值
                    # 强上升趋势中的买入信号更可靠
                    df.loc[df.index[i], 'signal'] = min(1.1, df['signal'].iloc[i] * 1.1)  # 降低信号增强倍数
                elif df['trend_score'].iloc[i] <= 0.3:  # 降低弱趋势阈值
                    # 趋势较弱，降低信号强度
                    df.loc[df.index[i], 'signal'] = 0.3
            elif df['signal'].iloc[i] < 0:
                if df['trend_score'].iloc[i] <= 0.3:  # 降低趋势得分阈值
                    # 强下降趋势中的卖出信号更可靠
                    df.loc[df.index[i], 'signal'] = max(-1.1, df['signal'].iloc[i] * 1.1)
                elif df['trend_score'].iloc[i] >= 1.5:
                    # 趋势向上，降低卖出信号强度
                    df.loc[df.index[i], 'signal'] = -0.3
    
    def _enhance_with_rsi_signals(self, df: pd.DataFrame) -> None:
        """
        添加RSI超卖反弹信号，优化牛市买入条件
        
        参数:
            df: 计算好指标的DataFrame
        """
        rsi_oversold = self.rsi_oversold
        
        # 添加RSI超卖反弹买入条件
        for i in range(5, len(df)):  # 增加观察期
            # RSI超卖反弹 (RSI从低于超卖线反弹且连续上涨)
            rsi_bounce = (df['rsi'].iloc[i-2:i+1].is_monotonic_increasing and  # 确保RSI连续上涨
                         df['rsi'].iloc[i-2] < rsi_oversold and
                         df['rsi'].iloc[i] > rsi_oversold)
            
            # 价格在支撑位附近反弹
            near_support = df['close'].iloc[i] < df['support'].iloc[i] * 1.03  # 增加范围到3%
            
            # 动量确认 (要求更强的动量)
            momentum_confirm = (df['momentum'].iloc[i] > 0.01 and  # 要求至少1%的动量
                              df['momentum'].iloc[i] > df['momentum'].iloc[i-1])
            
            # 结合反弹、支撑位和动量条件作为新买入点
            if rsi_bounce and near_support and momentum_confirm and df['signal'].iloc[i] == 0:
                # 检查当前市场环境
                if self.get_market_regime(df.iloc[:i+1]) == MarketRegime.BULLISH:
                    # 牛市环境下的RSI超卖反弹是不错的买入机会
                    df.loc[df.index[i], 'signal'] = 0.5  # 降低初始信号强度
    
    def _add_breakout_signals(self, df: pd.DataFrame) -> None:
        """
        添加突破买入信号
        
        参数:
            df: 计算好指标的DataFrame
        """
        # 当价格突破阻力位且成交量放大时
        breakout_condition = df['breakout'] == 1
        
        # 如果趋势得分>=1.8且动量为正，增强突破买入信号的可信度
        strong_breakout = breakout_condition & (df['trend_score'] >= 1.8) & (df['momentum'] > 0)
        
        # 为突破信号添加买入信号
        df.loc[breakout_condition, 'signal'] = np.where(df.loc[breakout_condition, 'signal'] == 0, 0.8, df.loc[breakout_condition, 'signal'])
        df.loc[strong_breakout, 'signal'] = np.where(df.loc[strong_breakout, 'signal'] == 0.8, 1.0, df.loc[strong_breakout, 'signal'])
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            'cpgw_diff': {
                'name': 'CPGW差值',
                'description': f"{self.macd_fast}和{self.macd_slow}周期EMA的差值",
                'type': 'trend',
                'time_scale': 'short',
                'min_value': None,
                'max_value': None
            },
            'cpgw_signal': {
                'name': 'CPGW信号线',
                'description': f"{self.macd_signal}周期EMA平滑的差值",
                'type': 'trend',
                'time_scale': 'short', 
                'min_value': None,
                'max_value': None
            },
            'cpgw_histogram': {
                'name': 'CPGW柱状图',
                'description': '差值与信号线的差值',
                'type': 'momentum',
                'time_scale': 'short',
                'min_value': None,
                'max_value': None
            },
            'trend_score': {
                'name': '趋势得分',
                'description': '多周期趋势综合得分',
                'type': 'trend',
                'time_scale': 'multi',
                'min_value': 0,
                'max_value': 3
            },
            'rsi': {
                'name': 'RSI',
                'description': f"{self.rsi_period}周期相对强弱指标",
                'type': 'oscillator',
                'time_scale': 'medium',
                'min_value': 0, 
                'max_value': 100
            },
            'breakout': {
                'name': '价格突破',
                'description': '价格突破阻力位且成交量放大',
                'type': 'breakout',
                'time_scale': 'short',
                'min_value': 0,
                'max_value': 1
            }
        }
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, SignalComponent]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        # 计算技术指标
        df = self.calculate_indicators(data)
        
        # 创建信号组件字典
        components = {}
        
        # 从signal_interface导入需要的类
        from .signal_interface import SignalMetadata, SignalComponent, SignalType, SignalTimeframe
        
        # 构建CPGW差值组件
        if 'cpgw_diff' in df.columns:
            components['cpgw_diff'] = SignalComponent(
                series=df['cpgw_diff'],
                metadata=SignalMetadata(
                    name='CPGW差值',
                    description=f"{self.macd_fast}和{self.macd_slow}周期EMA的差值",
                    signal_type=SignalType.TREND,
                    timeframe=SignalTimeframe.DAILY,
                    weight=0.7
                )
            )
        
        # 构建CPGW信号线组件
        if 'cpgw_signal' in df.columns:
            components['cpgw_signal'] = SignalComponent(
                series=df['cpgw_signal'],
                metadata=SignalMetadata(
                    name='CPGW信号线',
                    description=f"{self.macd_signal}周期EMA平滑的差值",
                    signal_type=SignalType.TREND,
                    timeframe=SignalTimeframe.DAILY,
                    weight=0.6
                )
            )
        
        # 构建CPGW柱状图组件
        if 'cpgw_histogram' in df.columns:
            components['cpgw_histogram'] = SignalComponent(
                series=df['cpgw_histogram'],
                metadata=SignalMetadata(
                    name='CPGW柱状图',
                    description='差值与信号线的差值',
                    signal_type=SignalType.MOMENTUM,
                    timeframe=SignalTimeframe.DAILY,
                    weight=0.8
                )
            )
        
        # 构建趋势得分组件
        if 'trend_score' in df.columns:
            components['trend_score'] = SignalComponent(
                series=df['trend_score'],
                metadata=SignalMetadata(
                    name='趋势得分',
                    description='多周期趋势综合得分',
                    signal_type=SignalType.TREND,
                    timeframe=SignalTimeframe.DAILY,
                    weight=0.9
                )
            )
        
        # 构建RSI组件
        if 'rsi' in df.columns:
            components['rsi'] = SignalComponent(
                series=df['rsi'],
                metadata=SignalMetadata(
                    name='RSI',
                    description=f"{self.rsi_period}周期相对强弱指标",
                    signal_type=SignalType.MOMENTUM,
                    timeframe=SignalTimeframe.DAILY,
                    weight=0.6
                )
            )
        
        # 构建突破组件
        if 'breakout' in df.columns:
            components['breakout'] = SignalComponent(
                series=df['breakout'],
                metadata=SignalMetadata(
                    name='价格突破',
                    description='价格突破阻力位且成交量放大',
                    signal_type=SignalType.BREAKOUT,
                    timeframe=SignalTimeframe.DAILY,
                    weight=1.0
                )
            )
            
        return components
    
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据市场环境调整信号
        
        参数:
            data: 原始数据
            signals: 信号数据
            
        返回:
            调整后的信号
        """
        # 首先调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 获取当前市场环境
        regime = self.get_market_regime(data)
        
        # 根据CPGW策略特性进行额外的调整
        if regime == MarketRegime.BEARISH:
            self.logger.info("CPGW策略: 熊市环境，增强卖出信号权重")
            # 在熊市中，增强卖出信号，弱化买入信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 当快线距离慢线较近时，忽略弱买入信号
                    diff_pct = adjusted_signals['cpgw_diff'].iloc[i] / adjusted_signals['close'].iloc[i] * 100
                    if diff_pct < 0.5:  # 增加阈值到0.5%
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0.0
                elif adjusted_signals['signal'].iloc[i] < 0:
                    # 增强卖出信号
                    if adjusted_signals['trend_score'].iloc[i] <= 0.3:  # 降低趋势得分要求
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = max(
                            adjusted_signals['signal'].iloc[i] * 1.05,  # 降低增强倍数
                            -1.2 if i == 0 else adjusted_signals['signal'].iloc[i-1] - 1.0  # 降低最大变化
                        )
                
        elif regime == MarketRegime.BULLISH:
            self.logger.info("CPGW策略: 牛市环境，增强买入信号权重")
            # 在牛市中，增强买入信号，弱化卖出信号
            for i in range(len(adjusted_signals)):
                # 增强买入信号
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 趋势得分高的买入信号更可靠
                    if adjusted_signals['trend_score'].iloc[i] >= 1.5:  # 降低趋势得分要求
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = min(
                            adjusted_signals['signal'].iloc[i] * 1.05,  # 降低增强倍数
                            1.2 if i == 0 else adjusted_signals['signal'].iloc[i-1] + 1.0  # 降低最大变化
                        )
                
                # 弱化卖出信号
                elif adjusted_signals['signal'].iloc[i] < 0:
                    # 当快线距离慢线较近时，忽略弱卖出信号
                    diff_pct = abs(adjusted_signals['cpgw_diff'].iloc[i]) / adjusted_signals['close'].iloc[i] * 100
                    if diff_pct < 0.5:  # 增加阈值到0.5%
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0.0
        
        elif regime == MarketRegime.RANGING:
            self.logger.info("CPGW策略: 震荡环境，优化区间交易")
            # 在震荡环境中，在支撑位买入，在阻力位卖出
            for i in range(1, len(adjusted_signals)):
                close_price = adjusted_signals['close'].iloc[i]
                support = adjusted_signals['support'].iloc[i]
                resistance = adjusted_signals['resistance'].iloc[i]
                
                # 价格接近支撑位且RSI低，考虑买入
                near_support = close_price < support * 1.03  # 增加范围到3%
                low_rsi = adjusted_signals['rsi'].iloc[i] < 30  # 降低RSI阈值
                
                # 价格接近阻力位且RSI高，考虑卖出
                near_resistance = close_price > resistance * 0.97  # 增加范围到3%
                high_rsi = adjusted_signals['rsi'].iloc[i] > 70  # 提高RSI阈值
                
                if near_support and low_rsi and adjusted_signals['signal'].iloc[i] == 0:
                    # 限制信号变化
                    new_signal = 0.5  # 降低初始信号强度
                    if i > 0:
                        new_signal = min(new_signal, adjusted_signals['signal'].iloc[i-1] + 1.0)
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = new_signal
                elif near_resistance and high_rsi and adjusted_signals['signal'].iloc[i] == 0:
                    # 限制信号变化
                    new_signal = -0.5  # 降低初始信号强度
                    if i > 0:
                        new_signal = max(new_signal, adjusted_signals['signal'].iloc[i-1] - 1.0)
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = new_signal
        
        elif regime == MarketRegime.VOLATILE:
            self.logger.info("CPGW策略: 高波动环境，减少信号频率")
            # 在高波动环境中，减少信号频率，只保留强信号
            for i in range(len(adjusted_signals)):
                if abs(adjusted_signals['signal'].iloc[i]) <= 0.5:  # 降低阈值到0.5
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0.0
                    
        # 确保所有信号变化不超过1.0
        for i in range(1, len(adjusted_signals)):
            prev_signal = adjusted_signals['signal'].iloc[i-1]
            curr_signal = adjusted_signals['signal'].iloc[i]
            max_change = 1.0  # 降低最大变化幅度
            
            if abs(curr_signal - prev_signal) > max_change:
                if curr_signal > prev_signal:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = prev_signal + max_change
                else:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = prev_signal - max_change
        
        return adjusted_signals 