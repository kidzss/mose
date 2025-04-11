import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from .strategy_base import Strategy, MarketRegime
from .signal_interface import (
    SignalType, SignalTimeframe, SignalStrength, 
    SignalMetadata, SignalComponent, SignalCombiner
)


class BreakoutStrategy(Strategy):
    """
    突破策略
    
    在价格突破重要支撑位/阻力位时产生交易信号:
    1. 突破阻力位 = 买入信号
    2. 跌破支撑位 = 卖出信号
    
    支撑/阻力位计算方法:
    - 近期高点/低点
    - 关键移动平均线
    - 布林带突破
    - 成交量确认
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """初始化策略"""
        # 设置默认参数
        default_params = {
            # 突破参数
            'resistance_lookback': 20,    # 寻找阻力位的回看周期
            'support_lookback': 20,       # 寻找支撑位的回看周期
            'breakout_threshold': 1.0,    # 突破阈值百分比
            
            # 移动平均线参数
            'ma_short': 20,               # 短期均线
            'ma_medium': 50,              # 中期均线
            'ma_long': 200,               # 长期均线
            
            # ATR参数
            'atr_period': 14,             # ATR周期
            'atr_multiplier': 1.5,        # ATR乘数，用于确认突破
            
            # 成交量参数
            'volume_confirm_ratio': 1.5,  # 成交量确认比率
            'volume_ma_period': 20,       # 成交量移动平均周期
            
            # 其他参数
            'use_market_regime': True,    # 是否使用市场环境
            'consolidation_days': 5,      # 盘整天数，用于确认突破前的盘整
            'confirmation_days': 2,       # 确认突破的天数
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
            
        # 初始化基类
        super().__init__('BreakoutStrategy', default_params)
        
        # 初始化信号元数据
        self._initialize_signal_metadata()
    
    def _initialize_signal_metadata(self):
        """初始化信号元数据"""
        self.signal_metadata = {
            'price_breakout': SignalMetadata(
                name="价格突破",
                description="价格突破重要支撑/阻力位",
                signal_type=SignalType.BREAKOUT,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='minmax'
            ),
            'volume_confirm': SignalMetadata(
                name="成交量确认",
                description="成交量是否确认突破",
                signal_type=SignalType.BREAKOUT,
                timeframe=SignalTimeframe.DAILY,
                weight=0.7,
                normalization='minmax'
            ),
            'volatility_breakout': SignalMetadata(
                name="波动率突破",
                description="基于ATR的波动率突破",
                signal_type=SignalType.BREAKOUT,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='minmax'
            )
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 获取参数
        ma_short = self.parameters['ma_short']
        ma_medium = self.parameters['ma_medium']
        ma_long = self.parameters['ma_long']
        atr_period = self.parameters['atr_period']
        volume_ma_period = self.parameters['volume_ma_period']
        
        # 计算移动平均线
        df['ma_short'] = df['close'].rolling(window=ma_short).mean()
        df['ma_medium'] = df['close'].rolling(window=ma_medium).mean()
        df['ma_long'] = df['close'].rolling(window=ma_long).mean()
        
        # 计算高点和低点
        resistance_lookback = self.parameters['resistance_lookback']
        support_lookback = self.parameters['support_lookback']
        
        # 计算近期高点
        df['recent_high'] = df['high'].rolling(window=resistance_lookback).max()
        # 计算近期低点
        df['recent_low'] = df['low'].rolling(window=support_lookback).min()
        
        # 计算与高点和低点的距离（百分比）
        df['dist_to_high'] = (df['close'] / df['recent_high'] - 1) * 100
        df['dist_to_low'] = (df['close'] / df['recent_low'] - 1) * 100
        
        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.DataFrame(data={
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        df['atr'] = tr.rolling(window=atr_period).mean()
        
        # 计算突破标志
        atr_multiplier = self.parameters['atr_multiplier']
        df['upper_band'] = df['recent_high'] + (df['atr'] * atr_multiplier)
        df['lower_band'] = df['recent_low'] - (df['atr'] * atr_multiplier)
        
        # 计算波动率缩小指标 (通过ATR的变化率)
        df['atr_change'] = df['atr'].pct_change(periods=5)
        
        # 计算成交量相关指标
        if 'volume' in df.columns:
            # 计算成交量移动平均
            df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
            # 计算相对成交量
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # 计算价格和成交量趋势一致性
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            df['price_volume_agreement'] = np.sign(price_change) * np.sign(volume_change)
        
        # 计算盘整指标 (高低点之间的范围变窄)
        consolidation_days = self.parameters['consolidation_days']
        df['price_range'] = (df['high'].rolling(window=consolidation_days).max() - 
                           df['low'].rolling(window=consolidation_days).min()) / df['close']
        
        # 计算布林带
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_breakout_up'] = df['close'] > df['bb_upper']
        df['bb_breakout_down'] = df['close'] < df['bb_lower']
        
        return df
    
    def is_consolidating(self, data: pd.DataFrame, index: int) -> bool:
        """判断市场是否处于盘整状态"""
        if index < self.parameters['consolidation_days']:
            return False
            
        price_range = data['price_range'].iloc[index]
        avg_range = data['price_range'].iloc[index-10:index].mean()
        
        # 如果当前范围比平均范围小，说明在盘整
        return price_range < avg_range * 0.7
    
    def is_volume_confirming(self, data: pd.DataFrame, index: int, direction: int) -> bool:
        """判断成交量是否确认突破"""
        if 'volume' not in data.columns or 'volume_ma' not in data.columns:
            return True  # 如果没有成交量数据，默认确认
            
        # 获取成交量比率
        volume_ratio = data['relative_volume'].iloc[index]
        
        # 判断成交量是否足够放大
        volume_confirm = volume_ratio > self.parameters['volume_confirm_ratio']
        
        # 确认成交量与价格方向一致
        if direction > 0:  # 上涨突破
            return volume_confirm and data['price_volume_agreement'].iloc[index] > 0
        else:  # 下跌突破
            return volume_confirm and data['price_volume_agreement'].iloc[index] < 0
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        # 计算技术指标
        df = self.calculate_indicators(data.copy())
        
        # 获取参数
        breakout_threshold = self.parameters['breakout_threshold']
        confirmation_days = self.parameters['confirmation_days']
        
        # 初始化信号列
        df['signal'] = 0
        df['price_breakout'] = 0  # 价格突破指标
        df['volume_confirm'] = 0  # 成交量确认指标
        df['volatility_breakout'] = 0  # 波动率突破指标
        
        # 突破信号生成逻辑
        for i in range(max(self.parameters['ma_long'], self.parameters['resistance_lookback']), len(df)):
            # 判断之前是否处于盘整状态
            if self.is_consolidating(df, i-1):
                # 向上突破
                if (df['close'].iloc[i] > df['recent_high'].iloc[i-1] * (1 + breakout_threshold/100)):
                    # 检查是否有足够的确认天数
                    confirmed = True
                    for j in range(1, min(confirmation_days+1, i)):
                        if df['close'].iloc[i-j] <= df['recent_high'].iloc[i-j-1]:
                            confirmed = False
                            break
                    
                    if confirmed:
                        # 检查成交量确认
                        volume_confirmed = self.is_volume_confirming(df, i, 1)
                        
                        # 记录突破指标
                        df.loc[df.index[i], 'price_breakout'] = 1
                        df.loc[df.index[i], 'volume_confirm'] = 1 if volume_confirmed else 0
                        
                        # 检查波动率突破
                        if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
                            df.loc[df.index[i], 'volatility_breakout'] = 1
                            
                        # 如果有足够的确认，产生买入信号
                        if volume_confirmed or df['volatility_breakout'].iloc[i] > 0:
                            df.loc[df.index[i], 'signal'] = 1
                            
                # 向下突破
                elif (df['close'].iloc[i] < df['recent_low'].iloc[i-1] * (1 - breakout_threshold/100)):
                    # 检查是否有足够的确认天数
                    confirmed = True
                    for j in range(1, min(confirmation_days+1, i)):
                        if df['close'].iloc[i-j] >= df['recent_low'].iloc[i-j-1]:
                            confirmed = False
                            break
                    
                    if confirmed:
                        # 检查成交量确认
                        volume_confirmed = self.is_volume_confirming(df, i, -1)
                        
                        # 记录突破指标
                        df.loc[df.index[i], 'price_breakout'] = -1
                        df.loc[df.index[i], 'volume_confirm'] = 1 if volume_confirmed else 0
                        
                        # 检查波动率突破
                        if df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
                            df.loc[df.index[i], 'volatility_breakout'] = -1
                            
                        # 如果有足够的确认，产生卖出信号
                        if volume_confirmed or df['volatility_breakout'].iloc[i] < 0:
                            df.loc[df.index[i], 'signal'] = -1
            
            # 移动平均线交叉突破
            elif (df['ma_short'].iloc[i-1] <= df['ma_medium'].iloc[i-1] and 
                 df['ma_short'].iloc[i] > df['ma_medium'].iloc[i]):
                # 短期均线上穿中期均线，可能产生买入信号
                # 需要额外确认
                if df['close'].iloc[i] > df['ma_medium'].iloc[i] * 1.01:  # 确认有足够的突破空间
                    if 'volume' not in df.columns or df['relative_volume'].iloc[i] > 1.0:
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'price_breakout'] = 0.5  # 均线突破强度较弱
            
            elif (df['ma_short'].iloc[i-1] >= df['ma_medium'].iloc[i-1] and 
                 df['ma_short'].iloc[i] < df['ma_medium'].iloc[i]):
                # 短期均线下穿中期均线，可能产生卖出信号
                # 需要额外确认
                if df['close'].iloc[i] < df['ma_medium'].iloc[i] * 0.99:  # 确认有足够的突破空间
                    if 'volume' not in df.columns or df['relative_volume'].iloc[i] > 1.0:
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'price_breakout'] = int(-0.5)  # 均线突破强度较弱
            
            # 布林带突破
            elif df['bb_breakout_up'].iloc[i] and not df['bb_breakout_up'].iloc[i-1]:
                # 布林带上轨突破
                if 'volume' not in df.columns or df['relative_volume'].iloc[i] > 1.2:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'volatility_breakout'] = int(0.7)
            
            elif df['bb_breakout_down'].iloc[i] and not df['bb_breakout_down'].iloc[i-1]:
                # 布林带下轨突破
                if 'volume' not in df.columns or df['relative_volume'].iloc[i] > 1.2:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'volatility_breakout'] = int(-0.7)
                    
        # 获取信号组件
        signal_components = self.extract_signal_components(df)
        
        # 创建信号组合器
        combiner = SignalCombiner(signal_components)
        
        # 获取组合信号序列
        combined_signal = combiner.combine(method='weighted_average')
        
        # 添加组合信号到DataFrame
        df['combined_signal'] = combined_signal
        
        # 如果启用市场环境适配，调整信号
        if self.parameters['use_market_regime']:
            df = self.adjust_for_market_regime(df, df)
            
        return df
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, SignalComponent]:
        """提取并标准化策略的核心信号组件"""
        components = {}
        
        # 1. 价格突破组件
        if 'price_breakout' in data.columns:
            components['price_breakout'] = SignalComponent(
                series=data['price_breakout'],
                metadata=self.signal_metadata['price_breakout']
            )
        
        # 2. 成交量确认组件
        if 'volume_confirm' in data.columns:
            components['volume_confirm'] = SignalComponent(
                series=data['volume_confirm'],
                metadata=self.signal_metadata['volume_confirm']
            )
        
        # 3. 波动率突破组件
        if 'volatility_breakout' in data.columns:
            components['volatility_breakout'] = SignalComponent(
                series=data['volatility_breakout'],
                metadata=self.signal_metadata['volatility_breakout']
            )
            
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号组件的元数据"""
        return {name: metadata.to_dict() for name, metadata in self.signal_metadata.items()}
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """判断当前市场环境"""
        # 使用基类的市场环境检测
        regime = super().get_market_regime(data)
        
        try:
            if len(data) < 50:
                return regime
                
            # 检查盘整状态
            recent_consolidation = data['price_range'].iloc[-5:].mean()
            avg_range = data['price_range'].iloc[-20:-5].mean()
            
            # 如果近期波动范围明显小于平均范围，可能是盘整市场
            if recent_consolidation < avg_range * 0.6:
                self.logger.info("检测到盘整市场，有潜在突破机会")
                return MarketRegime.RANGING
                
            # 检查高波动状态
            if 'atr' in data.columns:
                latest_atr = data['atr'].iloc[-1]
                avg_atr = data['atr'].iloc[-20:].mean()
                
                if latest_atr > avg_atr * 1.5:
                    return MarketRegime.VOLATILE
        except Exception as e:
            self.logger.error(f"市场环境判断出错: {e}")
            
        return regime
    
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """根据市场环境调整信号"""
        # 调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 在盘整市场中，突破策略更有效
        if regime == MarketRegime.RANGING:
            self.logger.info("盘整市场环境，增强突破信号")
            # 无需特殊处理，这是突破策略的理想环境
            
        # 在高波动环境中，需要更多确认
        elif regime == MarketRegime.VOLATILE:
            self.logger.info("高波动环境，需要额外确认突破信号")
            
            for i in range(len(adjusted_signals)):
                # 检查是否有足够的成交量确认
                if 'volume_confirm' in adjusted_signals.columns and 'volume' in adjusted_signals.columns:
                    if (adjusted_signals['signal'].iloc[i] != 0 and 
                        (adjusted_signals['volume_confirm'].iloc[i] == 0 or 
                         adjusted_signals['relative_volume'].iloc[i] < 2.0)):
                        # 在高波动环境需要更大的成交量确认
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        
        # 在趋势明确的市场中，突破策略可能产生更多假信号
        elif regime in (MarketRegime.BULLISH, MarketRegime.BEARISH):
            trend_direction = 1 if regime == MarketRegime.BULLISH else -1
            
            for i in range(len(adjusted_signals)):
                # 如果信号与趋势方向相反，降低信号权重或忽略
                if adjusted_signals['signal'].iloc[i] * trend_direction < 0:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
        return adjusted_signals
    
    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """计算仓位大小"""
        # 调用基类的仓位计算
        base_position = super().get_position_size(data, signal)
        
        # 基于成交量和波动率调整仓位
        if 'relative_volume' in data.columns and 'volume_confirm' in data.columns:
            volume_factor = data['relative_volume'].iloc[-1] / 2.0  # 标准化成交量因子
            
            # 成交量越大，确认度越高，仓位越大
            return min(base_position * min(volume_factor, 1.5), 1.0)
            
        return base_position
    
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止损价格"""
        # 获取ATR
        if 'atr' in data.columns:
            latest_atr = data['atr'].iloc[-1]
            
            # 使用ATR的倍数作为止损距离
            atr_multiplier = self.parameters['atr_multiplier']
            
            if position > 0:  # 多头止损
                return entry_price - (latest_atr * atr_multiplier)
            elif position < 0:  # 空头止损
                return entry_price + (latest_atr * atr_multiplier)
                
        # 如果没有ATR数据，使用基类的止损计算方法
        return super().get_stop_loss(data, entry_price, position)
    
    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止盈价格"""
        # 获取ATR
        if 'atr' in data.columns:
            latest_atr = data['atr'].iloc[-1]
            
            # 使用ATR的倍数作为止盈目标
            # 通常止盈目标是止损距离的2-3倍
            atr_multiplier = self.parameters['atr_multiplier'] * 2.5
            
            if position > 0:  # 多头止盈
                return entry_price + (latest_atr * atr_multiplier)
            elif position < 0:  # 空头止盈
                return entry_price - (latest_atr * atr_multiplier)
                
        # 如果没有ATR数据，使用基类的止盈计算方法
        return super().get_take_profit(data, entry_price, position) 