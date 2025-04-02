import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List

from .strategy_base import Strategy, MarketRegime
from .signal_interface import (
    SignalType, SignalTimeframe, SignalStrength, 
    SignalMetadata, SignalComponent, SignalCombiner
)


class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    
    当价格偏离其长期均值过远时，认为会有回归趋势，产生反向交易信号:
    1. 价格大幅高于移动平均线 = 卖出信号（预期回归均值）
    2. 价格大幅低于移动平均线 = 买入信号（预期回归均值）
    
    同时使用波动率和超买超卖指标进行信号确认
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """初始化策略"""
        # 设置默认参数
        default_params = {
            # 均线参数
            'ma_short': 20,            # 短期均线周期
            'ma_long': 50,             # 长期均线周期
            'std_dev_multiplier': 2.0, # 标准差乘数，用于计算偏离度
            
            # RSI参数
            'rsi_length': 14,          # RSI计算周期
            'rsi_overbought': 70,      # RSI超买阈值
            'rsi_oversold': 30,        # RSI超卖阈值
            
            # 布林带参数
            'bb_length': 20,           # 布林带周期
            'bb_std_dev': 2.0,         # 布林带标准差
            
            # 信号阈值
            'price_deviation_pct': 5.0,  # 价格偏离百分比阈值
            
            # 其他参数
            'use_market_regime': True,   # 是否使用市场环境
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
            
        # 初始化基类
        super().__init__('MeanReversionStrategy', default_params)
        
        # 初始化信号元数据
        self._initialize_signal_metadata()
    
    def _initialize_signal_metadata(self):
        """初始化信号元数据"""
        self.signal_metadata = {
            'price_deviation': SignalMetadata(
                name="价格偏离",
                description="价格与移动平均线的偏离度",
                signal_type=SignalType.MEAN_REVERSION,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='zscore'
            ),
            'rsi': SignalMetadata(
                name="RSI",
                description="相对强弱指数",
                signal_type=SignalType.MEAN_REVERSION,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='minmax',
                normalization_params={'min': 0, 'max': 100}
            ),
            'bb_position': SignalMetadata(
                name="布林带位置",
                description="价格在布林带中的相对位置",
                signal_type=SignalType.MEAN_REVERSION,
                timeframe=SignalTimeframe.DAILY,
                weight=0.7,
                normalization='minmax',
                normalization_params={'min': -1, 'max': 1}
            )
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 获取参数
        ma_short = self.parameters['ma_short']
        ma_long = self.parameters['ma_long']
        rsi_length = self.parameters['rsi_length']
        bb_length = self.parameters['bb_length']
        bb_std_dev = self.parameters['bb_std_dev']
        
        # 计算移动平均线
        df['ma_short'] = df['close'].rolling(window=ma_short).mean()
        df['ma_long'] = df['close'].rolling(window=ma_long).mean()
        
        # 计算价格与长期均线的偏离百分比
        df['price_deviation_pct'] = (df['close'] - df['ma_long']) / df['ma_long'] * 100
        
        # 计算价格与长期均线的标准差
        df['std_dev'] = df['close'].rolling(window=ma_long).std()
        df['deviation_zscore'] = (df['close'] - df['ma_long']) / df['std_dev']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
        rs = avg_gain / avg_loss.replace(0, 0.00001)  # 避免除以零
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['bb_middle'] = df['close'].rolling(window=bb_length).mean()
        df['bb_std'] = df['close'].rolling(window=bb_length).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std_dev)
        
        # 计算价格在布林带中的相对位置 (-1至1)
        df['bb_position'] = (df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower']) * 2
        
        # 计算波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        # 计算技术指标
        df = self.calculate_indicators(data.copy())
        
        # 获取信号组件
        signal_components = self.extract_signal_components(df)
        
        # 创建信号组合器
        combiner = SignalCombiner(signal_components)
        
        # 获取组合信号序列 (注意：均值回归策略中，信号方向与价格偏离方向相反)
        combined_signal = -1 * combiner.combine(method='weighted_average')
        
        # 添加组合信号到DataFrame
        df['combined_signal'] = combined_signal
        
        # 获取参数
        price_deviation_pct = self.parameters['price_deviation_pct']
        std_dev_multiplier = self.parameters['std_dev_multiplier']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        ma_long = self.parameters['ma_long']  # 从参数中获取ma_long值
        
        # 初始化信号列
        df['signal'] = 0
        
        # 生成均值回归信号
        for i in range(ma_long, len(df)):
            # 价格偏离均线
            if df['price_deviation_pct'].iloc[i] > price_deviation_pct:
                # 价格显著高于均线，可能产生卖出信号
                if df['rsi'].iloc[i] > rsi_overbought:
                    # RSI确认超买，产生卖出信号
                    df.loc[df.index[i], 'signal'] = -1
                    
            elif df['price_deviation_pct'].iloc[i] < -price_deviation_pct:
                # 价格显著低于均线，可能产生买入信号
                if df['rsi'].iloc[i] < rsi_oversold:
                    # RSI确认超卖，产生买入信号
                    df.loc[df.index[i], 'signal'] = 1
                    
            # 基于布林带的信号
            elif df['close'].iloc[i] > df['bb_upper'].iloc[i]:
                # 价格突破上轨，可能产生卖出信号
                df.loc[df.index[i], 'signal'] = -1
                
            elif df['close'].iloc[i] < df['bb_lower'].iloc[i]:
                # 价格突破下轨，可能产生买入信号
                df.loc[df.index[i], 'signal'] = 1
        
        # 如果启用市场环境适配，调整信号
        if self.parameters['use_market_regime']:
            df = self.adjust_for_market_regime(df, df)
            
        return df
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, SignalComponent]:
        """提取并标准化策略的核心信号组件"""
        components = {}
        
        # 1. 价格偏离组件
        if 'deviation_zscore' in data.columns:
            components['price_deviation'] = SignalComponent(
                series=data['deviation_zscore'],
                metadata=self.signal_metadata['price_deviation']
            )
        
        # 2. RSI组件
        if 'rsi' in data.columns:
            components['rsi'] = SignalComponent(
                series=data['rsi'],
                metadata=self.signal_metadata['rsi']
            )
        
        # 3. 布林带位置组件
        if 'bb_position' in data.columns:
            components['bb_position'] = SignalComponent(
                series=data['bb_position'],
                metadata=self.signal_metadata['bb_position']
            )
            
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号组件的元数据"""
        return {name: metadata.to_dict() for name, metadata in self.signal_metadata.items()}
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """判断当前市场环境"""
        # 使用基类的市场环境检测
        regime = super().get_market_regime(data)
        
        # 增强版市场环境判断
        if len(data) < 50:
            return regime
            
        # 1. 判断波动率
        if 'volatility' in data.columns:
            latest_volatility = data['volatility'].iloc[-1]
            avg_volatility = data['volatility'].mean()
            
            # 高波动环境下，均值回归策略可能不适用
            if latest_volatility > avg_volatility * 1.5:
                return MarketRegime.VOLATILE
                
            # 低波动环境下，均值回归策略可能更有效
            if latest_volatility < avg_volatility * 0.5:
                return MarketRegime.LOW_VOLATILITY
                
        # 2. 判断趋势强度
        if 'price_deviation_pct' in data.columns:
            # 计算价格偏离均线的持续性
            deviation_series = data['price_deviation_pct'].tail(20)
            
            # 如果价格持续高于均线，可能是强烈的趋势市场
            if deviation_series.mean() > 5 and (deviation_series > 0).all():
                return MarketRegime.BULLISH
                
            # 如果价格持续低于均线，可能是强烈的下跌趋势
            if deviation_series.mean() < -5 and (deviation_series < 0).all():
                return MarketRegime.BEARISH
                
        return regime
    
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """根据市场环境调整信号"""
        # 调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 在强趋势市场中，减少或忽略均值回归信号
        if regime in (MarketRegime.BULLISH, MarketRegime.BEARISH):
            self.logger.info(f"强趋势市场环境 ({regime.value})，降低均值回归信号权重")
            
            for i in range(len(adjusted_signals)):
                # 在强烈的牛市中，降低卖出信号强度
                if regime == MarketRegime.BULLISH and adjusted_signals['signal'].iloc[i] < 0:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
                # 在强烈的熊市中，降低买入信号强度
                if regime == MarketRegime.BEARISH and adjusted_signals['signal'].iloc[i] > 0:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        # 在低波动环境中，增强均值回归信号
        elif regime == MarketRegime.LOW_VOLATILITY:
            self.logger.info(f"低波动环境，增强均值回归信号")
            # 无需特殊处理，这是均值回归策略的理想环境
            
        # 在高波动环境中，减少信号
        elif regime == MarketRegime.VOLATILE:
            self.logger.info(f"高波动环境，减少均值回归信号")
            for i in range(len(adjusted_signals)):
                # 简单地减少所有信号
                if abs(adjusted_signals['signal'].iloc[i]) > 0:
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
        return adjusted_signals
        
    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """计算仓位大小"""
        # 调用基类的仓位计算
        base_position = super().get_position_size(data, signal)
        
        # 考虑价格偏离度的绝对值
        if 'price_deviation_pct' in data.columns:
            latest_deviation = abs(data['price_deviation_pct'].iloc[-1])
            
            # 偏离越大，仓位越大（但设置上限）
            deviation_factor = min(latest_deviation / 10.0, 1.5)
            
            # 根据偏离度调整仓位
            adjusted_position = base_position * deviation_factor
            
            # 确保仓位在合理范围内
            return min(adjusted_position, 1.0)
            
        return base_position 