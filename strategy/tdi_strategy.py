import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

from .strategy_base import Strategy, MarketRegime
from .signal_interface import SignalComponent, SignalMetadata, SignalType, SignalTimeframe

class TDIStrategy(Strategy):
    """
    TDI策略 - 交易者动态指标策略
    
    TDI (Traders Dynamic Index) 策略基于RSI指标和其移动平均线，
    通过追踪RSI线、信号线和波动带来生成交易信号。
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化TDI策略
        
        参数:
            parameters: 策略参数字典
        """
        # 设置默认参数
        default_params = {
            'rsi_length': 13,  # RSI计算周期
            'signal_length': 7,  # 信号线计算周期
            'band_length': 34,  # 波动带计算周期
            'band_std': 1.6185,  # 波动带标准差倍数
            'use_market_regime': True  # 是否使用市场环境调整
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
        
        # 初始化基类
        super().__init__('TDIStrategy', default_params)
        
        # 初始化信号元数据
        self.signal_metadata = {
            'rsi': SignalMetadata(
                name='RSI线',
                description=f"{self.parameters['rsi_length']}周期RSI",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=1.0,
                normalization='minmax'
            ),
            'signal': SignalMetadata(
                name='信号线',
                description=f"RSI的{self.parameters['signal_length']}周期平均",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=0.8,
                normalization='minmax'
            ),
            'upper_band': SignalMetadata(
                name='上波动带',
                description='波动带上轨',
                signal_type=SignalType.VOLATILITY,
                timeframe=SignalTimeframe.DAILY,
                weight=0.6,
                normalization='minmax'
            ),
            'lower_band': SignalMetadata(
                name='下波动带',
                description='波动带下轨',
                signal_type=SignalType.VOLATILITY,
                timeframe=SignalTimeframe.DAILY,
                weight=0.6,
                normalization='minmax'
            ),
            'midline': SignalMetadata(
                name='中线',
                description='RSI中线(50)',
                signal_type=SignalType.SUPPORT,
                timeframe=SignalTimeframe.DAILY,
                weight=0.5,
                normalization='minmax'
            ),
            'price': SignalMetadata(
                name='价格',
                description='资产收盘价',
                signal_type=SignalType.UNKNOWN,
                timeframe=SignalTimeframe.DAILY,
                weight=0.5,
                normalization='none'
            )
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        rsi_length = self.parameters['rsi_length']
        signal_length = self.parameters['signal_length']
        band_length = self.parameters['band_length']
        band_std = self.parameters['band_std']
        
        # 计算价格变化
        df['price_change'] = df['close'].diff()
        
        # 计算RSI
        df['tdi_gain'] = df['price_change'].clip(lower=0)
        df['tdi_loss'] = -df['price_change'].clip(upper=0)
        
        df['tdi_avg_gain'] = df['tdi_gain'].rolling(window=rsi_length).mean()
        df['tdi_avg_loss'] = df['tdi_loss'].rolling(window=rsi_length).mean()
        
        rs = df['tdi_avg_gain'] / df['tdi_avg_loss']
        df['tdi_rsi'] = 100 - (100 / (1 + rs))
        
        # 计算RSI的移动平均（信号线）
        df['tdi_signal'] = df['tdi_rsi'].rolling(window=signal_length).mean()
        
        # 计算RSI的波动带
        df['tdi_rsi_ma'] = df['tdi_rsi'].rolling(window=band_length).mean()
        df['tdi_rsi_std'] = df['tdi_rsi'].rolling(window=band_length).std()
        
        df['tdi_upper_band'] = df['tdi_rsi_ma'] + (df['tdi_rsi_std'] * band_std)
        df['tdi_lower_band'] = df['tdi_rsi_ma'] - (df['tdi_rsi_std'] * band_std)
        
        # 计算中心线（50值）
        df['tdi_midline'] = 50.0
        
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
        
        # 初始化信号列
        df['signal'] = 0
        
        # 计算TDI技术条件
        # 买入条件：RSI从下方穿过信号线且在50中线上方
        buy_condition = (df['tdi_rsi'] > df['tdi_signal']) & \
                        (df['tdi_rsi'].shift(1) <= df['tdi_signal'].shift(1)) & \
                        (df['tdi_rsi'] > df['tdi_midline'])
        
        # 卖出条件：RSI从上方穿过信号线且在50中线下方
        sell_condition = (df['tdi_rsi'] < df['tdi_signal']) & \
                         (df['tdi_rsi'].shift(1) >= df['tdi_signal'].shift(1)) & \
                         (df['tdi_rsi'] < df['tdi_midline'])
        
        # 生成信号
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # 填充NaN值
        df = df.fillna(0)
        df = df.infer_objects(copy=False)
        
        # 如果启用市场环境适配，调整信号
        if self.parameters['use_market_regime']:
            df = self.adjust_for_market_regime(df, df)
            
        return df
    
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
        
        # 根据TDI策略特性进行额外的调整
        if regime == MarketRegime.BEARISH:
            self.logger.info("TDI策略: 熊市环境，增强RSI过滤条件")
            # 在熊市中，加强卖出条件，对买入信号更保守
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 在熊市中，RSI要求需要更强(更高)才考虑买入
                    if adjusted_signals['tdi_rsi'].iloc[i] < 60:  # 提高RSI阈值
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        
        elif regime == MarketRegime.BULLISH:
            self.logger.info("TDI策略: 牛市环境，减少卖出信号")
            # 在牛市中，减少卖出信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] < 0:
                    # 在牛市中，RSI要求需要更低才考虑卖出
                    if adjusted_signals['tdi_rsi'].iloc[i] > 40:  # 降低RSI阈值
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        
        elif regime == MarketRegime.VOLATILE:
            self.logger.info("TDI策略: 高波动环境，调整为波动带突破策略")
            # 在高波动环境中，使用波动带突破策略
            for i in range(len(adjusted_signals)):
                # 只在RSI触及波动带时生成信号
                rsi = adjusted_signals['tdi_rsi'].iloc[i]
                upper_band = adjusted_signals['tdi_upper_band'].iloc[i]
                lower_band = adjusted_signals['tdi_lower_band'].iloc[i]
                
                if adjusted_signals['signal'].iloc[i] > 0 and rsi < upper_band * 0.95:
                    # 如果RSI未接近上波动带，忽略买入信号
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
                if adjusted_signals['signal'].iloc[i] < 0 and rsi > lower_band * 1.05:
                    # 如果RSI未接近下波动带，忽略卖出信号
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
        elif regime == MarketRegime.RANGING:
            self.logger.info("TDI策略: 震荡市场，增强均值回归操作")
            # 在震荡市场中，增强均值回归操作
            for i in range(len(adjusted_signals)):
                rsi = adjusted_signals['tdi_rsi'].iloc[i]
                
                # 在震荡市场中，当RSI极值时更容易产生均值回归信号
                if rsi > 70 and adjusted_signals['signal'].iloc[i] >= 0:
                    # RSI较高时，考虑卖出（反转当前信号）
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = -1
                elif rsi < 30 and adjusted_signals['signal'].iloc[i] <= 0:
                    # RSI较低时，考虑买入（反转当前信号）
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 1
        
        return adjusted_signals
        
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, SignalComponent]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        result = self.calculate_indicators(data)
        
        # 提取关键组件
        components = {
            'rsi': SignalComponent(
                series=result.get('tdi_rsi', pd.Series()),
                metadata=self.signal_metadata['rsi']
            ),
            'signal': SignalComponent(
                series=result.get('tdi_signal', pd.Series()),
                metadata=self.signal_metadata['signal']
            ),
            'upper_band': SignalComponent(
                series=result.get('tdi_upper_band', pd.Series()),
                metadata=self.signal_metadata['upper_band']
            ),
            'lower_band': SignalComponent(
                series=result.get('tdi_lower_band', pd.Series()),
                metadata=self.signal_metadata['lower_band']
            ),
            'midline': SignalComponent(
                series=result.get('tdi_midline', pd.Series()),
                metadata=self.signal_metadata['midline']
            ),
            'price': SignalComponent(
                series=result.get('close', pd.Series()),
                metadata=self.signal_metadata['price']
            )
        }
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {name: metadata.to_dict() for name, metadata in self.signal_metadata.items()} 