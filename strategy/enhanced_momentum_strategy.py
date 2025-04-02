import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple

from .strategy_base import Strategy, MarketRegime
from .signal_interface import (
    SignalType, SignalTimeframe, SignalStrength, 
    SignalMetadata, SignalComponent, SignalCombiner
)


class EnhancedMomentumStrategy(Strategy):
    """
    增强版动量策略
    
    扩展了基础动量策略，添加了以下功能:
    1. 多时间周期分析
    2. 多指标融合 (动量、RSI、MACD)
    3. 市场环境自适应
    4. 标准化信号输出
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化策略
        
        参数:
            parameters: 策略参数字典
        """
        # 设置默认参数
        default_params = {
            # 动量参数
            'momentum_length': 10,       # 动量计算周期
            'momentum_ma_length': 5,     # 动量移动平均周期
            'momentum_weight': 1.0,      # 动量信号权重
            
            # RSI参数
            'rsi_length': 14,            # RSI计算周期
            'rsi_overbought': 70,        # RSI超买阈值
            'rsi_oversold': 30,          # RSI超卖阈值
            'rsi_weight': 0.7,           # RSI信号权重
            
            # MACD参数
            'macd_fast': 12,             # MACD快线周期
            'macd_slow': 26,             # MACD慢线周期
            'macd_signal': 9,            # MACD信号线周期
            'macd_weight': 0.5,          # MACD信号权重
            
            # 信号阈值
            'signal_threshold': 0.3,     # 信号阈值
            
            # 其他参数
            'use_market_regime': True,   # 是否使用市场环境
            'vola_lookback': 20,         # 波动率计算周期
            'trend_lookback': 50,        # 趋势计算周期
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
        
        # 初始化基类
        super().__init__('EnhancedMomentumStrategy', default_params)
        
        # 初始化信号元数据
        self._initialize_signal_metadata()
    
    def _initialize_signal_metadata(self) -> None:
        """初始化信号元数据"""
        self.signal_metadata = {
            'momentum': SignalMetadata(
                name="动量",
                description=f"价格与{self.parameters['momentum_length']}周期前价格的差值",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=self.parameters['momentum_weight'],
                normalization='minmax'
            ),
            'momentum_change': SignalMetadata(
                name="动量变化",
                description="动量的一阶导数，衡量动量变化速度",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=self.parameters['momentum_weight'] * 0.8,
                normalization='minmax'
            ),
            'rsi': SignalMetadata(
                name="RSI",
                description=f"{self.parameters['rsi_length']}周期RSI指标",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=self.parameters['rsi_weight'],
                normalization='minmax',
                normalization_params={'min': 0, 'max': 100}
            ),
            'macd_hist': SignalMetadata(
                name="MACD柱状图",
                description="MACD柱状图，衡量价格动量",
                signal_type=SignalType.MOMENTUM,
                timeframe=SignalTimeframe.DAILY,
                weight=self.parameters['macd_weight'],
                normalization='zscore'
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
        
        # 获取参数
        momentum_length = self.parameters['momentum_length']
        momentum_ma_length = self.parameters['momentum_ma_length']
        rsi_length = self.parameters['rsi_length']
        macd_fast = self.parameters['macd_fast']
        macd_slow = self.parameters['macd_slow']
        macd_signal = self.parameters['macd_signal']
        
        # 1. 计算动量指标
        # 当前价格与n周期前价格的差
        df['momentum'] = df['close'] - df['close'].shift(momentum_length)
        # 动量的移动平均
        df['momentum_ma'] = df['momentum'].rolling(window=momentum_ma_length).mean()
        # 动量变化
        df['momentum_change'] = df['momentum'] - df['momentum'].shift(1)
        
        # 2. 计算RSI
        # 获取价格变化
        delta = df['close'].diff()
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # 计算平均涨跌幅
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
        # 计算相对强弱值
        rs = avg_gain / avg_loss
        # 计算RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. 计算MACD
        # 快速EMA
        df['macd_fast_ema'] = df['close'].ewm(span=macd_fast, adjust=False).mean()
        # 慢速EMA
        df['macd_slow_ema'] = df['close'].ewm(span=macd_slow, adjust=False).mean()
        # MACD线 = 快线 - 慢线
        df['macd'] = df['macd_fast_ema'] - df['macd_slow_ema']
        # 信号线 = MACD的EMA
        df['macd_signal_line'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
        # 柱状图 = MACD线 - 信号线
        df['macd_hist'] = df['macd'] - df['macd_signal_line']
        
        # 4. 计算波动率指标 (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.DataFrame(data={
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        df['atr'] = tr.rolling(window=self.parameters['vola_lookback']).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100  # 百分比ATR
        
        # 5. 计算趋势指标 - 价格相对于长期均线的位置
        df['sma50'] = df['close'].rolling(window=self.parameters['trend_lookback']).mean()
        df['price_vs_sma'] = df['close'] / df['sma50'] - 1  # 价格与均线的偏差百分比
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame，其中:
            1 = 买入信号
            0 = 持有/无信号
            -1 = 卖出信号
        """
        # 计算技术指标
        df = self.calculate_indicators(data.copy())
        
        # 获取信号组件
        signal_components = self.extract_signal_components(df)
        
        # 创建信号组合器
        combiner = SignalCombiner(signal_components)
        
        # 获取组合信号序列
        combined_signal = combiner.combine(method='weighted_average')
        
        # 添加组合信号到DataFrame
        df['combined_signal'] = combined_signal
        
        # 应用信号阈值生成离散信号
        threshold = self.parameters['signal_threshold']
        df['signal'] = 0  # 默认无信号
        df.loc[df['combined_signal'] > threshold, 'signal'] = 1  # 买入信号
        df.loc[df['combined_signal'] < -threshold, 'signal'] = -1  # 卖出信号
        
        # 如果启用市场环境适配，调整信号
        if self.parameters['use_market_regime']:
            df = self.adjust_for_market_regime(df, df)
        
        return df
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, SignalComponent]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        # 提取指标数据
        components = {}
        
        # 1. 动量指标
        if 'momentum' in data.columns:
            momentum_series = data['momentum']
            components['momentum'] = SignalComponent(
                series=momentum_series,
                metadata=self.signal_metadata['momentum']
            )
        
        # 2. 动量变化
        if 'momentum_change' in data.columns:
            momentum_change_series = data['momentum_change']
            components['momentum_change'] = SignalComponent(
                series=momentum_change_series,
                metadata=self.signal_metadata['momentum_change']
            )
        
        # 3. RSI指标
        if 'rsi' in data.columns:
            # 将RSI从[0,100]映射到[-1,1]
            rsi_series = data['rsi']
            # 创建RSI组件
            components['rsi'] = SignalComponent(
                series=rsi_series,
                metadata=self.signal_metadata['rsi']
            )
        
        # 4. MACD柱状图
        if 'macd_hist' in data.columns:
            macd_hist_series = data['macd_hist']
            components['macd_hist'] = SignalComponent(
                series=macd_hist_series,
                metadata=self.signal_metadata['macd_hist']
            )
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {name: metadata.to_dict() for name, metadata in self.signal_metadata.items()}
        
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        判断当前市场环境
        
        参数:
            data: 市场数据
            
        返回:
            市场环境枚举值
        """
        # 使用基类的市场环境检测
        regime = super().get_market_regime(data)
        
        # 增强版市场环境判断
        try:
            if len(data) < 50:
                return regime
                
            # 获取最新数据
            df = data.copy()
            
            # 使用RSI增强判断
            if 'rsi' in df.columns:
                latest_rsi = df['rsi'].iloc[-1]
                
                # RSI超过70可能表示趋势很强或即将反转
                if latest_rsi > 80 and regime == MarketRegime.BULLISH:
                    return MarketRegime.VOLATILE  # 可能即将反转
                    
                # RSI低于30可能表示趋势很弱或即将反转
                if latest_rsi < 20 and regime == MarketRegime.BEARISH:
                    return MarketRegime.VOLATILE  # 可能即将反转
            
            # 使用MACD增强判断
            if 'macd' in df.columns and 'macd_signal_line' in df.columns:
                # MACD多头排列（MACD线在信号线上方且MACD为正）
                if (df['macd'].iloc[-1] > df['macd_signal_line'].iloc[-1] and 
                    df['macd'].iloc[-1] > 0 and regime == MarketRegime.UNKNOWN):
                    return MarketRegime.BULLISH
                
                # MACD空头排列（MACD线在信号线下方且MACD为负）
                if (df['macd'].iloc[-1] < df['macd_signal_line'].iloc[-1] and 
                    df['macd'].iloc[-1] < 0 and regime == MarketRegime.UNKNOWN):
                    return MarketRegime.BEARISH
        
        except Exception as e:
            self.logger.error(f"增强市场环境判断出错: {e}")
            
        return regime
        
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据市场环境调整信号
        
        参数:
            data: 原始数据
            signals: 信号数据
            
        返回:
            调整后的信号数据
        """
        # 调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 使用RSI与市场环境结合进行额外调整
        if 'rsi' in data.columns:
            rsi_values = data['rsi']
            
            # 在牛市中，避免在RSI高位开多
            if regime == MarketRegime.BULLISH:
                for i in range(len(adjusted_signals)):
                    if (adjusted_signals['signal'].iloc[i] > 0 and 
                        rsi_values.iloc[i] > self.parameters['rsi_overbought'] + 5):
                        # RSI过高，降低买入信号或变为中性
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        self.logger.info(f"牛市高RSI({rsi_values.iloc[i]})，忽略买入信号")
            
            # 在熊市中，避免在RSI低位开空
            elif regime == MarketRegime.BEARISH:
                for i in range(len(adjusted_signals)):
                    if (adjusted_signals['signal'].iloc[i] < 0 and 
                        rsi_values.iloc[i] < self.parameters['rsi_oversold'] - 5):
                        # RSI过低，降低卖出信号或变为中性
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        self.logger.info(f"熊市低RSI({rsi_values.iloc[i]})，忽略卖出信号")
        
        return adjusted_signals
    
    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """
        计算仓位大小
        
        参数:
            data: 市场数据
            signal: 信号值(-1.0至1.0)
            
        返回:
            仓位大小(0.0-1.0)
        """
        # 调用基类的仓位计算
        base_position = super().get_position_size(data, signal)
        
        # 根据信号强度进一步调整仓位
        signal_strength = abs(signal)
        
        # 获取市场波动率
        if 'atr_pct' in data.columns:
            latest_atr_pct = data['atr_pct'].iloc[-1]
            
            # 在高波动率环境下减少仓位
            if latest_atr_pct > 3.0:  # 3%以上的ATR被视为高波动率
                position_multiplier = max(0.5, 1.0 - (latest_atr_pct - 3.0) / 10.0)
                self.logger.info(f"高波动率环境(ATR={latest_atr_pct:.2f}%)，仓位乘数={position_multiplier:.2f}")
                return base_position * position_multiplier
        
        return base_position
    
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格
        
        参数:
            data: 市场数据
            entry_price: 入场价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            止损价格
        """
        # 调用基类的止损计算方法
        base_stop = super().get_stop_loss(data, entry_price, position)
        
        # 如果有ATR数据，使用ATR设置止损
        if 'atr' in data.columns:
            latest_atr = data['atr'].iloc[-1]
            
            # 使用2倍ATR作为止损距离
            atr_multiplier = 2.0
            
            if position > 0:  # 多头止损
                atr_stop = entry_price - (latest_atr * atr_multiplier)
                # 使用较紧的止损（ATR止损和基本止损的最大值）
                return max(atr_stop, base_stop)
            elif position < 0:  # 空头止损
                atr_stop = entry_price + (latest_atr * atr_multiplier)
                # 使用较紧的止损（ATR止损和基本止损的最小值）
                return min(atr_stop, base_stop)
        
        return base_stop
    
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float,
                                stop_loss: float, position: int) -> float:
        """
        是否应该调整止损价格(追踪止损)
        
        参数:
            data: 市场数据
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            新的止损价格，如果不需要调整则返回原止损价格
        """
        # 获取ATR数据
        if 'atr' in data.columns:
            latest_atr = data['atr'].iloc[-1]
            
            # 计算盈利点数
            profit_points = 0
            if position > 0:  # 多头
                profit_points = current_price - stop_loss
            elif position < 0:  # 空头
                profit_points = stop_loss - current_price
            
            # 如果盈利大于3倍ATR，开始移动止损
            if profit_points > latest_atr * 3:
                if position > 0:  # 多头追踪止损
                    new_stop = current_price - latest_atr * 2
                    # 只有新止损比旧止损高时才更新
                    if new_stop > stop_loss:
                        return new_stop
                elif position < 0:  # 空头追踪止损
                    new_stop = current_price + latest_atr * 2
                    # 只有新止损比旧止损低时才更新
                    if new_stop < stop_loss:
                        return new_stop
        
        return stop_loss 