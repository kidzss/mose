import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
import talib

from .strategy_base import Strategy, MarketRegime
from .signal_interface import SignalComponent, SignalMetadata, SignalType, SignalTimeframe

class CustomTDIStrategy(Strategy):
    """
    TDI策略 - 趋势方向指标策略
    
    TDI (Trend Direction Indicator) 策略基于多个时间周期的趋势判断，
    结合RSI、MACD和成交量指标，生成买入和卖出信号。
    
    优化点：
    1. 多周期趋势判断
    2. 趋势强度确认
    3. 市场环境自适应
    """
    
    def __init__(self):
        # 初始化基类
        parameters = {
            'short_term_period': 10,
            'medium_term_period': 30,
            'long_term_period': 60,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'volume_period': 20,
            'volume_threshold': 1.5,
            'signal_threshold': 0.7,
            'min_holding_period': 5,
            'max_holding_period': 20,
            'min_trade_interval': 5,
            'profit_target': 0.15,
            'stop_loss': -0.10,
            'trailing_stop': 0.05
        }
        super().__init__(name="TDI", parameters=parameters)
        
        # 设置参数
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 创建DataFrame的副本
        df = df.copy()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df.loc[:, 'MACD'] = exp1 - exp2
        df.loc[:, 'Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        
        # 计算成交量指标
        df.loc[:, 'Volume_MA'] = df['volume'].rolling(window=self.volume_period).mean()
        df.loc[:, 'Volume_Ratio'] = df['volume'] / df['Volume_MA']
        
        # 计算多周期趋势
        df.loc[:, 'Trend_Short'] = df['close'].rolling(window=self.short_term_period).mean()
        df.loc[:, 'Trend_Medium'] = df['close'].rolling(window=self.medium_term_period).mean()
        df.loc[:, 'Trend_Long'] = df['close'].rolling(window=self.long_term_period).mean()
        
        # 计算波动率
        df.loc[:, 'Volatility'] = df['close'].pct_change().rolling(window=self.atr_period).std()
        
        # 计算动量指标
        df.loc[:, 'Momentum'] = df['close'].pct_change(periods=self.short_term_period)
        
        # 计算支撑和阻力
        df.loc[:, 'High_MA'] = df['high'].rolling(window=20).mean()
        df.loc[:, 'Low_MA'] = df['low'].rolling(window=20).mean()
        df.loc[:, 'Price_Position'] = (df['close'] - df['Low_MA']) / (df['High_MA'] - df['Low_MA'])
        
        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df.loc[:, 'ATR'] = true_range.rolling(window=self.atr_period).mean()
        
        # 新增指标计算
        # ADX指标
        df.loc[:, 'ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        
        # 动态波动率
        df.loc[:, 'Dynamic_Volatility'] = df['ATR'].rolling(window=20).mean()
        df.loc[:, 'Dynamic_Threshold'] = df['Dynamic_Volatility'] * 1.5
        
        # 趋势强度
        df.loc[:, 'Trend_Strength'] = df['ADX'] / 100.0  # 归一化到[0,1]
        
        # 动量强度
        df.loc[:, 'Momentum_Strength'] = (df['MACD'] - df['Signal']) / df['ATR']
        
        # 成交量强度
        df.loc[:, 'Volume_Strength'] = (df['Volume_Ratio'] - 1) / df['Dynamic_Threshold']
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        try:
            # 计算趋势得分
            trend_score = (
                (df['close'] > df['Trend_Short']) * 0.4 +
                (df['close'] > df['Trend_Medium']) * 0.3 +
                (df['close'] > df['Trend_Long']) * 0.3
            )
            
            # 计算动量得分
            momentum_score = (
                (df['RSI'] > 30) * 0.3 +
                (df['MACD'] > df['Signal']) * 0.4 +
                (df['close'] > df['close'].shift(1)) * 0.3
            )
            
            # 计算成交量得分
            volume_score = (
                (df['Volume_Ratio'] > self.volume_threshold) * 0.5 +
                (df['Volume_Ratio'] > df['Volume_Ratio'].shift(1)) * 0.5
            )
            
            # 计算ADX得分
            adx_score = (
                (df['ADX'] > self.adx_threshold) * 0.6 +
                (df['ADX'] > df['ADX'].shift(1)) * 0.4
            )
            
            # 计算ATR得分
            atr_score = (
                (df['ATR'] > df['ATR'].rolling(window=self.atr_period).mean() * self.atr_multiplier) * 0.5 +
                (df['ATR'] > df['ATR'].shift(1)) * 0.5
            )
            
            # 计算综合得分
            total_score = (
                trend_score * 0.3 +
                momentum_score * 0.25 +
                volume_score * 0.15 +
                adx_score * 0.15 +
                atr_score * 0.15
            )
            
            # 生成交易信号
            signals = pd.Series(0, index=df.index)
            
            # 买入信号
            buy_condition = (
                (total_score > self.signal_threshold) &  # 降低信号阈值要求
                (df['close'] > df['Trend_Short']) &  # 价格在短期均线上方
                (df['Volume_Ratio'] > df['Volume_MA']) &  # 成交量放大
                (df['ADX'] > self.adx_threshold)  # 趋势强度足够
            )
            
            # 卖出信号
            sell_condition = (
                (total_score < -self.signal_threshold) |  # 降低信号阈值要求
                (df['close'] < df['Trend_Short']) |  # 价格跌破短期均线
                (df['RSI'] > 70) |  # RSI超买
                (df['MACD'] < df['Signal'])  # MACD死叉
            )
            
            # 设置信号
            signals[buy_condition] = 1
            signals[sell_condition] = -1
            
            # 添加信号平滑
            signals = signals.rolling(window=3).mean()
            signals = signals.round()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            raise
        
    def adjust_signals(self, signals: Dict[str, pd.DataFrame], market_regime: MarketRegime) -> Dict[str, pd.DataFrame]:
        """根据市场环境调整信号"""
        if not self.market_adaptation:
            return signals
            
        # 获取趋势信号
        trend_signals = signals['trend']
        momentum_signals = signals['momentum']
        volume_signals = signals['volume']
        market_signals = signals['market']
        
        # 根据市场环境调整信号
        if market_regime == MarketRegime.BULLISH:
            # 牛市环境，增强买入信号
            trend_signals['trend_score'] *= 1.1
            momentum_signals['rsi_signal'] = np.where(
                momentum_signals['rsi_signal'] == 1, 1.2,
                momentum_signals['rsi_signal']
            )
        elif market_regime == MarketRegime.BEARISH:
            # 熊市环境，增强卖出信号
            trend_signals['trend_score'] *= 0.9
            momentum_signals['rsi_signal'] = np.where(
                momentum_signals['rsi_signal'] == -1, -1.2,
                momentum_signals['rsi_signal']
            )
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动环境，降低信号强度
            trend_signals['trend_score'] *= 0.8
            momentum_signals['rsi_signal'] *= 0.8
            volume_signals['volume_signal'] *= 0.8
            
        # 更新信号
        signals['trend'] = trend_signals
        signals['momentum'] = momentum_signals
        signals['volume'] = volume_signals
        signals['market'] = market_signals
        
        return signals
        
    def generate_final_signal(self, signals: Dict[str, pd.DataFrame]) -> pd.Series:
        """生成最终交易信号"""
        # 获取各个信号
        trend_signals = signals['trend']
        momentum_signals = signals['momentum']
        volume_signals = signals['volume']
        market_signals = signals['market']
        
        # 合并信号
        final_signal = pd.Series(0, index=trend_signals.index)
        
        # 合并信号
        final_signal += trend_signals['trend_score'] * self.weights['trend']
        final_signal += momentum_signals['rsi_signal'] * self.weights['momentum']
        final_signal += momentum_signals['macd_signal'] * self.weights['momentum']
        final_signal += volume_signals['volume_signal'] * self.weights['volume']
        final_signal += market_signals['volatility_signal'] * self.weights['market']
        
        # 生成交易信号
        final_signal = np.where(final_signal > 0.5, 1,
                              np.where(final_signal < -0.5, -1, 0))
        
        return pd.Series(final_signal, index=trend_signals.index)
        
    def get_signal_metadata(self) -> SignalMetadata:
        """获取信号元数据"""
        return SignalMetadata(
            strategy_name="TDI",
            signal_type=SignalType.TREND,
            timeframe=SignalTimeframe.DAILY,
            description="基于多周期趋势和动量的交易信号"
        )
        
    def extract_signal_components(self, df: pd.DataFrame) -> Dict[str, SignalComponent]:
        """提取信号组件"""
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        # 趋势信号
        trend_signals = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            # 计算趋势得分
            trend_score = 0.0
            # 价格相对于趋势线的位置
            if df['close'].iloc[i] > df['Trend_Short'].iloc[i]:
                trend_score += 1.0
            if df['close'].iloc[i] > df['Trend_Medium'].iloc[i]:
                trend_score += 1.0
            if df['close'].iloc[i] > df['Trend_Long'].iloc[i]:
                trend_score += 1.0
            # 趋势线的相对位置
            if (df['Trend_Short'].iloc[i] > df['Trend_Medium'].iloc[i] and 
                df['Trend_Medium'].iloc[i] > df['Trend_Long'].iloc[i]):
                trend_score += 1.0
            elif (df['Trend_Short'].iloc[i] < df['Trend_Medium'].iloc[i] and 
                  df['Trend_Medium'].iloc[i] < df['Trend_Long'].iloc[i]):
                trend_score -= 1.0
            # 趋势强度
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                trend_score += 0.5
            else:
                trend_score -= 0.5
            trend_signals.iloc[i] = trend_score / 4.5  # 标准化到[-1, 1]
            
        # 动量信号
        momentum_signals = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            # 计算动量得分
            momentum_score = 0.0
            # RSI超买超卖
            if df['RSI'].iloc[i] < 30:
                momentum_score += 1.0
            elif df['RSI'].iloc[i] > 70:
                momentum_score -= 1.0
            # MACD金叉死叉
            if df['MACD'].iloc[i] > df['Signal'].iloc[i]:
                momentum_score += 1.0
            else:
                momentum_score -= 1.0
            # 价格动量
            if df['Momentum'].iloc[i] > 0:
                momentum_score += 0.5
            else:
                momentum_score -= 0.5
            momentum_signals.iloc[i] = momentum_score / 2.5  # 标准化到[-1, 1]
            
        # 成交量信号
        volume_signals = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            # 成交量突破
            if df['Volume_Ratio'].iloc[i] > self.volume_threshold:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:  # 上涨放量
                    volume_signals.iloc[i] = 1.0
                else:  # 下跌放量
                    volume_signals.iloc[i] = -1.0
                    
        # 市场环境信号
        market_signals = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if self.market_adaptation:
                # 波动率判断
                volatility = df['Volatility'].iloc[i]
                volatility_ma = df['Volatility'].rolling(window=self.volatility_period).mean().iloc[i]
                if volatility < volatility_ma:  # 低波动率
                    market_signals.iloc[i] = 1.0
                else:
                    market_signals.iloc[i] = -1.0
                    
        # 更新信号组件
        self.trend_signal.series = trend_signals
        self.momentum_signal.series = momentum_signals
        self.volume_signal.series = volume_signals
        self.market_signal.series = market_signals
        
        return {
            'trend': self.trend_signal,
            'momentum': self.momentum_signal,
            'volume': self.volume_signal,
            'market': self.market_signal
        } 

    def risk_control(self, position: int, price: float, stop_loss: float) -> bool:
        """风险控制"""
        # 1. 动态止损
        atr = self.calculate_atr()
        dynamic_stop = price - (atr * self.atr_multiplier)
        
        # 2. 波动率调整
        if self.volatility_adjustment:
            volatility = self.calculate_volatility()
            self.position_size = self.base_position_size * (1 - volatility)
        
        # 3. 连续亏损控制
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.logger.warning(f"连续亏损{self.consecutive_losses}次，暂停交易")
            return False
        
        # 4. 回撤控制
        if self.current_drawdown > self.max_drawdown_threshold:
            self.logger.warning(f"当前回撤{self.current_drawdown:.2%}超过阈值{self.max_drawdown_threshold:.2%}")
            return False
        
        # 5. 趋势强度确认
        if position != 0:
            adx = self.calculate_adx()
            if adx < self.adx_threshold:
                self.logger.warning(f"趋势强度不足，ADX={adx:.2f}")
                return False
        
        # 6. 波动率过滤
        if self.calculate_volatility() > self.volatility_threshold:
            self.logger.warning("波动率过高，暂停交易")
            return False
        
        return True

    def calculate_atr(self) -> float:
        """计算ATR"""
        high = self.data['high'].iloc[-1]
        low = self.data['low'].iloc[-1]
        close = self.data['close'].iloc[-1]
        
        tr1 = high - low
        tr2 = abs(high - self.data['close'].iloc[-2])
        tr3 = abs(low - self.data['close'].iloc[-2])
        
        tr = max(tr1, tr2, tr3)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr.iloc[-1]

    def calculate_volatility(self) -> float:
        """计算波动率"""
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_period).std()
        return volatility.iloc[-1]

    def calculate_adx(self) -> float:
        """计算ADX"""
        return self.data['ADX'].iloc[-1]

    def update_trade_status(self, profit: float):
        """更新交易状态"""
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # 更新回撤
        if profit > 0:
            self.peak_value = max(self.peak_value, self.current_value)
        self.current_drawdown = (self.peak_value - self.current_value) / self.peak_value 

    def backtest(self, data: pd.DataFrame) -> np.ndarray:
        """回测策略并返回收益率序列"""
        try:
            # 计算技术指标
            df = self.calculate_indicators(data.copy())
            
            # 生成交易信号
            signals = self.generate_signals(df)
            
            # 初始化变量
            position = 0
            returns = []
            entry_price = 0
            holding_period = 0
            last_trade_index = 0
            
            # 遍历数据
            for i in range(1, len(df)):
                current_price = df['close'].iloc[i]
                signal = signals.iloc[i]
                
                # 计算当日收益率
                daily_return = 0.0
                
                if position != 0:
                    # 计算价格变化百分比
                    price_change = (current_price / df['close'].iloc[i-1]) - 1
                    daily_return = position * price_change
                    holding_period += 1
                    
                    # 检查止盈止损条件
                    total_return = (current_price / entry_price - 1) * position
                    
                    # 止盈
                    if total_return >= self.profit_target:
                        daily_return = position * (current_price / df['close'].iloc[i-1] - 1)
                        position = 0
                        holding_period = 0
                        last_trade_index = i
                    # 止损
                    elif total_return <= self.stop_loss:
                        daily_return = position * (current_price / df['close'].iloc[i-1] - 1)
                        position = 0
                        holding_period = 0
                        last_trade_index = i
                    # 移动止损
                    elif position > 0 and current_price < entry_price * (1 + self.trailing_stop):
                        daily_return = position * (current_price / df['close'].iloc[i-1] - 1)
                        position = 0
                        holding_period = 0
                        last_trade_index = i
                    elif position < 0 and current_price > entry_price * (1 - self.trailing_stop):
                        daily_return = position * (current_price / df['close'].iloc[i-1] - 1)
                        position = 0
                        holding_period = 0
                        last_trade_index = i
                    
                    # 检查最大持仓时间
                    if holding_period >= self.max_holding_period:
                        daily_return = position * (current_price / df['close'].iloc[i-1] - 1)
                        position = 0
                        holding_period = 0
                        last_trade_index = i
                
                # 生成新的交易信号
                if position == 0 and signal != 0 and (i - last_trade_index) >= self.min_trade_interval:
                    if signal > 0:
                        position = 1
                        entry_price = current_price
                        holding_period = 0
                    elif signal < 0:
                        position = -1
                        entry_price = current_price
                        holding_period = 0
                
                returns.append(daily_return)
            
            return np.array(returns)
            
        except Exception as e:
            self.logger.error(f"回测过程中出错: {str(e)}")
            raise 