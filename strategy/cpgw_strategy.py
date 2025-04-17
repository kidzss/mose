from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging

from .strategy_base import Strategy, MarketRegime

logger = logging.getLogger(__name__)

class CPGWStrategy(Strategy):
    """
    CPGW Strategy - based on RSI and MA crossover
    """
    
    def __init__(self, lookback_period: int = 14, overbought: int = 70, 
                 oversold: int = 30, fast_ma: int = 5, slow_ma: int = 20,
                 use_market_regime: bool = True):
        """
        Initialize CPGW Strategy
        
        Args:
            lookback_period: Period for RSI calculation
            overbought: RSI overbought threshold
            oversold: RSI oversold threshold
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            use_market_regime: Whether to adjust signals based on market regime
        """
        # Create a parameters dictionary
        parameters = {
            'lookback_period': lookback_period,
            'overbought': overbought,
            'oversold': oversold,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'use_market_regime': use_market_regime
        }
        
        # Initialize the base class
        super().__init__('CPGWStrategy', parameters)
        
        # Store parameters as instance variables for easy access
        self.lookback_period = lookback_period
        self.overbought = overbought
        self.oversold = oversold
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.use_market_regime = use_market_regime
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators used by the strategy"""
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.lookback_period).mean()
        avg_loss = loss.rolling(window=self.lookback_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Calculate MA crossover
        df['ma_cross'] = np.where(df['fast_ma'] > df['slow_ma'], 1, -1)
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(periods=5)
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on CPGW strategy"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0
        
        # 获取市场环境
        market_regime = self.get_market_regime(df)
        volatility_regime = self.get_volatility_regime(df)
        
        # Generate signals based on RSI, MA crossover, and momentum
        for i in range(1, len(df)):
            # 趋势判断
            trend_up = df['close'].iloc[i-1] > df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1]
            trend_down = df['close'].iloc[i-1] < df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1]
            
            # Buy signal conditions
            buy_condition1 = df['rsi'].iloc[i-1] < self.oversold  # RSI oversold
            buy_condition2 = df['ma_cross'].iloc[i-1] == 1  # MA crossover bullish
            buy_condition3 = df['momentum'].iloc[i-1] > 0  # Positive momentum
            
            # Sell signal conditions
            sell_condition1 = df['rsi'].iloc[i-1] > self.overbought  # RSI overbought
            sell_condition2 = df['ma_cross'].iloc[i-1] == -1  # MA crossover bearish
            sell_condition3 = df['momentum'].iloc[i-1] < 0  # Negative momentum
            
            # 根据市场环境调整信号强度
            signal_strength = 1.0
            if market_regime == MarketRegime.BULLISH:
                signal_strength *= 1.5
            elif market_regime == MarketRegime.BEARISH:
                signal_strength *= 0.5
                
            if volatility_regime.iloc[i] == 'high':
                signal_strength *= 0.7
            elif volatility_regime.iloc[i] == 'low':
                signal_strength *= 1.3
            
            # Generate signals
            if trend_up and ((buy_condition1 and buy_condition2) or (buy_condition1 and buy_condition3) or (buy_condition2 and buy_condition3)):
                df.loc[df.index[i], 'signal'] = signal_strength  # Buy signal
            elif trend_down and ((sell_condition1 and sell_condition2) or (sell_condition1 and sell_condition3) or (sell_condition2 and sell_condition3)):
                df.loc[df.index[i], 'signal'] = -signal_strength  # Sell signal
        
        return df
    
    def get_position_size(self, df: pd.DataFrame, current_price: float) -> float:
        """根据市场环境和信号强度动态调整仓位大小"""
        # 获取当前市场环境
        market_regime = self.get_market_regime(df)
        volatility_regime = self.get_volatility_regime(df).iloc[-1]
        
        # 基础仓位大小
        base_position = 0.1
        
        # 根据市场环境调整仓位
        if market_regime == MarketRegime.BULLISH:
            base_position *= 1.5
        elif market_regime == MarketRegime.BEARISH:
            base_position *= 0.5
            
        # 根据波动率调整仓位
        if volatility_regime == 'high':
            base_position *= 0.7
        elif volatility_regime == 'low':
            base_position *= 1.3
            
        return base_position
    
    def get_stop_loss(self, df: pd.DataFrame, entry_price: float, position: str) -> float:
        """根据市场环境动态调整止损价格"""
        # 获取当前市场环境
        market_regime = self.get_market_regime(df)
        volatility_regime = self.get_volatility_regime(df).iloc[-1]
        
        # 基础止损比例
        base_stop_loss = 0.03
        
        # 根据市场环境调整止损
        if market_regime == MarketRegime.BULLISH:
            base_stop_loss *= 1.2
        elif market_regime == MarketRegime.BEARISH:
            base_stop_loss *= 0.8
            
        # 根据波动率调整止损
        if volatility_regime == 'high':
            base_stop_loss *= 1.2
        elif volatility_regime == 'low':
            base_stop_loss *= 0.8
            
        # 计算止损价格
        if position == 'long':
            stop_loss_price = entry_price * (1 - base_stop_loss)
        else:
            stop_loss_price = entry_price * (1 + base_stop_loss)
            
        return stop_loss_price
    
    def get_take_profit(self, df: pd.DataFrame, entry_price: float, position: str) -> float:
        """根据市场环境动态调整止盈价格"""
        # 获取当前市场环境
        market_regime = self.get_market_regime(df)
        volatility_regime = self.get_volatility_regime(df).iloc[-1]
        
        # 基础止盈比例
        base_take_profit = 0.1
        
        # 根据市场环境调整止盈
        if market_regime == MarketRegime.BULLISH:
            base_take_profit *= 1.2
        elif market_regime == MarketRegime.BEARISH:
            base_take_profit *= 0.8
            
        # 根据波动率调整止盈
        if volatility_regime == 'high':
            base_take_profit *= 1.2
        elif volatility_regime == 'low':
            base_take_profit *= 0.8
            
        # 计算止盈价格
        if position == 'long':
            take_profit_price = entry_price * (1 + base_take_profit)
        else:
            take_profit_price = entry_price * (1 - base_take_profit)
            
        return take_profit_price
    
    def extract_signal_components(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract individual components that make up the signal"""
        # Calculate indicators if they haven't been calculated yet
        if 'rsi' not in df.columns:
            df = self.calculate_indicators(df)
            
        components = {}
        
        # RSI component
        components['rsi'] = df['rsi']
        
        # MA Crossover component
        components['ma_cross'] = df['ma_cross']
        
        # Momentum component
        components['momentum'] = df['momentum']
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about the strategy's signals"""
        return {
            "rsi": {
                "name": "RSI",
                "description": f"Relative Strength Index ({self.lookback_period} periods)",
                "weight": 0.4,
                "normalization": "minmax",
                "interpretation": "Low RSI indicates oversold conditions (buy), high RSI indicates overbought (sell)"
            },
            "ma_cross": {
                "name": "MA Crossover",
                "description": f"Moving Average Crossover ({self.fast_ma}/{self.slow_ma})",
                "weight": 0.4,
                "normalization": "none",
                "interpretation": "Positive values indicate bullish crossover, negative values indicate bearish crossover"
            },
            "momentum": {
                "name": "Price Momentum",
                "description": "5-period price momentum",
                "weight": 0.2,
                "normalization": "percentile",
                "interpretation": "Positive values indicate upward momentum, negative values indicate downward momentum"
            }
        }
    
    def update_trade_stats(self, trade_result):
        """Update strategy's trade statistics"""
        # Implement if needed
        pass 
    
    def get_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        判断波动率环境
        
        Args:
            df: 市场数据
            
        Returns:
            pd.Series: 波动率环境 ('high', 'low', 'normal')
        """
        # 计算波动率
        volatility = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)  # 年化波动率
        
        # 计算波动率分位数
        high_threshold = volatility.quantile(0.7)
        low_threshold = volatility.quantile(0.3)
        
        # 判断波动率环境
        volatility_regime = pd.Series(index=df.index, data='normal')
        volatility_regime[volatility > high_threshold] = 'high'
        volatility_regime[volatility < low_threshold] = 'low'
        
        return volatility_regime 