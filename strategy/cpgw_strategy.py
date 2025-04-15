from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging

from strategy.strategy_base import Strategy, MarketRegime

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
        
        # Generate signals based on RSI, MA crossover, and momentum
        for i in range(1, len(df)):
            # Buy signal conditions
            buy_condition1 = df['rsi'].iloc[i-1] < 30  # RSI < 30
            buy_condition2 = df['ma_cross'].iloc[i] == 1
            buy_condition3 = df['momentum'].iloc[i] > 0
            
            # Sell signal conditions
            sell_condition1 = df['rsi'].iloc[i-1] > 70  # RSI > 70
            sell_condition2 = df['ma_cross'].iloc[i] == -1
            sell_condition3 = df['momentum'].iloc[i] < 0
            
            # Generate signals with original conditions
            if (buy_condition1 and buy_condition2) or (buy_condition2 and buy_condition3):
                df.loc[df.index[i], 'signal'] = 1
                print(f"\n生成买入信号: {df.index[i]}")
                print(f"RSI: {df['rsi'].iloc[i-1]:.2f}")
                print(f"MA Cross: {df['ma_cross'].iloc[i]}")
                print(f"Momentum: {df['momentum'].iloc[i]:.2%}")
            elif (sell_condition1 and sell_condition2) or (sell_condition2 and sell_condition3):
                df.loc[df.index[i], 'signal'] = -1
                print(f"\n生成卖出信号: {df.index[i]}")
                print(f"RSI: {df['rsi'].iloc[i-1]:.2f}")
                print(f"MA Cross: {df['ma_cross'].iloc[i]}")
                print(f"Momentum: {df['momentum'].iloc[i]:.2%}")
        
        # Adjust signals based on market regime if enabled
        if self.use_market_regime:
            df = self.adjust_for_market_regime(df)
        
        return df
    
    def adjust_for_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust signals based on current market regime"""
        market_regime = self.get_market_regime(df)
        
        # Adjust signals based on market regime
        if market_regime == MarketRegime.BEARISH:
            # In bearish markets, enhance sell signals and reduce buy signals
            df['signal'] = np.where(df['signal'] == 1, 0.5, df['signal'])  # Reduce buy signal strength
            df['signal'] = np.where(df['signal'] == -1, -1.5, df['signal'])  # Enhance sell signal strength
            
        elif market_regime == MarketRegime.BULLISH:
            # In bullish markets, enhance buy signals and reduce sell signals
            df['signal'] = np.where(df['signal'] == 1, 1.5, df['signal'])  # Enhance buy signal strength
            df['signal'] = np.where(df['signal'] == -1, -0.5, df['signal'])  # Reduce sell signal strength
            
        elif market_regime == MarketRegime.VOLATILE:
            # In volatile markets, reduce all signals
            df['signal'] = df['signal'] * 0.7
            
        elif market_regime == MarketRegime.RANGING:
            # In ranging markets, use normal signals but slightly reduce strength
            df['signal'] = df['signal'] * 0.9
            
        return df
    
    def get_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Determine the current market regime"""
        # Calculate volatility using ATR (Approximate)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Get recent trend direction using shorter MA
        ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        ma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        
        # Current volatility
        current_volatility = atr.iloc[-1] / df['close'].iloc[-1]
        
        # Determine market regime
        if current_volatility > 0.02:  # Lower volatility threshold
            return MarketRegime.VOLATILE
        elif ma_20 > ma_50 * 1.02:  # Less strict uptrend condition
            return MarketRegime.BULLISH
        elif ma_20 < ma_50 * 0.98:  # Less strict downtrend condition
            return MarketRegime.BEARISH
        else:  # Ranging market
            return MarketRegime.RANGING
    
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
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about the strategy's signals"""
        return {
            "rsi": {
                "name": "RSI",
                "description": f"Relative Strength Index ({self.lookback_period} periods)",
                "weight": 0.6,
                "normalization": "minmax",
                "interpretation": "Low RSI indicates oversold conditions (buy), high RSI indicates overbought (sell)"
            },
            "ma_cross": {
                "name": "MA Crossover",
                "description": f"Moving Average Crossover ({self.fast_ma}/{self.slow_ma})",
                "weight": 0.4,
                "normalization": "none",
                "interpretation": "Positive values indicate bullish crossover, negative values indicate bearish crossover"
            }
        } 