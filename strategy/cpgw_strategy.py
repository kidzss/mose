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
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.lookback_period).mean()
        avg_loss = loss.rolling(window=self.lookback_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['fast_ma'] = df['Close'].rolling(window=self.fast_ma).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_ma).mean()
        
        # Calculate MA crossover
        df['ma_cross'] = np.where(df['fast_ma'] > df['slow_ma'], 1, -1)
        
        # Calculate price momentum
        df['momentum'] = df['Close'].pct_change(periods=5)
        
        # Calculate volatility
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on CPGW strategy"""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate base signals
        for i in range(1, len(df)):
            # 趋势判断
            trend_up = df['Close'].iloc[i-1] > df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1]
            trend_down = df['Close'].iloc[i-1] < df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1]
            
            # Buy signal conditions
            buy_condition1 = df['rsi'].iloc[i-1] < self.oversold  # RSI oversold
            buy_condition2 = df['ma_cross'].iloc[i-1] == 1  # MA crossover bullish
            buy_condition3 = df['momentum'].iloc[i-1] > 0  # Positive momentum
            
            # Sell signal conditions
            sell_condition1 = df['rsi'].iloc[i-1] > self.overbought  # RSI overbought
            sell_condition2 = df['ma_cross'].iloc[i-1] == -1  # MA crossover bearish
            sell_condition3 = df['momentum'].iloc[i-1] < 0  # Negative momentum
            
            # Generate signals
            if trend_up and ((buy_condition1 and buy_condition2) or (buy_condition1 and buy_condition3) or (buy_condition2 and buy_condition3)):
                df.loc[df.index[i], 'signal'] = 1  # Buy signal
            elif trend_down and ((sell_condition1 and sell_condition2) or (sell_condition1 and sell_condition3) or (sell_condition2 and sell_condition3)):
                df.loc[df.index[i], 'signal'] = -1  # Sell signal
        
        # 使用基类的市场环境调整方法
        if self.use_market_regime:
            df = self.adjust_for_market_regime(df, df)
        
        return df
    
    def get_position_size(self, df: pd.DataFrame, current_price: float) -> float:
        """根据市场环境和信号强度动态调整仓位大小"""
        # 使用基类的position_size方法
        signal = df['signal'].iloc[-1] if 'signal' in df.columns else 0
        return super().get_position_size(df, signal)
    
    def get_stop_loss(self, df: pd.DataFrame, entry_price: float, position: str) -> float:
        """根据市场环境动态调整止损价格"""
        # 使用基类的stop_loss方法
        pos = 1 if position == 'long' else -1
        market_regime = self.get_market_regime(df)
        return super().get_stop_loss(entry_price, pos, market_regime.value)
    
    def get_take_profit(self, df: pd.DataFrame, entry_price: float, position: str) -> float:
        """根据市场环境动态调整止盈价格"""
        # 使用基类的take_profit方法
        pos = 1 if position == 'long' else -1
        market_regime = self.get_market_regime(df)
        return super().get_take_profit(entry_price, pos, market_regime.value)
    
    def analyze(self, data: pd.DataFrame, market_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析市场状态并提供建议"""
        # 使用基类的analyze方法
        analysis_result = super().analyze(data, market_state)
        
        # 添加策略特定的分析
        signal_components = self.extract_signal_components(data)
        score = self._calculate_score(signal_components)
        market_regime = self.get_market_regime(data)
        
        analysis_result.update({
            'rsi_state': 'oversold' if signal_components['rsi'].iloc[-1] < self.oversold else 'overbought' if signal_components['rsi'].iloc[-1] > self.overbought else 'neutral',
            'ma_trend': 'bullish' if signal_components['ma_cross'].iloc[-1] > 0 else 'bearish',
            'momentum_state': 'positive' if signal_components['momentum'].iloc[-1] > 0 else 'negative',
            'strategy_score': score,
            'market_regime': market_regime.value
        })
        
        return analysis_result
    
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
                "interpretation": "Positive momentum supports trend continuation"
            }
        }
    
    def get_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """判断波动率状态"""
        if 'volatility' not in df.columns:
            df = self.calculate_indicators(df)
            
        # 计算波动率分位数
        vol_percentile = df['volatility'].rolling(window=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 根据分位数判断波动率状态
        volatility_regime = pd.Series(index=df.index, dtype=str)
        volatility_regime[vol_percentile >= 0.7] = 'high'
        volatility_regime[vol_percentile <= 0.3] = 'low'
        volatility_regime[(vol_percentile > 0.3) & (vol_percentile < 0.7)] = 'normal'
        
        return volatility_regime 