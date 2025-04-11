import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats

@dataclass
class MarketState:
    trend: str  # 上升/下降/震荡
    volatility: float  # 波动率
    momentum: float  # 动量
    volume_trend: str  # 放量/缩量
    support_level: float  # 支撑位
    resistance_level: float  # 压力位
    market_regime: str  # 市场状态（牛市/熊市/震荡市）

class MarketAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        初始化市场分析器
        
        参数:
            df: 市场数据，包含OHLCV数据
        """
        self.df = df
        self._prepare_data()
        
    def _prepare_data(self):
        """准备分析所需的技术指标"""
        # 计算移动平均
        self.df['SMA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        
        # 计算波动率
        self.df['Returns'] = self.df['Close'].pct_change()
        annualization_factor = np.full(len(self.df), np.sqrt(252))
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std().mul(annualization_factor)
        
        # 计算动量
        self.df['Momentum'] = self.df['Close'].pct_change(periods=20)
        
        # 计算成交量变化
        self.df['Volume_MA20'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA20']
        
    def identify_trend(self, window: int = 20) -> str:
        """识别市场趋势"""
        current_price = self.df['Close'].iloc[-1]
        sma = self.df['SMA20'].iloc[-1]
        momentum = self.df['Momentum'].iloc[-1]
        
        if current_price > sma and momentum > 0:
            return "上升"
        elif current_price < sma and momentum < 0:
            return "下降"
        else:
            return "震荡"
            
    def calculate_support_resistance(self, window: int = 20) -> Tuple[float, float]:
        """计算支撑位和压力位"""
        recent_lows = self.df['Low'].rolling(window=window).min()
        recent_highs = self.df['High'].rolling(window=window).max()
        
        support = recent_lows.iloc[-1]
        resistance = recent_highs.iloc[-1]
        
        return support, resistance
        
    def analyze_volume_trend(self, window: int = 20) -> str:
        """分析成交量趋势"""
        volume_ratio = self.df['Volume_Ratio'].iloc[-1]
        
        if volume_ratio > 1.5:
            return "放量"
        elif volume_ratio < 0.7:
            return "缩量"
        else:
            return "正常"
            
    def identify_market_regime(self, window: int = 50) -> str:
        """识别市场状态"""
        returns = self.df['Returns'].iloc[-window:]
        volatility = self.df['Volatility'].iloc[-1]
        trend = self.identify_trend()
        
        if trend == "上升" and volatility < 0.2:
            return "牛市"
        elif trend == "下降" and volatility > 0.3:
            return "熊市"
        else:
            return "震荡市"
            
    def calculate_market_strength(self) -> float:
        """计算市场强度指标"""
        momentum = self.df['Momentum'].iloc[-1]
        volatility = self.df['Volatility'].iloc[-1]
        volume_ratio = self.df['Volume_Ratio'].iloc[-1]
        
        # 综合考虑动量、波动率和成交量
        strength = (
            0.4 * np.sign(momentum) * min(abs(momentum), 1) +
            0.3 * (1 - min(volatility, 0.5) / 0.5) +
            0.3 * min(volume_ratio, 2) / 2
        )
        
        return strength
        
    def analyze_market_state(self) -> MarketState:
        """分析市场状态"""
        trend = self.identify_trend()
        support, resistance = self.calculate_support_resistance()
        volume_trend = self.analyze_volume_trend()
        market_regime = self.identify_market_regime()
        
        return MarketState(
            trend=trend,
            volatility=self.df['Volatility'].iloc[-1],
            momentum=self.df['Momentum'].iloc[-1],
            volume_trend=volume_trend,
            support_level=support,
            resistance_level=resistance,
            market_regime=market_regime
        )
        
    def generate_market_report(self) -> str:
        """生成市场分析报告"""
        state = self.analyze_market_state()
        strength = self.calculate_market_strength()
        
        report = f"""
市场分析报告
===========
市场状态:
- 主要趋势: {state.trend}
- 市场阶段: {state.market_regime}
- 市场强度: {strength:.2f}

技术指标:
- 波动率: {state.volatility:.2%}
- 动量: {state.momentum:.2%}
- 成交量趋势: {state.volume_trend}

价格区间:
- 支撑位: {state.support_level:.2f}
- 压力位: {state.resistance_level:.2f}

市场分析:
1. {self._get_trend_analysis(state)}
2. {self._get_volatility_analysis(state)}
3. {self._get_volume_analysis(state)}
4. {self._get_support_resistance_analysis(state)}

交易建议:
1. {self._get_trading_advice(state, strength)}
2. {self._get_risk_advice(state)}
"""
        return report
        
    def _get_trend_analysis(self, state: MarketState) -> str:
        """根据趋势给出分析"""
        if state.trend == "上升":
            return "市场处于上升趋势，可以考虑顺势操作"
        elif state.trend == "下降":
            return "市场处于下降趋势，建议保持谨慎"
        else:
            return "市场处于震荡阶段，建议等待明确信号"
            
    def _get_volatility_analysis(self, state: MarketState) -> str:
        """根据波动率给出分析"""
        if state.volatility > 0.3:
            return "市场波动较大，建议控制仓位"
        else:
            return "市场波动在正常范围，可以正常交易"
            
    def _get_volume_analysis(self, state: MarketState) -> str:
        """根据成交量给出分析"""
        if state.volume_trend == "放量":
            return "成交量放大，说明市场活跃度增加"
        elif state.volume_trend == "缩量":
            return "成交量萎缩，说明市场观望情绪加重"
        else:
            return "成交量正常，市场交投活跃度适中"
            
    def _get_support_resistance_analysis(self, state: MarketState) -> str:
        """根据支撑压力位给出分析"""
        current_price = self.df['Close'].iloc[-1]
        distance_to_resistance = (state.resistance_level - current_price) / current_price
        distance_to_support = (current_price - state.support_level) / current_price
        
        if distance_to_resistance < 0.02:
            return "价格接近压力位，注意可能的反转"
        elif distance_to_support < 0.02:
            return "价格接近支撑位，关注反弹机会"
        else:
            return "价格处于支撑位和压力位之间，可以根据趋势进行操作"
            
    def _get_trading_advice(self, state: MarketState, strength: float) -> str:
        """根据市场状态和强度给出交易建议"""
        if strength > 0.6 and state.trend == "上升":
            return "市场强势上涨，可以考虑逢低买入"
        elif strength < -0.6 and state.trend == "下降":
            return "市场弱势下跌，建议观望或者做空"
        else:
            return "市场强度一般，建议小仓位试探性操作"
            
    def _get_risk_advice(self, state: MarketState) -> str:
        """根据市场状态给出风险建议"""
        if state.market_regime == "牛市":
            return "市场处于牛市阶段，可以适当提高仓位，但注意设置止损"
        elif state.market_regime == "熊市":
            return "市场处于熊市阶段，建议以防守为主，控制风险"
        else:
            return "市场处于震荡阶段，建议以轻仓为主，严格执行止损" 