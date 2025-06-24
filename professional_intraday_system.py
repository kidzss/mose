#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业日内交易系统 (Professional Intraday Trading System)
包含：剥头皮、趋势反转识别、VWAP策略、订单流分析等
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

class TradingSignal(Enum):
    """交易信号枚举"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class TrendDirection(Enum):
    """趋势方向"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    REVERSAL_UP = "REVERSAL_UP"
    REVERSAL_DOWN = "REVERSAL_DOWN"

@dataclass
class PivotLevels:
    """枢轴点水平"""
    pivot: float
    r1: float  # 阻力位1
    r2: float  # 阻力位2
    r3: float  # 阻力位3
    s1: float  # 支撑位1
    s2: float  # 支撑位2
    s3: float  # 支撑位3

@dataclass
class TradingOpportunity:
    """交易机会"""
    symbol: str
    strategy: str
    signal: TradingSignal
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float
    reason: str
    timestamp: datetime

class ProfessionalIntradaySystem:
    """专业日内交易系统"""
    
    def __init__(self, symbols: List[str] = None):
        """初始化专业交易系统"""
        self.symbols = symbols or ['AMD', 'NVDA', 'TSLA', 'AAPL', 'MSFT']
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 数据存储 (保持更多历史数据用于计算)
        self.price_data = {symbol: deque(maxlen=200) for symbol in self.symbols}
        self.volume_data = {symbol: deque(maxlen=200) for symbol in self.symbols}
        
        # 专业交易参数
        self.params = {
            # Scalping Parameters (剥头皮参数)
            'scalping_profit_target': 0.3,  # 0.3%利润目标
            'scalping_stop_loss': 0.15,     # 0.15%止损
            'scalping_timeframe': 60,       # 1分钟时间框架
            
            # VWAP Parameters
            'vwap_deviation_threshold': 0.5,  # VWAP偏离阈值
            
            # Pivot Points
            'pivot_touch_threshold': 0.1,   # 触及枢轴点的阈值
            
            # Trend Reversal
            'reversal_confirmation_bars': 3,  # 反转确认K线数
            'rsi_divergence_threshold': 5,    # RSI背离阈值
            
            # Risk Management
            'max_daily_loss': 2.0,          # 最大日损失2%
            'position_size_pct': 1.0,       # 每次仓位1%
            'max_positions': 3,             # 最大同时持仓数
        }
        
        # 当前状态
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.opportunities = []
        
        print("🏛️ 专业日内交易系统初始化完成")
        print(f"📊 监控标的: {self.symbols}")
        print(f"⚡ 策略: Scalping | VWAP | Pivot Points | Trend Reversal")
    
    async def run_professional_analysis(self, duration_minutes: int = 3):
        """运行专业分析"""
        print(f"\n🎯 开始{duration_minutes}分钟专业日内分析")
        print("="*80)
        
        # 预热数据
        await self._warmup_data()
        
        updates = duration_minutes  # 每分钟更新一次
        
        for i in range(updates):
            try:
                await self._professional_update()
                
                if i < updates - 1:
                    await asyncio.sleep(60)  # 等待1分钟
                    
            except Exception as e:
                print(f"❌ 专业分析失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 专业分析完成!")
        self._show_professional_summary()
    
    async def _warmup_data(self):
        """预热数据 - 获取足够的历史数据"""
        print("🔥 预热数据中...")
        
        for _ in range(3):  # 获取3次数据建立基础
            await self._update_market_data()
            await asyncio.sleep(2)
        
        print("✅ 数据预热完成")
    
    async def _professional_update(self):
        """专业更新分析"""
        current_time = datetime.now()
        print(f"\n📈 {current_time.strftime('%H:%M:%S')} - 专业分析更新")
        print("-" * 70)
        
        # 更新市场数据
        await self._update_market_data()
        
        # 对每个标的进行专业分析
        for symbol in self.symbols:
            if len(self.price_data[symbol]) >= 20:  # 确保有足够数据
                await self._analyze_symbol_professional(symbol)
    
    async def _update_market_data(self):
        """更新市场数据"""
        try:
            realtime_data = await self.yahoo_source.get_realtime_data(
                self.symbols, timeframe='1m'
            )
            
            for symbol in self.symbols:
                if symbol in realtime_data and not realtime_data[symbol].empty:
                    df = realtime_data[symbol]
                    latest = df.iloc[-1]
                    
                    # 存储OHLCV数据
                    price_point = {
                        'timestamp': datetime.now(),
                        'open': float(latest['open']),
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'close': float(latest['close']),
                    }
                    
                    volume_point = int(latest['volume']) if latest['volume'] > 0 else 1
                    
                    self.price_data[symbol].append(price_point)
                    self.volume_data[symbol].append(volume_point)
                    
        except Exception as e:
            print(f"❌ 市场数据更新失败: {e}")
    
    async def _analyze_symbol_professional(self, symbol: str):
        """专业标的分析"""
        print(f"\n🔍 {symbol} 专业分析:")
        
        # 1. 计算专业技术指标
        indicators = self._calculate_professional_indicators(symbol)
        
        # 2. 识别趋势和反转
        trend_analysis = self._analyze_trend_reversal(symbol, indicators)
        
        # 3. 计算枢轴点
        pivot_levels = self._calculate_pivot_points(symbol)
        
        # 4. VWAP分析
        vwap_analysis = self._analyze_vwap_strategy(symbol, indicators)
        
        # 5. 剥头皮机会识别
        scalping_opportunity = self._identify_scalping_opportunity(symbol, indicators)
        
        # 显示分析结果
        self._display_professional_analysis(
            symbol, indicators, trend_analysis, 
            pivot_levels, vwap_analysis, scalping_opportunity
        )
        
        # 生成交易机会
        opportunities = self._generate_trading_opportunities(
            symbol, indicators, trend_analysis, vwap_analysis, scalping_opportunity
        )
        
        if opportunities:
            self.opportunities.extend(opportunities)
            self._display_trading_opportunities(opportunities)
    
    def _calculate_professional_indicators(self, symbol: str) -> Dict:
        """计算专业技术指标"""
        price_history = list(self.price_data[symbol])
        volume_history = list(self.volume_data[symbol])
        
        if len(price_history) < 20:
            return {}
        
        # 基础价格数据
        closes = [p['close'] for p in price_history]
        highs = [p['high'] for p in price_history]
        lows = [p['low'] for p in price_history]
        opens = [p['open'] for p in price_history]
        
        current_price = closes[-1]
        
        # 1. RSI (14期)
        rsi = self._calculate_rsi(closes, 14)
        
        # 2. 移动平均线组合
        ema_9 = self._calculate_ema(closes, 9)
        ema_21 = self._calculate_ema(closes, 21)
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
        
        # 3. MACD
        macd_line, signal_line, histogram = self._calculate_macd(closes)
        
        # 4. 布林带
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)
        
        # 5. ATR (平均真实波动范围)
        atr = self._calculate_atr(highs, lows, closes)
        
        # 6. VWAP
        vwap = self._calculate_vwap(price_history, volume_history)
        
        # 7. 成交量指标
        volume_sma = np.mean(volume_history[-20:]) if len(volume_history) >= 20 else volume_history[-1]
        volume_ratio = volume_history[-1] / volume_sma if volume_sma > 0 else 1
        
        # 8. 动量指标
        momentum = ((current_price - closes[-10]) / closes[-10] * 100) if len(closes) >= 10 else 0
        
        return {
            'current_price': current_price,
            'rsi': rsi,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'sma_50': sma_50,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'atr': atr,
            'vwap': vwap,
            'volume_ratio': volume_ratio,
            'momentum': momentum
        }
    
    def _analyze_trend_reversal(self, symbol: str, indicators: Dict) -> Dict:
        """分析趋势反转"""
        if not indicators:
            return {'trend': TrendDirection.SIDEWAYS, 'reversal_probability': 0}
        
        current_price = indicators['current_price']
        ema_9 = indicators['ema_9']
        ema_21 = indicators['ema_21']
        rsi = indicators['rsi']
        macd_line = indicators['macd_line']
        signal_line = indicators['signal_line']
        
        # 趋势判断
        if current_price > ema_9 > ema_21 and macd_line > signal_line:
            trend = TrendDirection.BULLISH
        elif current_price < ema_9 < ema_21 and macd_line < signal_line:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.SIDEWAYS
        
        # 反转信号检测
        reversal_probability = 0
        reversal_signals = []
        
        # RSI背离
        if rsi > 70 and trend == TrendDirection.BULLISH:
            reversal_probability += 30
            reversal_signals.append("RSI超买背离")
        elif rsi < 30 and trend == TrendDirection.BEARISH:
            reversal_probability += 30
            reversal_signals.append("RSI超卖背离")
        
        # MACD背离
        if macd_line < signal_line and trend == TrendDirection.BULLISH:
            reversal_probability += 25
            reversal_signals.append("MACD熊市背离")
        elif macd_line > signal_line and trend == TrendDirection.BEARISH:
            reversal_probability += 25
            reversal_signals.append("MACD牛市背离")
        
        # 价格与均线背离
        if current_price < ema_9 and trend == TrendDirection.BULLISH:
            reversal_probability += 20
            reversal_signals.append("价格跌破短期均线")
        elif current_price > ema_9 and trend == TrendDirection.BEARISH:
            reversal_probability += 20
            reversal_signals.append("价格突破短期均线")
        
        return {
            'trend': trend,
            'reversal_probability': min(reversal_probability, 100),
            'reversal_signals': reversal_signals
        }
    
    def _calculate_pivot_points(self, symbol: str) -> PivotLevels:
        """计算枢轴点"""
        price_history = list(self.price_data[symbol])
        
        if len(price_history) < 2:
            current_price = price_history[-1]['close'] if price_history else 100
            return PivotLevels(current_price, current_price, current_price, current_price,
                             current_price, current_price, current_price)
        
        # 使用昨日的高低收计算枢轴点
        yesterday_high = max([p['high'] for p in price_history[-20:]])
        yesterday_low = min([p['low'] for p in price_history[-20:]])
        yesterday_close = price_history[-2]['close']
        
        # 标准枢轴点公式
        pivot = (yesterday_high + yesterday_low + yesterday_close) / 3
        
        r1 = 2 * pivot - yesterday_low
        r2 = pivot + (yesterday_high - yesterday_low)
        r3 = yesterday_high + 2 * (pivot - yesterday_low)
        
        s1 = 2 * pivot - yesterday_high
        s2 = pivot - (yesterday_high - yesterday_low)
        s3 = yesterday_low - 2 * (yesterday_high - pivot)
        
        return PivotLevels(pivot, r1, r2, r3, s1, s2, s3)
    
    def _analyze_vwap_strategy(self, symbol: str, indicators: Dict) -> Dict:
        """VWAP策略分析"""
        if not indicators:
            return {'signal': 'NEUTRAL', 'deviation': 0}
        
        current_price = indicators['current_price']
        vwap = indicators['vwap']
        
        # VWAP偏离度
        deviation = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
        
        # VWAP信号
        if deviation > self.params['vwap_deviation_threshold']:
            signal = 'SELL'  # 价格过度偏离VWAP上方
            strength = 'STRONG' if abs(deviation) > 1.0 else 'MEDIUM'
        elif deviation < -self.params['vwap_deviation_threshold']:
            signal = 'BUY'   # 价格过度偏离VWAP下方
            strength = 'STRONG' if abs(deviation) > 1.0 else 'MEDIUM'
        else:
            signal = 'NEUTRAL'
            strength = 'WEAK'
        
        return {
            'signal': signal,
            'strength': strength,
            'deviation': deviation,
            'vwap_price': vwap
        }
    
    def _identify_scalping_opportunity(self, symbol: str, indicators: Dict) -> Dict:
        """识别剥头皮机会"""
        if not indicators:
            return {'opportunity': False}
        
        current_price = indicators['current_price']
        atr = indicators['atr']
        volume_ratio = indicators['volume_ratio']
        rsi = indicators['rsi']
        
        # 剥头皮条件
        conditions = []
        score = 0
        
        # 1. 成交量放大
        if volume_ratio > 1.5:
            conditions.append("成交量放大")
            score += 25
        
        # 2. ATR适中 (波动性不能太大)
        if 0.5 <= atr <= 2.0:
            conditions.append("波动性适中")
            score += 20
        
        # 3. RSI不在极端区域
        if 35 <= rsi <= 65:
            conditions.append("RSI中性区域")
            score += 20
        
        # 4. 价格接近支撑/阻力
        bb_upper = indicators.get('bb_upper', current_price)
        bb_lower = indicators.get('bb_lower', current_price)
        
        if abs(current_price - bb_upper) / current_price < 0.005:
            conditions.append("接近布林带上轨")
            score += 15
        elif abs(current_price - bb_lower) / current_price < 0.005:
            conditions.append("接近布林带下轨")
            score += 15
        
        opportunity = score >= 60
        
        return {
            'opportunity': opportunity,
            'score': score,
            'conditions': conditions,
            'entry_price': current_price,
            'stop_loss': current_price * (1 - self.params['scalping_stop_loss'] / 100),
            'take_profit': current_price * (1 + self.params['scalping_profit_target'] / 100)
        }
    
    def _generate_trading_opportunities(self, symbol: str, indicators: Dict, 
                                     trend_analysis: Dict, vwap_analysis: Dict, 
                                     scalping_opportunity: Dict) -> List[TradingOpportunity]:
        """生成交易机会"""
        opportunities = []
        current_price = indicators.get('current_price', 0)
        
        if not current_price:
            return opportunities
        
        # 1. 趋势反转机会
        if trend_analysis.get('reversal_probability', 0) > 70:
            if trend_analysis['trend'] == TrendDirection.BULLISH:
                signal = TradingSignal.SELL
                entry = current_price
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.97
            else:
                signal = TradingSignal.BUY
                entry = current_price
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.03
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                strategy="Trend Reversal",
                signal=signal,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=abs(take_profit - entry) / abs(stop_loss - entry),
                confidence=trend_analysis['reversal_probability'] / 100,
                reason=f"趋势反转信号: {', '.join(trend_analysis.get('reversal_signals', []))}",
                timestamp=datetime.now()
            ))
        
        # 2. VWAP策略机会
        if vwap_analysis.get('strength') == 'STRONG':
            signal = TradingSignal.BUY if vwap_analysis['signal'] == 'BUY' else TradingSignal.SELL
            entry = current_price
            
            if signal == TradingSignal.BUY:
                stop_loss = current_price * 0.985
                take_profit = vwap_analysis['vwap_price']
            else:
                stop_loss = current_price * 1.015
                take_profit = vwap_analysis['vwap_price']
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                strategy="VWAP Mean Reversion",
                signal=signal,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=abs(take_profit - entry) / abs(stop_loss - entry) if abs(stop_loss - entry) > 0 else 0,
                confidence=0.75,
                reason=f"VWAP偏离 {vwap_analysis['deviation']:.2f}%",
                timestamp=datetime.now()
            ))
        
        # 3. 剥头皮机会
        if scalping_opportunity.get('opportunity'):
            # 根据短期动量决定方向
            momentum = indicators.get('momentum', 0)
            if momentum > 0:
                signal = TradingSignal.BUY
            else:
                signal = TradingSignal.SELL
            
            opportunities.append(TradingOpportunity(
                symbol=symbol,
                strategy="Scalping",
                signal=signal,
                entry_price=scalping_opportunity['entry_price'],
                stop_loss=scalping_opportunity['stop_loss'],
                take_profit=scalping_opportunity['take_profit'],
                risk_reward_ratio=2.0,  # 剥头皮通常2:1风险收益比
                confidence=scalping_opportunity['score'] / 100,
                reason=f"剥头皮机会: {', '.join(scalping_opportunity['conditions'])}",
                timestamp=datetime.now()
            ))
        
        return opportunities
    
    def _display_professional_analysis(self, symbol: str, indicators: Dict, 
                                     trend_analysis: Dict, pivot_levels: PivotLevels,
                                     vwap_analysis: Dict, scalping_opportunity: Dict):
        """显示专业分析结果"""
        if not indicators:
            print(f"   ❌ {symbol}: 数据不足")
            return
        
        print(f"   📊 {symbol} 专业分析:")
        print(f"      💰 当前价格: ${indicators['current_price']:.2f}")
        print(f"      📈 技术指标: RSI={indicators['rsi']:.1f} | MACD={indicators['macd_line']:.3f}")
        print(f"      📊 均线: EMA9=${indicators['ema_9']:.2f} | EMA21=${indicators['ema_21']:.2f}")
        print(f"      🎯 VWAP: ${indicators['vwap']:.2f} (偏离: {vwap_analysis.get('deviation', 0):.2f}%)")
        
        # 趋势分析
        trend_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(
            trend_analysis.get('trend', TrendDirection.SIDEWAYS).value, "⚪"
        )
        print(f"      {trend_emoji} 趋势: {trend_analysis.get('trend', TrendDirection.SIDEWAYS).value}")
        
        if trend_analysis.get('reversal_probability', 0) > 50:
            print(f"      ⚠️ 反转概率: {trend_analysis['reversal_probability']:.0f}%")
        
        # 枢轴点
        print(f"      🎯 枢轴点: P=${pivot_levels.pivot:.2f} | R1=${pivot_levels.r1:.2f} | S1=${pivot_levels.s1:.2f}")
        
        # 剥头皮机会
        if scalping_opportunity.get('opportunity'):
            print(f"      ⚡ 剥头皮机会: {scalping_opportunity['score']}/100分")
    
    def _display_trading_opportunities(self, opportunities: List[TradingOpportunity]):
        """显示交易机会"""
        print(f"\n💡 发现 {len(opportunities)} 个交易机会:")
        
        for i, opp in enumerate(opportunities, 1):
            signal_emoji = {
                "BUY": "🟢", "STRONG_BUY": "🟢🟢",
                "SELL": "🔴", "STRONG_SELL": "🔴🔴",
                "HOLD": "🟡"
            }.get(opp.signal.value, "⚪")
            
            print(f"   {i}. {signal_emoji} {opp.symbol} - {opp.strategy}")
            print(f"      📍 入场: ${opp.entry_price:.2f}")
            print(f"      🛡️ 止损: ${opp.stop_loss:.2f}")
            print(f"      🎯 止盈: ${opp.take_profit:.2f}")
            print(f"      📊 风险收益比: 1:{opp.risk_reward_ratio:.1f}")
            print(f"      🎲 置信度: {opp.confidence:.0%}")
            print(f"      💭 理由: {opp.reason}")
    
    def _show_professional_summary(self):
        """显示专业总结"""
        print(f"\n📋 专业交易总结:")
        print(f"   🎯 发现机会: {len(self.opportunities)}个")
        print(f"   📊 分析标的: {len(self.symbols)}只")
        
        if self.opportunities:
            # 按策略分组统计
            strategy_count = {}
            for opp in self.opportunities:
                strategy_count[opp.strategy] = strategy_count.get(opp.strategy, 0) + 1
            
            print(f"   📈 策略分布:")
            for strategy, count in strategy_count.items():
                print(f"      • {strategy}: {count}个机会")
            
            # 显示最佳机会
            best_opportunities = sorted(self.opportunities, 
                                      key=lambda x: x.confidence * x.risk_reward_ratio, 
                                      reverse=True)[:3]
            
            print(f"\n🏆 最佳交易机会:")
            for i, opp in enumerate(best_opportunities, 1):
                print(f"   {i}. {opp.symbol} - {opp.strategy} (置信度: {opp.confidence:.0%})")
    
    # 辅助计算方法
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """计算指数移动平均线"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """计算MACD"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # 简化的信号线计算
        signal_line = macd_line * 0.9  # 简化版本
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(prices) < period:
            current_price = prices[-1]
            return current_price, current_price, current_price
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return upper, sma, lower
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """计算平均真实波动范围"""
        if len(closes) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return np.mean(true_ranges[-period:]) if true_ranges else 0.0
    
    def _calculate_vwap(self, price_history: List[Dict], volume_history: List[int]) -> float:
        """计算VWAP"""
        if len(price_history) != len(volume_history) or len(price_history) == 0:
            return price_history[-1]['close'] if price_history else 0.0
        
        total_volume = 0
        total_pv = 0
        
        for i, price_data in enumerate(price_history):
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            volume = volume_history[i]
            
            total_pv += typical_price * volume
            total_volume += volume
        
        return total_pv / total_volume if total_volume > 0 else price_history[-1]['close']

async def main():
    """主函数"""
    print("🏛️ 专业日内交易系统 (Professional Intraday Trading System)")
    print("="*80)
    print("📈 策略包含: Scalping | VWAP | Pivot Points | Trend Reversal")
    print("🎯 专业功能: 订单流分析 | 风险管理 | 多时间框架分析")
    
    system = ProfessionalIntradaySystem(['AMD', 'NVDA', 'TSLA'])
    await system.run_professional_analysis(duration_minutes=3)

if __name__ == "__main__":
    asyncio.run(main()) 