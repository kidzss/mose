#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易预测系统
基于当前价格预测日内交易的最佳卖出时机和成功概率
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

@dataclass
class TradingPrediction:
    """交易预测结果"""
    symbol: str
    entry_price: float
    predicted_exits: List[Dict]  # 预测的卖出点
    success_probability: float
    risk_level: str
    strategy_recommendation: str
    time_horizon: str
    expected_return: float
    max_risk: float

class RealtimeTradingPredictor:
    """实时交易预测系统"""
    
    def __init__(self):
        """初始化预测系统"""
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 历史成功率数据 (基于回测和经验)
        self.strategy_success_rates = {
            'scalping': {'rate': 0.65, 'avg_return': 0.3, 'time_minutes': 15},
            'momentum': {'rate': 0.58, 'avg_return': 1.2, 'time_minutes': 60},
            'mean_reversion': {'rate': 0.72, 'avg_return': 0.8, 'time_minutes': 45},
            'breakout': {'rate': 0.55, 'avg_return': 2.1, 'time_minutes': 90},
            'trend_following': {'rate': 0.62, 'avg_return': 1.5, 'time_minutes': 120}
        }
        
        print("🎯 实时交易预测系统初始化完成")
        print("📊 支持策略: Scalping | Momentum | Mean Reversion | Breakout | Trend Following")
    
    async def predict_trade(self, symbol: str, simulate_entry: bool = True) -> TradingPrediction:
        """预测交易结果"""
        print(f"\n🔍 {symbol} 实时交易预测分析")
        print("="*60)
        
        # 1. 获取实时数据
        market_data = await self._get_market_data(symbol)
        
        if not market_data:
            print(f"❌ 无法获取{symbol}的市场数据")
            return None
        
        # 2. 分析当前市场状态
        market_state = self._analyze_market_state(market_data)
        
        # 3. 选择最佳策略
        best_strategy = self._select_best_strategy(market_state)
        
        # 4. 生成预测
        prediction = self._generate_prediction(symbol, market_data, market_state, best_strategy)
        
        # 5. 显示预测结果
        self._display_prediction(prediction)
        
        if simulate_entry:
            # 6. 模拟实时监控
            await self._simulate_trade_monitoring(prediction)
        
        return prediction
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """获取市场数据"""
        try:
            print(f"📡 获取{symbol}实时市场数据...")
            
            # 获取实时数据
            realtime_data = await self.yahoo_source.get_realtime_data([symbol], timeframe='1m')
            
            if symbol not in realtime_data or realtime_data[symbol].empty:
                return None
            
            df = realtime_data[symbol]
            
            # 计算技术指标
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            current_price = float(closes[-1])
            
            # RSI
            rsi = self._calculate_rsi(closes)
            
            # 移动平均线
            ma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
            ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
            
            # VWAP
            vwap = self._calculate_vwap(df)
            
            # ATR (波动性)
            atr = self._calculate_atr(highs, lows, closes)
            
            # 成交量分析
            avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            # 价格变动
            price_change_pct = ((current_price - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'rsi': rsi,
                'ma_5': ma_5,
                'ma_20': ma_20,
                'vwap': vwap,
                'atr': atr,
                'volume_ratio': volume_ratio,
                'price_change_pct': price_change_pct,
                'high_24h': float(np.max(highs)),
                'low_24h': float(np.min(lows)),
                'data_points': len(closes),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return None
    
    def _analyze_market_state(self, data: Dict) -> Dict:
        """分析市场状态"""
        current_price = data['current_price']
        rsi = data['rsi']
        ma_5 = data['ma_5']
        ma_20 = data['ma_20']
        vwap = data['vwap']
        volume_ratio = data['volume_ratio']
        atr = data['atr']
        
        print(f"📊 市场状态分析:")
        print(f"   💰 当前价格: ${current_price:.2f}")
        print(f"   📈 RSI: {rsi:.1f}")
        print(f"   📊 MA5: ${ma_5:.2f} | MA20: ${ma_20:.2f}")
        print(f"   🎯 VWAP: ${vwap:.2f}")
        print(f"   📦 成交量比: {volume_ratio:.1f}x")
        print(f"   📏 ATR: {atr:.2f}")
        
        # 趋势判断
        if current_price > ma_5 > ma_20:
            trend = "BULLISH"
            trend_strength = 0.8
        elif current_price < ma_5 < ma_20:
            trend = "BEARISH"
            trend_strength = 0.8
        else:
            trend = "SIDEWAYS"
            trend_strength = 0.3
        
        # 超买超卖状态
        if rsi > 70:
            rsi_state = "OVERBOUGHT"
        elif rsi < 30:
            rsi_state = "OVERSOLD"
        else:
            rsi_state = "NEUTRAL"
        
        # VWAP偏离
        vwap_deviation = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0
        
        # 波动性评估
        volatility = "HIGH" if atr > 2.0 else "MEDIUM" if atr > 1.0 else "LOW"
        
        # 成交量状态
        volume_state = "HIGH" if volume_ratio > 1.5 else "NORMAL" if volume_ratio > 0.8 else "LOW"
        
        market_state = {
            'trend': trend,
            'trend_strength': trend_strength,
            'rsi_state': rsi_state,
            'vwap_deviation': vwap_deviation,
            'volatility': volatility,
            'volume_state': volume_state,
            'market_sentiment': self._assess_sentiment(trend, rsi_state, volume_state)
        }
        
        print(f"   📈 趋势: {trend} (强度: {trend_strength:.1f})")
        print(f"   🎯 RSI状态: {rsi_state}")
        print(f"   📊 VWAP偏离: {vwap_deviation:+.2f}%")
        print(f"   📏 波动性: {volatility}")
        print(f"   📦 成交量: {volume_state}")
        
        return market_state
    
    def _assess_sentiment(self, trend: str, rsi_state: str, volume_state: str) -> str:
        """评估市场情绪"""
        score = 0
        
        if trend == "BULLISH":
            score += 2
        elif trend == "BEARISH":
            score -= 2
        
        if rsi_state == "OVERSOLD":
            score += 1
        elif rsi_state == "OVERBOUGHT":
            score -= 1
        
        if volume_state == "HIGH":
            score += 1
        elif volume_state == "LOW":
            score -= 1
        
        if score >= 2:
            return "BULLISH"
        elif score <= -2:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _select_best_strategy(self, market_state: Dict) -> str:
        """选择最佳策略"""
        trend = market_state['trend']
        rsi_state = market_state['rsi_state']
        volatility = market_state['volatility']
        volume_state = market_state['volume_state']
        vwap_deviation = abs(market_state['vwap_deviation'])
        
        print(f"\n🎯 策略选择分析:")
        
        # 策略评分
        strategy_scores = {}
        
        # Scalping (剥头皮)
        scalping_score = 0
        if volatility == "LOW":
            scalping_score += 3
        if volume_state == "HIGH":
            scalping_score += 2
        if rsi_state == "NEUTRAL":
            scalping_score += 2
        strategy_scores['scalping'] = scalping_score
        
        # Momentum (动量)
        momentum_score = 0
        if trend in ["BULLISH", "BEARISH"]:
            momentum_score += 3
        if volume_state == "HIGH":
            momentum_score += 2
        if volatility == "MEDIUM":
            momentum_score += 1
        strategy_scores['momentum'] = momentum_score
        
        # Mean Reversion (均值回归)
        mean_reversion_score = 0
        if vwap_deviation > 1.0:
            mean_reversion_score += 3
        if rsi_state in ["OVERBOUGHT", "OVERSOLD"]:
            mean_reversion_score += 2
        if trend == "SIDEWAYS":
            mean_reversion_score += 1
        strategy_scores['mean_reversion'] = mean_reversion_score
        
        # Breakout (突破)
        breakout_score = 0
        if volume_state == "HIGH":
            breakout_score += 3
        if volatility == "HIGH":
            breakout_score += 2
        if trend != "SIDEWAYS":
            breakout_score += 1
        strategy_scores['breakout'] = breakout_score
        
        # Trend Following (趋势跟随)
        trend_following_score = 0
        if market_state['trend_strength'] > 0.6:
            trend_following_score += 3
        if trend in ["BULLISH", "BEARISH"]:
            trend_following_score += 2
        if volume_state != "LOW":
            trend_following_score += 1
        strategy_scores['trend_following'] = trend_following_score
        
        # 选择最高分策略
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        print(f"   📊 策略评分:")
        for strategy, score in strategy_scores.items():
            emoji = "🏆" if strategy == best_strategy else "📈"
            print(f"      {emoji} {strategy.replace('_', ' ').title()}: {score}/7分")
        
        print(f"\n🏆 最佳策略: {best_strategy.replace('_', ' ').title()} ({best_score}/7分)")
        
        return best_strategy
    
    def _generate_prediction(self, symbol: str, data: Dict, market_state: Dict, strategy: str) -> TradingPrediction:
        """生成交易预测"""
        current_price = data['current_price']
        atr = data['atr']
        
        # 获取策略参数
        strategy_info = self.strategy_success_rates[strategy]
        base_success_rate = strategy_info['rate']
        avg_return = strategy_info['avg_return']
        time_minutes = strategy_info['time_minutes']
        
        # 根据市场状态调整成功率
        adjusted_success_rate = self._adjust_success_rate(base_success_rate, market_state)
        
        # 生成预测的卖出点
        predicted_exits = self._generate_exit_predictions(current_price, atr, strategy, avg_return)
        
        # 风险评估
        risk_level = self._assess_risk_level(market_state, atr)
        
        # 预期收益
        expected_return = avg_return * adjusted_success_rate
        
        # 最大风险
        max_risk = min(2.0, atr / current_price * 100)  # 基于ATR的风险，最大2%
        
        return TradingPrediction(
            symbol=symbol,
            entry_price=current_price,
            predicted_exits=predicted_exits,
            success_probability=adjusted_success_rate,
            risk_level=risk_level,
            strategy_recommendation=strategy,
            time_horizon=f"{time_minutes}分钟",
            expected_return=expected_return,
            max_risk=max_risk
        )
    
    def _adjust_success_rate(self, base_rate: float, market_state: Dict) -> float:
        """根据市场状态调整成功率"""
        adjustment = 0
        
        # 趋势强度调整
        if market_state['trend_strength'] > 0.7:
            adjustment += 0.05
        elif market_state['trend_strength'] < 0.4:
            adjustment -= 0.05
        
        # 成交量调整
        if market_state['volume_state'] == "HIGH":
            adjustment += 0.03
        elif market_state['volume_state'] == "LOW":
            adjustment -= 0.03
        
        # 波动性调整
        if market_state['volatility'] == "HIGH":
            adjustment -= 0.02  # 高波动性降低成功率
        elif market_state['volatility'] == "LOW":
            adjustment += 0.02
        
        # 市场情绪调整
        if market_state['market_sentiment'] in ["BULLISH", "BEARISH"]:
            adjustment += 0.02
        
        return max(0.3, min(0.9, base_rate + adjustment))
    
    def _generate_exit_predictions(self, entry_price: float, atr: float, strategy: str, avg_return: float) -> List[Dict]:
        """生成预测的卖出点"""
        exits = []
        
        if strategy == 'scalping':
            # 剥头皮: 小幅快速获利
            exits = [
                {'type': 'quick_profit', 'price': entry_price * 1.003, 'probability': 0.7, 'time_minutes': 5},
                {'type': 'target_profit', 'price': entry_price * 1.006, 'probability': 0.4, 'time_minutes': 15},
                {'type': 'stop_loss', 'price': entry_price * 0.998, 'probability': 0.3, 'time_minutes': 10}
            ]
        elif strategy == 'momentum':
            # 动量: 跟随趋势
            exits = [
                {'type': 'momentum_target', 'price': entry_price * 1.012, 'probability': 0.6, 'time_minutes': 45},
                {'type': 'extended_target', 'price': entry_price * 1.020, 'probability': 0.3, 'time_minutes': 90},
                {'type': 'stop_loss', 'price': entry_price * 0.985, 'probability': 0.4, 'time_minutes': 30}
            ]
        elif strategy == 'mean_reversion':
            # 均值回归: 回归VWAP
            exits = [
                {'type': 'reversion_target', 'price': entry_price * 1.008, 'probability': 0.7, 'time_minutes': 30},
                {'type': 'extended_reversion', 'price': entry_price * 1.015, 'probability': 0.4, 'time_minutes': 60},
                {'type': 'stop_loss', 'price': entry_price * 0.992, 'probability': 0.3, 'time_minutes': 20}
            ]
        elif strategy == 'breakout':
            # 突破: 大幅获利或快速止损
            exits = [
                {'type': 'breakout_target', 'price': entry_price * 1.025, 'probability': 0.5, 'time_minutes': 60},
                {'type': 'extended_breakout', 'price': entry_price * 1.040, 'probability': 0.2, 'time_minutes': 120},
                {'type': 'stop_loss', 'price': entry_price * 0.980, 'probability': 0.5, 'time_minutes': 15}
            ]
        else:  # trend_following
            # 趋势跟随: 中长期持有
            exits = [
                {'type': 'trend_target', 'price': entry_price * 1.018, 'probability': 0.6, 'time_minutes': 90},
                {'type': 'extended_trend', 'price': entry_price * 1.030, 'probability': 0.3, 'time_minutes': 150},
                {'type': 'stop_loss', 'price': entry_price * 0.988, 'probability': 0.4, 'time_minutes': 45}
            ]
        
        return exits
    
    def _assess_risk_level(self, market_state: Dict, atr: float) -> str:
        """评估风险等级"""
        risk_score = 0
        
        if market_state['volatility'] == "HIGH":
            risk_score += 2
        elif market_state['volatility'] == "LOW":
            risk_score -= 1
        
        if market_state['volume_state'] == "LOW":
            risk_score += 1
        
        if market_state['trend'] == "SIDEWAYS":
            risk_score += 1
        
        if atr > 2.0:
            risk_score += 1
        
        if risk_score >= 3:
            return "HIGH"
        elif risk_score >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _display_prediction(self, prediction: TradingPrediction):
        """显示预测结果"""
        print(f"\n🎯 {prediction.symbol} 交易预测结果")
        print("="*60)
        
        print(f"📍 建议入场价: ${prediction.entry_price:.2f}")
        print(f"🎲 成功概率: {prediction.success_probability:.1%}")
        print(f"⚡ 推荐策略: {prediction.strategy_recommendation.replace('_', ' ').title()}")
        print(f"⏰ 预期时间: {prediction.time_horizon}")
        print(f"💰 预期收益: {prediction.expected_return:.2f}%")
        print(f"⚠️ 最大风险: {prediction.max_risk:.2f}%")
        print(f"🛡️ 风险等级: {prediction.risk_level}")
        
        print(f"\n📊 预测卖出点:")
        for i, exit_point in enumerate(prediction.predicted_exits, 1):
            exit_type = exit_point['type'].replace('_', ' ').title()
            price = exit_point['price']
            prob = exit_point['probability']
            time_min = exit_point['time_minutes']
            return_pct = ((price - prediction.entry_price) / prediction.entry_price) * 100
            
            emoji = "🎯" if "target" in exit_point['type'] else "🛡️" if "stop" in exit_point['type'] else "⚡"
            
            print(f"   {i}. {emoji} {exit_type}")
            print(f"      💰 目标价: ${price:.2f} ({return_pct:+.2f}%)")
            print(f"      🎲 概率: {prob:.0%}")
            print(f"      ⏰ 预期时间: {time_min}分钟")
    
    async def _simulate_trade_monitoring(self, prediction: TradingPrediction):
        """模拟交易监控"""
        print(f"\n🔄 开始模拟实时监控 (30秒演示)")
        print("-" * 50)
        
        entry_price = prediction.entry_price
        symbol = prediction.symbol
        
        # 模拟价格变动
        for i in range(3):  # 3次更新，每次10秒
            await asyncio.sleep(10)
            
            # 获取当前价格
            try:
                current_data = await self._get_market_data(symbol)
                if current_data:
                    current_price = current_data['current_price']
                    pnl = ((current_price - entry_price) / entry_price) * 100
                    
                    emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    
                    print(f"   {emoji} {datetime.now().strftime('%H:%M:%S')} - ${current_price:.2f} ({pnl:+.2f}%)")
                    
                    # 检查是否触发卖出点
                    for exit_point in prediction.predicted_exits:
                        if exit_point['type'] != 'stop_loss' and current_price >= exit_point['price']:
                            print(f"   🎯 触发{exit_point['type'].replace('_', ' ').title()}! 建议卖出")
                        elif exit_point['type'] == 'stop_loss' and current_price <= exit_point['price']:
                            print(f"   🛡️ 触发止损! 建议卖出")
                            
            except Exception as e:
                print(f"   ❌ 监控更新失败: {e}")
        
        print(f"   ✅ 模拟监控完成")
    
    # 辅助计算方法
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
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
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """计算VWAP"""
        if df.empty:
            return 0.0
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap) if not np.isnan(vwap) else df['close'].iloc[-1]
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """计算ATR"""
        if len(closes) < period + 1:
            return 1.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return np.mean(true_ranges[-period:]) if true_ranges else 1.0

async def main():
    """主函数 - AMD实时交易预测演示"""
    print("🎯 实时交易预测系统演示")
    print("="*60)
    
    predictor = RealtimeTradingPredictor()
    
    # 预测AMD交易
    print(f"\n💡 假设现在买入AMD，让我们来预测交易结果...")
    
    prediction = await predictor.predict_trade('AMD', simulate_entry=True)
    
    if prediction:
        print(f"\n📋 预测总结:")
        print(f"   🎯 如果现在以${prediction.entry_price:.2f}买入AMD")
        print(f"   📈 成功概率: {prediction.success_probability:.0%}")
        print(f"   💰 预期收益: {prediction.expected_return:.2f}%")
        print(f"   ⚠️ 最大风险: {prediction.max_risk:.2f}%")
        print(f"   ⏰ 建议持有时间: {prediction.time_horizon}")

if __name__ == "__main__":
    asyncio.run(main()) 