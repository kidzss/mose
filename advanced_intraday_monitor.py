#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级实时日内交易监控系统
包含完整的技术分析和交易信号
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List
from collections import deque

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

class AdvancedIntradayMonitor:
    """高级日内交易监控系统"""
    
    def __init__(self, symbols: List[str] = None):
        """初始化监控系统"""
        self.symbols = symbols or ['AMD', 'NVDA', 'TSLA']
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 存储价格历史 (最近50个数据点)
        self.price_history = {symbol: deque(maxlen=50) for symbol in self.symbols}
        
        # 交易参数
        self.params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ma_short': 5,
            'ma_long': 20,
            'volume_spike_threshold': 1.5,
            'price_change_threshold': 1.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0
        }
        
        # 模拟持仓
        self.positions = {}
        
        print(f"🚀 高级日内监控系统初始化完成")
        print(f"📊 监控股票: {self.symbols}")
    
    async def run_demo(self, duration_minutes: int = 3):
        """运行演示"""
        print(f"\n🎯 开始{duration_minutes}分钟监控演示")
        print("="*60)
        
        # 模拟一些持仓
        self.add_position('AMD', 136.00, 100)
        self.add_position('NVDA', 146.00, 50)
        
        updates = duration_minutes * 2  # 每30秒更新一次
        
        for i in range(updates):
            try:
                await self._update_and_analyze()
                
                if i < updates - 1:
                    await asyncio.sleep(30)  # 等待30秒
                    
            except Exception as e:
                print(f"❌ 更新失败: {e}")
        
        print(f"\n📊 演示完成!")
        self._show_summary()
    
    async def _update_and_analyze(self):
        """更新数据并分析"""
        current_time = datetime.now()
        print(f"\n🔄 {current_time.strftime('%H:%M:%S')} - 数据更新")
        print("-" * 50)
        
        # 获取实时数据
        realtime_data = await self.yahoo_source.get_realtime_data(
            self.symbols, timeframe='1m'
        )
        
        for symbol in self.symbols:
            if symbol in realtime_data and not realtime_data[symbol].empty:
                df = realtime_data[symbol]
                latest = df.iloc[-1]
                
                # 更新价格历史
                price_data = {
                    'timestamp': current_time,
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': int(latest['volume']) if latest['volume'] > 0 else 1
                }
                
                self.price_history[symbol].append(price_data)
                
                # 分析信号
                await self._analyze_symbol(symbol, price_data)
    
    async def _analyze_symbol(self, symbol: str, current_data: Dict):
        """分析单个股票的信号"""
        if len(self.price_history[symbol]) < self.params['rsi_period']:
            print(f"📊 {symbol}: 数据不足，等待更多数据...")
            return
        
        # 计算技术指标
        indicators = self._calculate_indicators(symbol)
        
        # 生成信号
        signals = self._generate_signals(symbol, indicators)
        
        # 显示分析结果
        self._display_analysis(symbol, current_data, indicators, signals)
        
        # 处理交易信号
        await self._process_trading_signals(symbol, signals, current_data)
    
    def _calculate_indicators(self, symbol: str) -> Dict:
        """计算技术指标"""
        history = list(self.price_history[symbol])
        
        # 提取价格数据
        closes = [item['close'] for item in history]
        highs = [item['high'] for item in history]
        lows = [item['low'] for item in history]
        volumes = [item['volume'] for item in history]
        
        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) > 1 else current_price
        
        # RSI计算
        rsi = self._calculate_rsi(closes)
        
        # 移动平均线
        ma_short = np.mean(closes[-self.params['ma_short']:]) if len(closes) >= self.params['ma_short'] else current_price
        ma_long = np.mean(closes[-self.params['ma_long']:]) if len(closes) >= self.params['ma_long'] else current_price
        
        # 成交量分析
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # 价格变动
        price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        # 布林带 (简化版)
        bb_period = min(20, len(closes))
        if bb_period >= 10:
            bb_middle = np.mean(closes[-bb_period:])
            bb_std = np.std(closes[-bb_period:])
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
        else:
            bb_middle = bb_upper = bb_lower = current_price
        
        return {
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'rsi': rsi,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'volume_ratio': volume_ratio,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower
        }
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """计算RSI"""
        if len(prices) < self.params['rsi_period'] + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.params['rsi_period']:])
        avg_loss = np.mean(losses[-self.params['rsi_period']:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_signals(self, symbol: str, indicators: Dict) -> List[Dict]:
        """生成交易信号"""
        signals = []
        
        current_price = indicators['current_price']
        rsi = indicators['rsi']
        ma_short = indicators['ma_short']
        ma_long = indicators['ma_long']
        volume_ratio = indicators['volume_ratio']
        price_change_pct = indicators['price_change_pct']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        # RSI信号
        if rsi > self.params['rsi_overbought']:
            signals.append({
                'type': 'RSI_OVERBOUGHT',
                'action': 'SELL',
                'strength': 'MEDIUM',
                'message': f'RSI超买 {rsi:.1f}',
                'priority': 2
            })
        elif rsi < self.params['rsi_oversold']:
            signals.append({
                'type': 'RSI_OVERSOLD',
                'action': 'BUY',
                'strength': 'MEDIUM',
                'message': f'RSI超卖 {rsi:.1f}',
                'priority': 2
            })
        
        # 均线信号
        if current_price > ma_short > ma_long:
            signals.append({
                'type': 'MA_BULLISH',
                'action': 'BUY',
                'strength': 'STRONG',
                'message': '均线多头排列',
                'priority': 3
            })
        elif current_price < ma_short < ma_long:
            signals.append({
                'type': 'MA_BEARISH',
                'action': 'SELL',
                'strength': 'STRONG',
                'message': '均线空头排列',
                'priority': 3
            })
        
        # 成交量信号
        if volume_ratio > self.params['volume_spike_threshold']:
            signals.append({
                'type': 'VOLUME_SPIKE',
                'action': 'WATCH',
                'strength': 'HIGH',
                'message': f'成交量异常 {volume_ratio:.1f}x',
                'priority': 2
            })
        
        # 价格突破信号
        if abs(price_change_pct) > self.params['price_change_threshold']:
            action = 'BUY' if price_change_pct > 0 else 'SELL'
            signals.append({
                'type': 'PRICE_BREAKOUT',
                'action': action,
                'strength': 'HIGH',
                'message': f'价格突破 {price_change_pct:+.2f}%',
                'priority': 3
            })
        
        # 布林带信号
        if current_price >= bb_upper:
            signals.append({
                'type': 'BB_UPPER',
                'action': 'SELL',
                'strength': 'MEDIUM',
                'message': '触及布林带上轨',
                'priority': 2
            })
        elif current_price <= bb_lower:
            signals.append({
                'type': 'BB_LOWER',
                'action': 'BUY',
                'strength': 'MEDIUM',
                'message': '触及布林带下轨',
                'priority': 2
            })
        
        # 按优先级排序
        signals.sort(key=lambda x: x['priority'], reverse=True)
        
        return signals
    
    def _display_analysis(self, symbol: str, current_data: Dict, indicators: Dict, signals: List[Dict]):
        """显示分析结果"""
        print(f"📊 {symbol} 分析:")
        print(f"   💰 价格: ${indicators['current_price']:.2f} ({indicators['price_change_pct']:+.2f}%)")
        print(f"   📈 RSI: {indicators['rsi']:.1f}")
        print(f"   📊 MA5: ${indicators['ma_short']:.2f} | MA20: ${indicators['ma_long']:.2f}")
        print(f"   📦 成交量比: {indicators['volume_ratio']:.1f}x")
        
        # 显示持仓盈亏
        if symbol in self.positions:
            entry_price = self.positions[symbol]['entry_price']
            current_pnl = ((indicators['current_price'] - entry_price) / entry_price) * 100
            emoji = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "⚪"
            print(f"   {emoji} 持仓盈亏: {current_pnl:+.2f}% (入场: ${entry_price:.2f})")
        
        # 显示信号
        if signals:
            print(f"   🎯 交易信号:")
            for signal in signals[:3]:  # 只显示前3个最重要的信号
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'WATCH': '🟡'}.get(signal['action'], '⚪')
                print(f"      {emoji} {signal['message']} - {signal['action']}")
        else:
            print(f"   ⚪ 无明显信号")
    
    async def _process_trading_signals(self, symbol: str, signals: List[Dict], current_data: Dict):
        """处理交易信号并生成建议"""
        if not signals:
            return
        
        # 获取最高优先级的信号
        top_signal = signals[0]
        current_price = current_data['close']
        
        if top_signal['priority'] >= 3 and top_signal['strength'] in ['HIGH', 'STRONG']:
            print(f"   💡 交易建议:")
            
            if top_signal['action'] == 'BUY':
                if symbol not in self.positions:
                    stop_loss = current_price * (1 - self.params['stop_loss_pct'] / 100)
                    take_profit = current_price * (1 + self.params['take_profit_pct'] / 100)
                    
                    print(f"      🟢 建议买入 ${current_price:.2f}")
                    print(f"      🛡️ 止损: ${stop_loss:.2f} (-{self.params['stop_loss_pct']:.1f}%)")
                    print(f"      🎯 止盈: ${take_profit:.2f} (+{self.params['take_profit_pct']:.1f}%)")
                else:
                    print(f"      🟡 已有持仓，建议继续持有")
            
            elif top_signal['action'] == 'SELL':
                if symbol in self.positions:
                    entry_price = self.positions[symbol]['entry_price']
                    pnl = ((current_price - entry_price) / entry_price) * 100
                    print(f"      🔴 建议卖出 ${current_price:.2f}")
                    print(f"      💰 预期盈亏: {pnl:+.2f}%")
                else:
                    print(f"      🟡 无持仓，建议观望")
    
    def add_position(self, symbol: str, entry_price: float, quantity: int):
        """添加持仓"""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now()
        }
        print(f"✅ 模拟持仓: {symbol} @ ${entry_price:.2f} x {quantity}股")
    
    def _show_summary(self):
        """显示总结"""
        print(f"\n📋 监控总结:")
        print(f"   监控股票: {len(self.symbols)}只")
        print(f"   数据点数: {[len(self.price_history[s]) for s in self.symbols]}")
        
        if self.positions:
            print(f"   持仓情况:")
            for symbol, pos in self.positions.items():
                if len(self.price_history[symbol]) > 0:
                    current_price = self.price_history[symbol][-1]['close']
                    pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    print(f"     {emoji} {symbol}: {pnl:+.2f}% (${pos['entry_price']:.2f} → ${current_price:.2f})")

async def main():
    """主函数"""
    print("🔥 高级实时日内交易监控系统")
    print("="*60)
    
    monitor = AdvancedIntradayMonitor(['AMD', 'NVDA', 'TSLA'])
    await monitor.run_demo(duration_minutes=3)  # 运行3分钟演示

if __name__ == "__main__":
    asyncio.run(main()) 