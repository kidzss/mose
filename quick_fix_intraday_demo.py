#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版实时日内监控演示
降低数据要求，快速展示效果
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

class QuickIntradayDemo:
    """快速日内监控演示"""
    
    def __init__(self, symbols: List[str] = None):
        """初始化"""
        self.symbols = symbols or ['AMD', 'NVDA', 'TSLA']
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 数据存储 (降低要求)
        self.price_history = {symbol: deque(maxlen=20) for symbol in self.symbols}
        
        # 简化参数 (降低数据要求)
        self.params = {
            'min_data_points': 3,        # 最少3个数据点就开始分析
            'rsi_period': 5,             # RSI周期降低到5
            'ma_short': 3,               # 短期均线3期
            'ma_long': 5,                # 长期均线5期
            'volume_threshold': 1.2,     # 成交量阈值
            'price_threshold': 0.5,      # 价格变动阈值
        }
        
        # 模拟持仓
        self.positions = {}
        
        print("🚀 快速日内监控演示系统")
        print(f"📊 监控股票: {self.symbols}")
        print(f"⚡ 最少数据要求: {self.params['min_data_points']}个点")
    
    async def run_quick_demo(self, duration_minutes: int = 2):
        """运行快速演示"""
        print(f"\n🎯 开始{duration_minutes}分钟快速演示")
        print("="*60)
        
        # 添加模拟持仓
        self.add_position('AMD', 137.00, 100)
        self.add_position('NVDA', 147.00, 50)
        
        updates = duration_minutes * 2  # 每30秒更新一次
        
        for i in range(updates):
            try:
                await self._quick_update()
                
                if i < updates - 1:
                    print(f"\n⏳ 等待30秒后进行下一次更新...")
                    await asyncio.sleep(30)
                    
            except Exception as e:
                print(f"❌ 更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ 快速演示完成!")
        self._show_summary()
    
    async def _quick_update(self):
        """快速更新"""
        current_time = datetime.now()
        print(f"\n🔄 {current_time.strftime('%H:%M:%S')} - 快速数据更新")
        print("-" * 50)
        
        # 获取实时数据
        try:
            realtime_data = await self.yahoo_source.get_realtime_data(
                self.symbols, timeframe='1m'
            )
            
            for symbol in self.symbols:
                if symbol in realtime_data and not realtime_data[symbol].empty:
                    df = realtime_data[symbol]
                    latest = df.iloc[-1]
                    
                    # 存储价格数据
                    price_data = {
                        'timestamp': current_time,
                        'open': float(latest['open']),
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'close': float(latest['close']),
                        'volume': int(latest['volume']) if latest['volume'] > 0 else 1
                    }
                    
                    self.price_history[symbol].append(price_data)
                    
                    # 立即分析 (降低要求)
                    await self._quick_analyze(symbol)
                else:
                    print(f"❌ {symbol}: 无法获取数据")
                    
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
    
    async def _quick_analyze(self, symbol: str):
        """快速分析 (降低数据要求)"""
        history = list(self.price_history[symbol])
        
        if len(history) < self.params['min_data_points']:
            print(f"📊 {symbol}: 需要{self.params['min_data_points']}个数据点，当前{len(history)}个")
            return
        
        print(f"📊 {symbol} 快速分析 (数据点: {len(history)}):")
        
        # 基础数据
        current_data = history[-1]
        current_price = current_data['close']
        
        # 计算简化指标
        closes = [item['close'] for item in history]
        volumes = [item['volume'] for item in history]
        
        # 价格变动
        if len(closes) > 1:
            prev_price = closes[-2]
            price_change = ((current_price - prev_price) / prev_price * 100)
        else:
            price_change = 0
        
        # 简化RSI (如果数据足够)
        if len(closes) >= self.params['rsi_period']:
            rsi = self._calculate_simple_rsi(closes)
        else:
            rsi = 50  # 默认中性
        
        # 移动平均线
        ma_short = np.mean(closes[-self.params['ma_short']:]) if len(closes) >= self.params['ma_short'] else current_price
        ma_long = np.mean(closes[-self.params['ma_long']:]) if len(closes) >= self.params['ma_long'] else current_price
        
        # 成交量比率
        if len(volumes) >= 3:
            avg_volume = np.mean(volumes[:-1])  # 排除当前成交量
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1
        
        # 显示分析结果
        print(f"   💰 当前价格: ${current_price:.2f} ({price_change:+.2f}%)")
        print(f"   📈 RSI: {rsi:.1f}")
        print(f"   📊 MA{self.params['ma_short']}: ${ma_short:.2f} | MA{self.params['ma_long']}: ${ma_long:.2f}")
        print(f"   📦 成交量比: {volume_ratio:.1f}x")
        
        # 显示持仓盈亏
        if symbol in self.positions:
            entry_price = self.positions[symbol]['entry_price']
            current_pnl = ((current_price - entry_price) / entry_price) * 100
            emoji = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "⚪"
            print(f"   {emoji} 持仓盈亏: {current_pnl:+.2f}% (入场: ${entry_price:.2f})")
        
        # 生成简单信号
        signals = self._generate_simple_signals(current_price, price_change, rsi, ma_short, ma_long, volume_ratio)
        
        if signals:
            print(f"   🎯 交易信号:")
            for signal in signals:
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'WATCH': '🟡'}.get(signal['action'], '⚪')
                print(f"      {emoji} {signal['message']}")
        else:
            print(f"   ⚪ 无明显信号")
    
    def _calculate_simple_rsi(self, prices: List[float]) -> float:
        """计算简化RSI"""
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
        return 100 - (100 / (1 + rs))
    
    def _generate_simple_signals(self, current_price: float, price_change: float, 
                                rsi: float, ma_short: float, ma_long: float, volume_ratio: float) -> List[Dict]:
        """生成简单信号"""
        signals = []
        
        # RSI信号
        if rsi > 70:
            signals.append({
                'action': 'SELL',
                'message': f'RSI超买 {rsi:.1f}'
            })
        elif rsi < 30:
            signals.append({
                'action': 'BUY',
                'message': f'RSI超卖 {rsi:.1f}'
            })
        
        # 均线信号
        if current_price > ma_short > ma_long:
            signals.append({
                'action': 'BUY',
                'message': '均线多头排列'
            })
        elif current_price < ma_short < ma_long:
            signals.append({
                'action': 'SELL',
                'message': '均线空头排列'
            })
        
        # 成交量信号
        if volume_ratio > self.params['volume_threshold']:
            signals.append({
                'action': 'WATCH',
                'message': f'成交量放大 {volume_ratio:.1f}x'
            })
        
        # 价格突破信号
        if abs(price_change) > self.params['price_threshold']:
            action = 'BUY' if price_change > 0 else 'SELL'
            signals.append({
                'action': action,
                'message': f'价格突破 {price_change:+.2f}%'
            })
        
        return signals
    
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
        print(f"\n📋 演示总结:")
        print(f"   📊 监控股票: {len(self.symbols)}只")
        print(f"   📈 数据点数: {[len(self.price_history[s]) for s in self.symbols]}")
        
        if self.positions:
            print(f"   💰 持仓情况:")
            for symbol, pos in self.positions.items():
                if len(self.price_history[symbol]) > 0:
                    current_price = self.price_history[symbol][-1]['close']
                    pnl = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    print(f"      {emoji} {symbol}: {pnl:+.2f}% (${pos['entry_price']:.2f} → ${current_price:.2f})")

async def main():
    """主函数"""
    print("⚡ 快速日内监控演示")
    print("="*50)
    
    demo = QuickIntradayDemo(['AMD', 'NVDA', 'TSLA'])
    await demo.run_quick_demo(duration_minutes=2)

if __name__ == "__main__":
    asyncio.run(main()) 