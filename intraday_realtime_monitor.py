#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时日内监控和交易系统
使用Yahoo Finance实时数据API
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

def load_portfolio_config():
    """从JSON配置文件加载持仓和观察仓信息"""
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取当前持仓股票 (排除港股和已卖出的股票)
        current_positions = []
        portfolio_info = {}
        
        for symbol, position in config.get('positions', {}).items():
            if (not symbol.endswith('.HK') and 
                position.get('shares', 0) > 0 and 
                position.get('status') != 'SOLD'):
                current_positions.append(symbol)
                portfolio_info[symbol] = {
                    'shares': position.get('shares', 0),
                    'cost_basis': position.get('cost_basis', 0),
                    'stop_loss_threshold': position.get('stop_loss_threshold', 0.08),
                    'sector': position.get('sector', 'Unknown')
                }
        
        # 提取观察仓股票
        watchlist_stocks = list(config.get('watchlist', {}).keys())
        
        # 完整监控列表
        all_stocks = current_positions + watchlist_stocks
        
        return {
            'current_positions': current_positions,
            'watchlist_stocks': watchlist_stocks,
            'all_stocks': all_stocks,
            'portfolio_info': portfolio_info
        }
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        # 返回默认配置
        return {
            'current_positions': ['AMD', 'NVDA'],
            'watchlist_stocks': ['MSFT', 'AAPL'],
            'all_stocks': ['AMD', 'NVDA', 'MSFT', 'AAPL'],
            'portfolio_info': {}
        }

class RealtimeIntradayMonitor:
    """实时日内监控系统"""
    
    def __init__(self, symbols: List[str] = None, update_interval: int = 60):
        """
        初始化实时监控系统
        
        Args:
            symbols: 监控的股票列表 (如果为None，则从配置文件加载)
            update_interval: 更新间隔(秒)
        """
        # 加载配置文件
        self.config = load_portfolio_config()
        
        # 如果没有指定股票列表，使用配置文件中的股票
        if symbols is None:
            self.symbols = self.config['all_stocks']
        else:
            self.symbols = symbols
            
        self.update_interval = update_interval
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 存储历史数据和信号
        self.price_history = {}
        self.alerts = []
        self.positions = self.config['portfolio_info']  # 从配置文件加载持仓信息
        self.running = False
        
        # 交易参数
        self.trading_params = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_threshold': 1.5,  # 成交量放大倍数
            'price_change_threshold': 2.0,  # 价格变动阈值(%)
            'stop_loss_pct': 2.0,  # 止损百分比
            'take_profit_pct': 3.0,  # 止盈百分比
        }
        
        print(f"🚀 实时日内监控系统初始化完成")
        print(f"📊 持仓股票: {self.config['current_positions']}")
        print(f"👀 观察仓股票: {self.config['watchlist_stocks']}")
        print(f"📈 总监控股票: {len(self.symbols)} 只")
        print(f"⏰ 更新间隔: {self.update_interval}秒")
    
    async def start_monitoring(self):
        """开始实时监控"""
        print(f"\n🔍 开始实时监控...")
        print(f"{'='*60}")
        
        self.running = True
        
        try:
            while self.running:
                await self._update_data()
                await self._analyze_signals()
                await self._check_alerts()
                
                # 等待下一次更新
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户停止监控")
        except Exception as e:
            print(f"\n❌ 监控过程出错: {e}")
        finally:
            self.running = False
            print(f"\n📊 监控已停止")
    
    async def _update_data(self):
        """更新实时数据"""
        try:
            # 获取实时数据
            realtime_data = await self.yahoo_source.get_realtime_data(
                self.symbols, timeframe='1m'
            )
            
            current_time = datetime.now()
            
            for symbol, df in realtime_data.items():
                if not df.empty:
                    # 存储最新数据
                    latest = df.iloc[-1]
                    
                    # 初始化历史数据
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    # 添加当前数据点
                    data_point = {
                        'timestamp': current_time,
                        'price': float(latest['close']),
                        'volume': int(latest['volume']) if latest['volume'] > 0 else 0,
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'open': float(latest['open'])
                    }
                    
                    self.price_history[symbol].append(data_point)
                    
                    # 保持最近100个数据点
                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]
            
            print(f"🔄 {current_time.strftime('%H:%M:%S')} 数据更新完成")
            
        except Exception as e:
            print(f"❌ 数据更新失败: {e}")
    
    async def _analyze_signals(self):
        """分析交易信号"""
        for symbol in self.symbols:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                continue
            
            try:
                signals = self._calculate_signals(symbol)
                await self._process_signals(symbol, signals)
                
            except Exception as e:
                print(f"❌ {symbol} 信号分析失败: {e}")
    
    def _calculate_signals(self, symbol: str) -> Dict:
        """计算技术指标和信号"""
        history = self.price_history[symbol]
        if len(history) < 14:
            return {}
        
        # 提取价格和成交量数据
        prices = [point['price'] for point in history]
        volumes = [point['volume'] for point in history]
        
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price
        
        # 计算RSI
        rsi = self._calculate_rsi(prices)
        
        # 计算移动平均线
        ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
        ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        
        # 计算成交量比率
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # 计算价格变动
        price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        # 生成信号
        signals = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'rsi': rsi,
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'volume_ratio': volume_ratio,
            'signals': []
        }
        
        # RSI信号
        if rsi > self.trading_params['rsi_overbought']:
            signals['signals'].append({
                'type': 'RSI_OVERBOUGHT',
                'message': f'RSI超买 {rsi:.1f}',
                'action': 'SELL',
                'strength': 'MEDIUM'
            })
        elif rsi < self.trading_params['rsi_oversold']:
            signals['signals'].append({
                'type': 'RSI_OVERSOLD',
                'message': f'RSI超卖 {rsi:.1f}',
                'action': 'BUY',
                'strength': 'MEDIUM'
            })
        
        # 均线信号
        if current_price > ma_5 > ma_10 > ma_20:
            signals['signals'].append({
                'type': 'MA_BULLISH',
                'message': '均线多头排列',
                'action': 'BUY',
                'strength': 'STRONG'
            })
        elif current_price < ma_5 < ma_10 < ma_20:
            signals['signals'].append({
                'type': 'MA_BEARISH',
                'message': '均线空头排列',
                'action': 'SELL',
                'strength': 'STRONG'
            })
        
        # 成交量信号
        if volume_ratio > self.trading_params['volume_threshold']:
            signals['signals'].append({
                'type': 'VOLUME_SPIKE',
                'message': f'成交量放大 {volume_ratio:.1f}倍',
                'action': 'WATCH',
                'strength': 'HIGH'
            })
        
        # 价格突破信号
        if abs(price_change_pct) > self.trading_params['price_change_threshold']:
            action = 'BUY' if price_change_pct > 0 else 'SELL'
            signals['signals'].append({
                'type': 'PRICE_BREAKOUT',
                'message': f'价格突破 {price_change_pct:+.1f}%',
                'action': action,
                'strength': 'HIGH'
            })
        
        return signals
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算RSI指标"""
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
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _process_signals(self, symbol: str, signals: Dict):
        """处理交易信号"""
        if not signals.get('signals'):
            return
        
        current_time = datetime.now()
        
        # 显示信号
        print(f"\n📊 {symbol} - {current_time.strftime('%H:%M:%S')}")
        print(f"   💰 价格: ${signals['current_price']:.2f} ({signals['price_change_pct']:+.2f}%)")
        print(f"   📈 RSI: {signals['rsi']:.1f}")
        print(f"   📦 成交量比: {signals['volume_ratio']:.1f}x")
        
        for signal in signals['signals']:
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'WATCH': '🟡'}.get(signal['action'], '⚪')
            print(f"   {emoji} {signal['message']} - {signal['action']}")
            
            # 生成交易建议
            if signal['strength'] in ['HIGH', 'STRONG']:
                await self._generate_trading_advice(symbol, signals, signal)
    
    async def _generate_trading_advice(self, symbol: str, signals: Dict, signal: Dict):
        """生成具体交易建议"""
        current_price = signals['current_price']
        action = signal['action']
        
        if action == 'BUY':
            entry_price = current_price
            stop_loss = current_price * (1 - self.trading_params['stop_loss_pct'] / 100)
            take_profit = current_price * (1 + self.trading_params['take_profit_pct'] / 100)
            
            advice = {
                'symbol': symbol,
                'action': 'BUY',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': signal['message'],
                'timestamp': datetime.now()
            }
            
            print(f"   💡 买入建议:")
            print(f"      📍 入场: ${entry_price:.2f}")
            print(f"      🛡️ 止损: ${stop_loss:.2f} (-{self.trading_params['stop_loss_pct']:.1f}%)")
            print(f"      🎯 止盈: ${take_profit:.2f} (+{self.trading_params['take_profit_pct']:.1f}%)")
            
        elif action == 'SELL':
            if symbol in self.positions:
                # 有持仓，建议卖出
                entry_price = self.positions[symbol]['entry_price']
                current_pnl = ((current_price - entry_price) / entry_price) * 100
                
                advice = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'exit_price': current_price,
                    'entry_price': entry_price,
                    'pnl_pct': current_pnl,
                    'reason': signal['message'],
                    'timestamp': datetime.now()
                }
                
                print(f"   💡 卖出建议:")
                print(f"      📍 出场: ${current_price:.2f}")
                print(f"      💰 盈亏: {current_pnl:+.2f}%")
            else:
                print(f"   💡 空仓建议: 等待更好入场时机")
    
    async def _check_alerts(self):
        """检查预警条件"""
        # 这里可以添加更多预警逻辑
        pass
    
    def add_position(self, symbol: str, entry_price: float, quantity: int = 100):
        """添加持仓"""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now()
        }
        print(f"✅ 添加持仓: {symbol} @ ${entry_price:.2f} x {quantity}")
    
    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"❌ 移除持仓: {symbol}")
    
    def get_current_status(self) -> Dict:
        """获取当前状态"""
        status = {
            'monitoring': self.running,
            'symbols': self.symbols,
            'positions': self.positions,
            'last_update': datetime.now(),
            'data_points': {symbol: len(history) for symbol, history in self.price_history.items()}
        }
        return status
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False

# 简单的使用示例
async def demo_realtime_monitoring():
    """演示实时监控功能"""
    print("🚀 启动实时日内监控演示")
    
    # 创建监控器
    monitor = RealtimeIntradayMonitor(
        symbols=['AMD', 'NVDA', 'TSLA'],
        update_interval=30  # 30秒更新一次
    )
    
    # 模拟添加一些持仓
    monitor.add_position('AMD', 135.00)
    monitor.add_position('NVDA', 145.00)
    
    try:
        # 运行5分钟的演示
        print(f"📊 开始5分钟监控演示...")
        
        # 创建监控任务
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # 5分钟后停止
        await asyncio.sleep(300)  # 5分钟
        monitor.stop_monitoring()
        
        # 等待监控任务完成
        await monitor_task
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 演示被用户中断")
        monitor.stop_monitoring()
    
    # 显示最终状态
    status = monitor.get_current_status()
    print(f"\n📊 最终状态:")
    print(f"   监控股票: {status['symbols']}")
    print(f"   数据点数: {status['data_points']}")
    print(f"   持仓情况: {list(status['positions'].keys())}")

if __name__ == "__main__":
    print("🔥 实时日内监控和交易系统")
    print("="*60)
    
    # 运行演示
    asyncio.run(demo_realtime_monitoring()) 