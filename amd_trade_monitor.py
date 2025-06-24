#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD交易实时监控系统
用户交易：5股 @ $136.90
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

class AMDTradeMonitor:
    """AMD交易监控系统"""
    
    def __init__(self):
        """初始化监控系统"""
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 交易信息
        self.symbol = 'AMD'
        self.entry_price = 136.90
        self.quantity = 5
        self.entry_time = datetime.now()
        
        # 预设的卖出目标 (基于之前的预测调整)
        self.targets = {
            'quick_profit': {'price': 137.31, 'probability': 0.7, 'description': '快速获利 (+0.30%)'},
            'target_profit': {'price': 137.72, 'probability': 0.4, 'description': '目标获利 (+0.60%)'},
            'stop_loss': {'price': 136.63, 'probability': 0.3, 'description': '止损 (-0.20%)'}
        }
        
        print("🎯 AMD交易实时监控系统启动")
        print("="*60)
        print(f"📊 交易详情:")
        print(f"   股票: {self.symbol}")
        print(f"   数量: {self.quantity}股")
        print(f"   入场价: ${self.entry_price:.2f}")
        print(f"   入场时间: {self.entry_time.strftime('%H:%M:%S')}")
        print(f"   总投资: ${self.entry_price * self.quantity:.2f}")
        
        print(f"\n🎯 预设目标:")
        for target_name, target_info in self.targets.items():
            price = target_info['price']
            prob = target_info['probability']
            desc = target_info['description']
            profit = (price - self.entry_price) * self.quantity
            print(f"   • {desc}: ${price:.2f} (盈亏: ${profit:+.2f})")
    
    async def start_monitoring(self, duration_minutes: int = 30):
        """开始实时监控"""
        print(f"\n🔄 开始{duration_minutes}分钟实时监控...")
        print("="*60)
        
        updates = duration_minutes * 2  # 每30秒更新一次
        
        for i in range(updates):
            try:
                await self._update_and_analyze()
                
                if i < updates - 1:
                    await asyncio.sleep(30)  # 等待30秒
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ 用户停止监控")
                break
            except Exception as e:
                print(f"❌ 监控更新失败: {e}")
        
        print(f"\n📊 监控结束")
        await self._final_summary()
    
    async def _update_and_analyze(self):
        """更新数据并分析"""
        current_time = datetime.now()
        
        # 获取实时数据
        try:
            realtime_data = await self.yahoo_source.get_realtime_data([self.symbol], timeframe='1m')
            
            if self.symbol in realtime_data and not realtime_data[self.symbol].empty:
                df = realtime_data[self.symbol]
                latest = df.iloc[-1]
                
                current_price = float(latest['close'])
                current_volume = int(latest['volume']) if latest['volume'] > 0 else 0
                
                # 计算盈亏
                price_change = current_price - self.entry_price
                price_change_pct = (price_change / self.entry_price) * 100
                position_pnl = price_change * self.quantity
                
                # 计算持有时间
                holding_time = current_time - self.entry_time
                holding_minutes = int(holding_time.total_seconds() / 60)
                
                # 显示当前状态
                emoji = "🟢" if position_pnl > 0 else "🔴" if position_pnl < 0 else "⚪"
                
                print(f"\n{emoji} {current_time.strftime('%H:%M:%S')} - AMD实时状态")
                print(f"   💰 当前价格: ${current_price:.2f}")
                print(f"   📈 价格变动: {price_change_pct:+.2f}% (${price_change:+.2f})")
                print(f"   💵 持仓盈亏: ${position_pnl:+.2f}")
                print(f"   ⏰ 持有时间: {holding_minutes}分钟")
                print(f"   📦 当前成交量: {current_volume:,}")
                
                # 检查目标触发
                self._check_targets(current_price, position_pnl)
                
                # 技术分析
                await self._quick_technical_analysis(df, current_price)
                
            else:
                print(f"❌ 无法获取{self.symbol}实时数据")
                
        except Exception as e:
            print(f"❌ 数据更新失败: {e}")
    
    def _check_targets(self, current_price: float, position_pnl: float):
        """检查目标触发情况"""
        triggered = []
        
        for target_name, target_info in self.targets.items():
            target_price = target_info['price']
            description = target_info['description']
            
            if target_name == 'stop_loss':
                if current_price <= target_price:
                    triggered.append(f"🛡️ {description} 触发! 建议止损卖出")
            else:
                if current_price >= target_price:
                    triggered.append(f"🎯 {description} 触发! 可考虑获利了结")
        
        if triggered:
            print(f"   ⚠️ 目标触发:")
            for trigger in triggered:
                print(f"      {trigger}")
        
        # 风险提醒
        if position_pnl < -10:  # 损失超过$10
            print(f"   ⚠️ 风险提醒: 损失已达${abs(position_pnl):.2f}，请考虑止损")
        elif position_pnl > 5:  # 盈利超过$5
            print(f"   💡 获利提醒: 已盈利${position_pnl:.2f}，可考虑部分获利了结")
    
    async def _quick_technical_analysis(self, df, current_price: float):
        """快速技术分析"""
        try:
            closes = df['close'].values
            volumes = df['volume'].values
            
            if len(closes) >= 5:
                # RSI
                rsi = self._calculate_rsi(closes)
                
                # 移动平均线
                ma_5 = np.mean(closes[-5:])
                
                # 成交量比较
                if len(volumes) >= 10:
                    avg_volume = np.mean(volumes[-10:-1])  # 排除当前成交量
                    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
                else:
                    volume_ratio = 1
                
                # 价格动量
                if len(closes) >= 3:
                    momentum = ((current_price - closes[-3]) / closes[-3]) * 100
                else:
                    momentum = 0
                
                print(f"   📊 技术指标:")
                print(f"      RSI: {rsi:.1f} {'(超买)' if rsi > 70 else '(超卖)' if rsi < 30 else '(中性)'}")
                print(f"      MA5: ${ma_5:.2f} {'(上方)' if current_price > ma_5 else '(下方)'}")
                print(f"      成交量比: {volume_ratio:.1f}x")
                print(f"      短期动量: {momentum:+.2f}%")
                
                # 简单建议
                signals = []
                if rsi > 70:
                    signals.append("RSI超买，注意回调风险")
                elif rsi < 30:
                    signals.append("RSI超卖，可能反弹")
                
                if current_price > ma_5:
                    signals.append("价格在MA5上方，短期偏强")
                else:
                    signals.append("价格在MA5下方，短期偏弱")
                
                if volume_ratio > 1.5:
                    signals.append("成交量放大，关注突破")
                
                if signals:
                    print(f"   💡 技术信号:")
                    for signal in signals:
                        print(f"      • {signal}")
                        
        except Exception as e:
            print(f"   ❌ 技术分析失败: {e}")
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            period = max(2, len(prices) - 1)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    async def _final_summary(self):
        """最终总结"""
        try:
            # 获取最终价格
            realtime_data = await self.yahoo_source.get_realtime_data([self.symbol], timeframe='1m')
            
            if self.symbol in realtime_data and not realtime_data[self.symbol].empty:
                df = realtime_data[self.symbol]
                final_price = float(df.iloc[-1]['close'])
                
                final_pnl = (final_price - self.entry_price) * self.quantity
                final_pnl_pct = ((final_price - self.entry_price) / self.entry_price) * 100
                
                total_time = datetime.now() - self.entry_time
                total_minutes = int(total_time.total_seconds() / 60)
                
                print(f"\n📋 交易总结:")
                print(f"   📊 入场: ${self.entry_price:.2f} x {self.quantity}股")
                print(f"   📊 现价: ${final_price:.2f}")
                print(f"   💰 总盈亏: ${final_pnl:+.2f} ({final_pnl_pct:+.2f}%)")
                print(f"   ⏰ 持有时间: {total_minutes}分钟")
                
                # 建议
                if final_pnl > 0:
                    print(f"   🎉 恭喜盈利! 可考虑获利了结或继续持有")
                elif final_pnl < -5:
                    print(f"   ⚠️ 注意风险! 建议考虑止损")
                else:
                    print(f"   📊 盈亏较小，可继续观察")
                    
        except Exception as e:
            print(f"❌ 最终总结失败: {e}")

async def main():
    """主函数"""
    print("🎯 AMD交易监控启动")
    
    monitor = AMDTradeMonitor()
    
    try:
        await monitor.start_monitoring(duration_minutes=10)  # 监控10分钟
    except KeyboardInterrupt:
        print(f"\n⏹️ 监控被用户中断")

if __name__ == "__main__":
    asyncio.run(main()) 