#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIG持仓跟踪器
用于监控AIG投资表现和风险管理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_interface import DataInterface
from datetime import datetime, timedelta
import pandas as pd

class AIGPositionTracker:
    """AIG持仓跟踪器"""
    
    def __init__(self):
        self.symbol = 'AIG'
        self.data_interface = DataInterface()
        
        # 持仓信息
        self.shares = 7
        self.entry_price = 84.32
        self.entry_date = '2025-06-10'
        self.position_pct = 2.13
        
        # 风险管理参数
        self.stop_loss = 82.20
        self.target_price_1 = 87.00  # 近期目标
        self.target_price_2 = 89.00  # 中期目标
        self.target_price_3 = 93.00  # 长期目标
        
    def get_current_status(self):
        """获取当前持仓状态"""
        # 获取最新数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        try:
            data = self.data_interface.get_historical_data(
                self.symbol, start_date, end_date
            )
            
            if data.empty:
                return None
                
            latest = data.iloc[-1]
            current_price = latest['close']
            
            # 计算收益
            total_value = current_price * self.shares
            entry_value = self.entry_price * self.shares
            pnl = total_value - entry_value
            pnl_pct = (current_price / self.entry_price - 1) * 100
            
            # 风险指标
            stop_loss_distance = (current_price - self.stop_loss) / current_price * 100
            target_1_distance = (self.target_price_1 - current_price) / current_price * 100
            
            status = {
                'datetime': data.index[-1],
                'current_price': current_price,
                'entry_price': self.entry_price,
                'shares': self.shares,
                'current_value': total_value,
                'entry_value': entry_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'stop_loss': self.stop_loss,
                'stop_loss_distance_pct': stop_loss_distance,
                'target_1': self.target_price_1,
                'target_1_distance_pct': target_1_distance,
                'volume': latest['volume'] if 'volume' in latest else 0
            }
            
            return status
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None
    
    def print_status_report(self):
        """打印状态报告"""
        status = self.get_current_status()
        
        if not status:
            print("❌ 无法获取当前状态")
            return
            
        print("=" * 60)
        print(f"🏷️  AIG持仓状态报告")
        print(f"📅 更新时间: {status['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print(f"\n💰 持仓信息:")
        print(f"   股数: {status['shares']} 股")
        print(f"   买入价格: ${status['entry_price']:.2f}")
        print(f"   当前价格: ${status['current_price']:.2f}")
        print(f"   持仓价值: ${status['current_value']:.2f}")
        
        print(f"\n📊 盈亏情况:")
        if status['pnl'] >= 0:
            print(f"   盈亏金额: +${status['pnl']:.2f} ✅")
            print(f"   盈亏比例: +{status['pnl_pct']:.2f}% ✅")
        else:
            print(f"   盈亏金额: ${status['pnl']:.2f} ❌")
            print(f"   盈亏比例: {status['pnl_pct']:.2f}% ❌")
        
        print(f"\n🎯 风险管理:")
        print(f"   止损价格: ${status['stop_loss']:.2f}")
        print(f"   止损距离: {status['stop_loss_distance_pct']:.1f}%")
        
        if status['stop_loss_distance_pct'] < 3:
            print("   ⚠️  警告: 接近止损位！")
        elif status['stop_loss_distance_pct'] < 5:
            print("   🟡 注意: 距离止损位较近")
        else:
            print("   ✅ 安全: 距离止损位较远")
            
        print(f"\n🚀 目标价格:")
        print(f"   近期目标: ${status['target_1']:.2f} (距离: {status['target_1_distance_pct']:.1f}%)")
        print(f"   中期目标: ${self.target_price_2:.2f}")
        print(f"   长期目标: ${self.target_price_3:.2f}")
        
        print(f"\n📈 市场信息:")
        print(f"   当日成交量: {status['volume']:,.0f}")
        
        # 操作建议
        print(f"\n💡 操作建议:")
        if status['current_price'] <= self.stop_loss:
            print("   🔴 建议止损出场")
        elif status['current_price'] >= self.target_price_1:
            print("   🟢 可考虑部分获利了结")
        elif status['pnl_pct'] > 2:
            print("   🟡 可考虑调整止损至成本价")
        else:
            print("   🔵 继续持有，关注止损位")
    
    def get_technical_signals(self):
        """获取技术指标信号"""
        try:
            # 获取更多历史数据用于技术分析
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            data = self.data_interface.get_historical_data(
                self.symbol, start_date, end_date
            )
            
            if data.empty:
                return None
                
            # 计算技术指标
            data['sma_5'] = data['close'].rolling(5).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['rsi'] = self._calculate_rsi(data['close'])
            
            latest = data.iloc[-1]
            
            signals = {
                'price': latest['close'],
                'sma_5': latest['sma_5'],
                'sma_20': latest['sma_20'],
                'rsi': latest['rsi'],
                'trend': 'up' if latest['close'] > latest['sma_20'] else 'down',
                'momentum': 'strong' if latest['rsi'] > 60 else 'weak' if latest['rsi'] < 40 else 'neutral'
            }
            
            return signals
            
        except Exception as e:
            print(f"获取技术指标失败: {e}")
            return None
    
    def _calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def export_to_csv(self, filename=None):
        """导出状态到CSV"""
        if filename is None:
            filename = f"aig_position_{datetime.now().strftime('%Y%m%d')}.csv"
            
        status = self.get_current_status()
        if not status:
            print("无法导出，获取数据失败")
            return
            
        df = pd.DataFrame([status])
        df.to_csv(filename, index=False)
        print(f"状态已导出到: {filename}")

def main():
    """主函数"""
    tracker = AIGPositionTracker()
    
    print("AIG持仓跟踪器启动...")
    tracker.print_status_report()
    
    # 获取技术指标
    signals = tracker.get_technical_signals()
    if signals:
        print(f"\n📊 技术指标:")
        print(f"   5日均线: ${signals['sma_5']:.2f}")
        print(f"   20日均线: ${signals['sma_20']:.2f}")
        print(f"   RSI: {signals['rsi']:.1f}")
        print(f"   趋势: {signals['trend']}")
        print(f"   动量: {signals['momentum']}")

if __name__ == "__main__":
    main() 