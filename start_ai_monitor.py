#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI增强监控系统启动脚本
快速启动AI分析模块
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trading_module import AITradingModule

class SimpleAIMonitor:
    """简单的AI监控系统"""
    
    def __init__(self):
        """初始化"""
        self.ai_module = AITradingModule()
        print("🤖 AI监控系统初始化完成")
    
    async def run_monitor(self):
        """运行监控"""
        print("🚀 启动AI监控系统...")
        print("按 Ctrl+C 停止监控")
        
        # 模拟股票数据
        stocks = [
            {"symbol": "NVDA", "price": 155.02, "change": 2.5, "rsi": 65},
            {"symbol": "AMD", "price": 59.19, "change": -1.2, "rsi": 75},
            {"symbol": "TSLA", "price": 296.50, "change": 3.8, "rsi": 70},
            {"symbol": "AAPL", "price": 198.45, "change": 1.5, "rsi": 60},
            {"symbol": "MSFT", "price": 415.20, "change": -0.8, "rsi": 55}
        ]
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - AI监控运行中...")
                
                for stock in stocks:
                    await self._analyze_stock(stock)
                
                print("\n" + "="*50)
                await asyncio.sleep(120)  # 每2分钟分析一次
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户停止监控")
        except Exception as e:
            print(f"❌ 监控出错: {e}")
    
    async def _analyze_stock(self, stock):
        """分析单个股票"""
        symbol = stock["symbol"]
        price = stock["price"]
        change = stock["change"]
        rsi = stock["rsi"]
        
        print(f"\n📊 分析 {symbol}...")
        print(f"  价格: ${price:.2f} ({change:+.1f}%)")
        print(f"  RSI: {rsi}")
        
        # 构建信号数据
        signal_data = {
            "current_price": price,
            "change_pct": change,
            "rsi": rsi
        }
        
        # 根据变化幅度选择分析类型
        if abs(change) > 3:
            analysis_type = "comprehensive"
        elif abs(change) > 1:
            analysis_type = "detailed"
        else:
            analysis_type = "quick"
        
        try:
            # 调用AI分析
            result = await self.ai_module.analyze_stock_signal(
                symbol, signal_data, analysis_type
            )
            
            if result.get('success'):
                action_suggestion = result.get('action_suggestion', {})
                action = action_suggestion.get('action', '不明确')
                reason = action_suggestion.get('reason', '无')
                risk = action_suggestion.get('risk_warning', '无')
                
                print(f"  🤖 AI建议: {action}")
                print(f"  📝 理由: {reason}")
                print(f"  ⚠️ 风险: {risk}")
                
                # 显示操作建议
                if action in ['止损', '止盈', '减仓']:
                    print(f"  🚨 需要关注: {action}")
                elif action == '加仓':
                    print(f"  📈 机会信号: {action}")
                else:
                    print(f"  👀 保持观望")
                    
            else:
                print(f"  ❌ AI分析失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"  ❌ 分析出错: {e}")
    
    def show_summary(self):
        """显示分析摘要"""
        summary = self.ai_module.get_analysis_summary()
        alerts = self.ai_module.get_alerts()
        
        print(f"\n📋 分析摘要:")
        print(f"  总分析次数: {summary.get('total_analyses', 0)}")
        print(f"  成功率: {summary.get('success_rate', 0):.1%}")
        print(f"  警报数量: {len(alerts)}")
        
        if alerts:
            print(f"  最新警报:")
            for alert in alerts[-3:]:
                print(f"    - {alert['symbol']}: {alert['action']} - {alert['reason']}")

async def main():
    """主函数"""
    print("🤖 AI增强监控系统")
    print("=" * 50)
    
    # 创建监控系统
    monitor = SimpleAIMonitor()
    
    # 运行监控
    await monitor.run_monitor()
    
    # 显示摘要
    monitor.show_summary()
    
    print("\n✅ 监控结束")

if __name__ == "__main__":
    asyncio.run(main()) 