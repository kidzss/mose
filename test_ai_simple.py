#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化AI测试脚本
Simple AI Test Script
"""

import asyncio
from datetime import datetime
from ai_realtime_analyzer import AIRealtimeAnalyzer

async def test_simple_ai():
    """测试简化AI分析"""
    print("🧪 测试简化AI分析")
    print("=" * 40)
    
    # 初始化AI分析器
    analyzer = AIRealtimeAnalyzer(
        use_daily_analysis=False,  # 先不使用每日分析，简化测试
        default_model="deepseek-r1:latest"
    )
    
    # 简单的市场数据
    market_data = {
        'current_price': 150.0,
        'change_pct': 2.5,
        'volume': 1000000,
        'rsi': 65.0
    }
    
    print("📊 测试股票: AMD")
    print(f"   价格: ${market_data['current_price']:.2f}")
    print(f"   涨跌幅: {market_data['change_pct']:+.2f}%")
    print(f"   RSI: {market_data['rsi']:.1f}")
    
    try:
        print(f"\n🤖 开始AI分析...")
        result = await analyzer.analyze_market_event(
            symbol="AMD",
            event_type="price_alert",
            market_data=market_data,
            analysis_type="quick"
        )
        
        if result['success']:
            print(f"✅ AI分析成功!")
            print(f"📝 操作建议: {result['action_suggestion'].get('action', 'N/A')}")
            print(f"🤖 使用模型: {result['model_used']}")
            
            # 显示分析内容
            ai_text = result['ai_analysis']
            print(f"\n📊 AI分析内容:")
            print(f"{'='*50}")
            print(ai_text)
            print(f"{'='*50}")
            
        else:
            print(f"❌ AI分析失败: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print(f"\n✅ 测试完成!")

def main():
    """主函数"""
    print("🚀 简化AI测试")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    asyncio.run(test_simple_ai())

if __name__ == "__main__":
    main() 