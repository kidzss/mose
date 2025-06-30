#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强AI分析器 - 使用每日持股分析结果
Test Enhanced AI Analyzer - Using Daily Holdings Analysis Results
"""

import asyncio
import json
from datetime import datetime
from ai_realtime_analyzer import AIRealtimeAnalyzer

async def test_enhanced_ai_analyzer():
    """测试增强AI分析器"""
    print("🧪 测试增强AI分析器 - 使用每日持股分析结果")
    print("=" * 60)
    
    # 初始化AI分析器
    analyzer = AIRealtimeAnalyzer(
        use_daily_analysis=True,
        default_model="deepseek-r1:latest"
    )
    
    # 测试股票
    test_symbols = ['AMD', 'NVDA', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n📊 分析 {symbol}...")
        
        # 模拟市场数据
        market_data = {
            'current_price': 150.0,
            'change_pct': 2.5,
            'volume': 1000000,
            'rsi': 65.0,
            'macd': 'positive',
            'bollinger_position': 'upper',
            'volume_ratio': 1.2
        }
        
        # 添加持仓信息（如果存在）
        try:
            with open('portfolio_config.json', 'r', encoding='utf-8') as f:
                portfolio_config = json.load(f)
                if symbol in portfolio_config.get('positions', {}):
                    position = portfolio_config['positions'][symbol]
                    market_data['position_info'] = {
                        'shares': position.get('shares', 0),
                        'cost_basis': position.get('cost_basis', 0),
                        'weight': position.get('weight', 0),
                        'sector': position.get('sector', 'Unknown')
                    }
                    print(f"   📈 找到持仓信息: {position.get('shares', 0)}股, 成本${position.get('cost_basis', 0):.2f}")
        except Exception as e:
            print(f"   ⚠️ 无法读取持仓信息: {e}")
        
        # 执行AI分析
        try:
            result = await analyzer.analyze_market_event(
                symbol=symbol,
                event_type="portfolio_position",
                market_data=market_data,
                analysis_type="comprehensive"
            )
            
            if result['success']:
                print(f"   ✅ AI分析成功")
                print(f"   🤖 模型: {result['model_used']}")
                print(f"   📝 建议: {result['action_suggestion'].get('action', 'N/A')}")
                print(f"   📊 原始分析:")
                print(f"   {'='*40}")
                print(result['ai_analysis'])
                print(f"   {'='*40}")
                
                # 显示每日分析数据摘要
                if 'daily_analysis' in result and result['daily_analysis']:
                    daily = result['daily_analysis']
                    if 'market_analysis' in daily:
                        market = daily['market_analysis']
                        print(f"   📈 市场环境: {market.get('vix_analysis', 'N/A')}")
                    if 'symbol_analysis' in daily:
                        symbol_analysis = daily['symbol_analysis']
                        if symbol_analysis:
                            print(f"   📊 {symbol}技术指标: RSI={symbol_analysis.get('rsi', 0):.1f}, 52周位置={symbol_analysis.get('position_52w', 0):.1f}%")
            else:
                print(f"   ❌ AI分析失败: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ 分析过程出错: {e}")
        
        print(f"   {'-'*40}")
    
    # 显示分析历史摘要
    print(f"\n📋 分析历史摘要:")
    history = analyzer.get_analysis_history(limit=5)
    for i, record in enumerate(history, 1):
        print(f"   {i}. {record['symbol']}: {record['action_suggestion'].get('action', 'N/A')} - {record['timestamp']}")

async def test_simple_analysis():
    """测试简单分析"""
    print(f"\n🧪 测试简单分析模式")
    print("=" * 40)
    
    analyzer = AIRealtimeAnalyzer(use_daily_analysis=False)
    
    # 简单价格分析
    result = await analyzer.analyze_price_alert(
        symbol="AAPL",
        current_price=180.0,
        change_pct=1.5
    )
    
    if result['success']:
        print(f"✅ 简单分析成功")
        print(f"📝 建议: {result['action_suggestion']}")
        print(f"📊 分析内容:")
        print(result['ai_analysis'])
    else:
        print(f"❌ 简单分析失败: {result.get('error', 'Unknown error')}")

def main():
    """主函数"""
    print("🚀 开始测试增强AI分析器")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    asyncio.run(test_enhanced_ai_analyzer())
    asyncio.run(test_simple_analysis())
    
    print(f"\n✅ 测试完成!")

if __name__ == "__main__":
    main() 