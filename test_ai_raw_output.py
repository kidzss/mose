"""
测试AI原文输出
验证AI分析器是否正确返回和显示原文
"""

import asyncio
import json
from datetime import datetime
from ai_realtime_analyzer import AIRealtimeAnalyzer
from ai_trading_module import AITradingModule

async def test_ai_raw_output():
    """测试AI原文输出"""
    print("🤖 测试AI原文输出")
    print("=" * 50)
    
    # 初始化AI分析器
    analyzer = AIRealtimeAnalyzer()
    ai_module = AITradingModule()
    
    # 测试数据
    test_symbol = "NVDA"
    test_data = {
        'current_price': 155.02,
        'change_pct': 2.5,
        'volume': 50000000,
        'volume_ratio': 1.2,
        'rsi': 65,
        'macd': 'bullish',
        'bollinger_position': 'middle_band'
    }
    
    print(f"📊 测试股票: {test_symbol}")
    print(f"📈 测试数据: {json.dumps(test_data, indent=2)}")
    print()
    
    # 测试1: 直接AI分析器
    print("🔍 测试1: 直接AI分析器")
    print("-" * 30)
    
    try:
        result = await analyzer.analyze_market_event(
            test_symbol, 
            "technical_signal", 
            test_data, 
            "comprehensive"
        )
        
        if result.get('success'):
            print("✅ AI分析成功")
            print()
            
            # 显示结构化建议
            action_suggestion = result.get('action_suggestion', {})
            print("🎯 结构化建议:")
            print(f"   建议操作: {action_suggestion.get('action', '不明确')}")
            print(f"   简单理由: {action_suggestion.get('reason', '无')}")
            print(f"   风险提醒: {action_suggestion.get('risk_warning', '无')}")
            print()
            
            # 显示AI原文
            ai_text = result.get('ai_analysis', '')
            if ai_text:
                print("🤖 AI原文分析:")
                print("=" * 40)
                print(ai_text)
                print("=" * 40)
            else:
                print("❌ 未找到AI原文")
        else:
            print(f"❌ AI分析失败: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
    
    print()
    print("=" * 50)
    
    # 测试2: AI交易模块
    print("🔍 测试2: AI交易模块")
    print("-" * 30)
    
    try:
        result = await ai_module.analyze_stock_signal(
            test_symbol,
            test_data,
            "comprehensive"
        )
        
        if result.get('success'):
            print("✅ AI交易模块分析成功")
            print()
            
            # 显示结构化建议
            action_suggestion = result.get('action_suggestion', {})
            print("🎯 结构化建议:")
            print(f"   建议操作: {action_suggestion.get('action', '不明确')}")
            print(f"   简单理由: {action_suggestion.get('reason', '无')}")
            print(f"   风险提醒: {action_suggestion.get('risk_warning', '无')}")
            print()
            
            # 显示AI原文
            ai_text = result.get('ai_analysis', '')
            if ai_text:
                print("🤖 AI原文分析:")
                print("=" * 40)
                print(ai_text)
                print("=" * 40)
            else:
                print("❌ 未找到AI原文")
        else:
            print(f"❌ AI交易模块分析失败: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
    
    print()
    print("=" * 50)
    
    # 测试3: 持仓感知分析
    print("🔍 测试3: 持仓感知分析")
    print("-" * 30)
    
    # 模拟持仓信息
    position_info = {
        'shares': 100,
        'cost_basis': 140.0,
        'weight': 15.5,
        'sector': 'Technology'
    }
    
    try:
        result = await ai_module.analyze_portfolio_position(
            test_symbol,
            test_data,
            position_info
        )
        
        if result.get('success'):
            print("✅ 持仓感知分析成功")
            print()
            
            # 显示结构化建议
            action_suggestion = result.get('action_suggestion', {})
            print("🎯 结构化建议:")
            print(f"   建议操作: {action_suggestion.get('action', '不明确')}")
            print(f"   简单理由: {action_suggestion.get('reason', '无')}")
            print(f"   风险提醒: {action_suggestion.get('risk_warning', '无')}")
            print()
            
            # 显示AI原文
            ai_text = result.get('ai_analysis', '')
            if ai_text:
                print("🤖 AI原文分析:")
                print("=" * 40)
                print(ai_text)
                print("=" * 40)
            else:
                print("❌ 未找到AI原文")
            
            # 显示多时间框架分析
            if 'multi_timeframe_analysis' in result:
                multi_result = result['multi_timeframe_analysis']
                if multi_result.get('success'):
                    print("⏰ 多时间框架分析:")
                    multi_ai_text = multi_result.get('ai_analysis', '')
                    if multi_ai_text:
                        print("=" * 40)
                        print(multi_ai_text)
                        print("=" * 40)
        else:
            print(f"❌ 持仓感知分析失败: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
    
    print()
    print("=" * 50)
    
    # 显示分析历史
    print("📊 分析历史摘要")
    print("-" * 30)
    
    summary = ai_module.get_analysis_summary()
    print(f"总分析次数: {summary.get('total_analyses', 0)}")
    print(f"成功率: {summary.get('success_rate', 0):.2%}")
    
    recent_analyses = ai_module.get_recent_analysis(limit=3)
    print(f"最近分析记录数: {len(recent_analyses)}")
    
    for i, analysis in enumerate(recent_analyses, 1):
        if analysis.get('success'):
            symbol = analysis.get('symbol', 'Unknown')
            action = analysis.get('action_suggestion', {}).get('action', '不明确')
            has_ai_text = bool(analysis.get('ai_analysis', ''))
            print(f"  {i}. {symbol} - {action} - 有原文: {'✅' if has_ai_text else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_ai_raw_output()) 