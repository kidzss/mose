#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI每日持股分析系统
Test AI Daily Holdings Analysis System
"""

import asyncio
from datetime import datetime
from ai_realtime_analyzer import AIRealtimeAnalyzer
from daily_holdings_analysis import DailyHoldingsAnalyzer

async def test_ai_daily_system():
    """测试AI每日持股分析系统"""
    print("🧪 测试AI每日持股分析系统")
    print("=" * 50)
    
    # 初始化系统
    print("📊 初始化每日持股分析器...")
    daily_analyzer = DailyHoldingsAnalyzer()
    
    print("🤖 初始化AI分析器...")
    ai_analyzer = AIRealtimeAnalyzer(use_daily_analysis=True)
    
    # 测试股票
    test_symbol = "AMD"
    
    print(f"\n📈 测试股票: {test_symbol}")
    
    # 获取每日分析数据
    print("📊 获取每日持股分析数据...")
    try:
        all_symbols = list(daily_analyzer.portfolio.keys()) + daily_analyzer.market_indices + daily_analyzer.watchlist
        all_symbols = list(set(all_symbols))
        
        data = daily_analyzer.get_today_data(all_symbols)
        
        if data and test_symbol in data:
            print(f"✅ 成功获取 {test_symbol} 数据")
            stock_data = data[test_symbol]
            print(f"   价格: ${stock_data['price']:.2f}")
            print(f"   涨跌幅: {stock_data['change_pct']:+.2f}%")
            print(f"   RSI: {stock_data['rsi']:.1f}")
            
            # 模拟市场数据
            market_data = {
                'current_price': stock_data['price'],
                'change_pct': stock_data['change_pct'],
                'volume': stock_data['volume'],
                'rsi': stock_data['rsi'],
                'volume_ratio': stock_data['volume_ratio']
            }
            
            # 添加持仓信息
            if test_symbol in daily_analyzer.portfolio:
                position = daily_analyzer.portfolio[test_symbol]
                market_data['position_info'] = {
                    'shares': position['shares'],
                    'cost_basis': position['cost'],
                    'weight': 0,  # 简化处理
                    'sector': 'Technology'
                }
                print(f"   持仓: {position['shares']}股 @ ${position['cost']:.2f}")
            
            # AI分析
            print(f"\n🤖 进行AI分析...")
            result = await ai_analyzer.analyze_market_event(
                symbol=test_symbol,
                event_type="portfolio_position",
                market_data=market_data,
                analysis_type="comprehensive"
            )
            
            if result['success']:
                print(f"✅ AI分析成功!")
                print(f"📝 操作建议: {result['action_suggestion'].get('action', 'N/A')}")
                print(f"🤖 使用模型: {result['model_used']}")
                
                # 显示分析摘要
                ai_text = result['ai_analysis']
                if len(ai_text) > 200:
                    print(f"📊 分析内容摘要:")
                    print(f"   {ai_text[:200]}...")
                else:
                    print(f"📊 分析内容:")
                    print(f"   {ai_text}")
                
                # 检查每日分析数据
                if 'daily_analysis' in result and result['daily_analysis']:
                    daily = result['daily_analysis']
                    print(f"\n📈 每日分析数据:")
                    if 'market_analysis' in daily:
                        market = daily['market_analysis']
                        print(f"   市场环境: {market.get('vix_analysis', 'N/A')}")
                    if 'symbol_analysis' in daily:
                        symbol_analysis = daily['symbol_analysis']
                        if symbol_analysis:
                            print(f"   技术指标: RSI={symbol_analysis.get('rsi', 0):.1f}, 52周位置={symbol_analysis.get('position_52w', 0):.1f}%")
                
            else:
                print(f"❌ AI分析失败: {result.get('error', 'Unknown error')}")
                
        else:
            print(f"❌ 无法获取 {test_symbol} 数据")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print(f"\n✅ 测试完成!")

def main():
    """主函数"""
    print("🚀 开始测试AI每日持股分析系统")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    asyncio.run(test_ai_daily_system())

if __name__ == "__main__":
    main() 