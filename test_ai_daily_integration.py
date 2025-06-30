#!/usr/bin/env python3
"""
测试AI每日持股分析功能集成
"""

import asyncio
import json
from datetime import datetime

# 导入相关模块
from ai_realtime_analyzer import AIRealtimeAnalyzer
from enhanced_ai_data_integrator import EnhancedAIDataIntegrator

async def test_ai_daily_analysis():
    """测试AI每日持股分析功能"""
    print("🧪 开始测试AI每日持股分析功能集成...")
    
    try:
        # 初始化AI分析器
        ai_analyzer = AIRealtimeAnalyzer(use_daily_analysis=True)
        data_integrator = EnhancedAIDataIntegrator()
        
        print("✅ AI分析器和数据集成器初始化成功")
        
        # 加载投资组合配置
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            portfolio_config = json.load(f)
        
        # 获取持仓股票
        positions = portfolio_config.get('positions', {})
        current_positions = []
        
        for symbol, position in positions.items():
            if (not symbol.endswith('.HK') and 
                position.get('shares', 0) > 0 and 
                position.get('status') != 'SOLD'):
                current_positions.append(symbol)
        
        print(f"📊 当前持仓股票: {current_positions}")
        
        if not current_positions:
            print("❌ 没有找到有效的持仓股票")
            return False
        
        # 选择第一个股票进行测试
        test_symbol = current_positions[0]
        print(f"🎯 测试股票: {test_symbol}")
        
        # 获取持仓信息
        position_info = positions.get(test_symbol, {})
        print(f"📋 持仓信息: {position_info}")
        
        # 模拟市场数据
        market_data = {
            test_symbol: {
                'price': 150.0,
                'change_pct': 2.5,
                'volume': 1000000,
                'rsi': 65.0,
                'ma_20': 145.0,
                'ma_50': 140.0,
                'position_info': {
                    'shares': position_info.get('shares', 0),
                    'cost_basis': position_info.get('cost_basis', 0),
                    'weight': position_info.get('weight', 0),
                    'sector': position_info.get('sector', 'Unknown')
                }
            }
        }
        
        # 获取增强数据
        enhanced_data = data_integrator.get_comprehensive_data_for_ai(test_symbol)
        print(f"📊 增强数据获取成功: {len(enhanced_data) if isinstance(enhanced_data, dict) else '数据获取失败'}")
        
        # 执行AI分析
        print("🤖 开始AI分析...")
        ai_result = await ai_analyzer.analyze_market_event(
            symbol=test_symbol,
            event_type="portfolio_position",
            market_data=market_data,
            analysis_type="comprehensive"
        )
        
        if ai_result and ai_result.get('success'):
            print("✅ AI分析成功完成")
            
            # 显示分析结果
            action_suggestion = ai_result.get('action_suggestion', {})
            action = action_suggestion.get('action', 'N/A')
            print(f"🎯 操作建议: {action}")
            
            # 显示详细分析
            ai_analysis = ai_result.get('ai_analysis', '无分析内容')
            print(f"📊 AI分析内容长度: {len(ai_analysis)} 字符")
            
            # 显示风险提示
            risk_warnings = ai_result.get('risk_warnings', [])
            if risk_warnings:
                print(f"⚠️ 风险提示数量: {len(risk_warnings)}")
            
            # 显示分析理由
            reasons = ai_result.get('reasons', [])
            if reasons:
                print(f"📋 分析理由数量: {len(reasons)}")
            
            print("\n🎉 AI每日持股分析功能集成测试成功！")
            return True
            
        else:
            print("❌ AI分析失败")
            return False
            
    except Exception as e:
        print(f"💥 测试过程中出现错误: {e}")
        return False

def main():
    """主函数"""
    print("🚀 启动AI每日持股分析功能集成测试...")
    
    # 运行异步测试
    success = asyncio.run(test_ai_daily_analysis())
    
    if success:
        print("\n✅ 所有测试通过！")
        print("🎯 系统已成功集成AI每日持股分析功能")
        print("🌐 请访问 http://localhost:8503 查看完整系统")
    else:
        print("\n💥 测试失败，请检查系统配置")

if __name__ == "__main__":
    main() 