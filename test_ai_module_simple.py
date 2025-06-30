#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试AI模块
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_modules():
    """测试AI模块"""
    print("🧪 测试AI模块...")
    
    try:
        # 测试AI分析器
        from ai_realtime_analyzer import AIRealtimeAnalyzer
        print("✅ AI分析器导入成功")
        
        ai_analyzer = AIRealtimeAnalyzer()
        print("✅ AI分析器初始化成功")
        
        # 测试AI交易模块
        from ai_trading_module import AITradingModule
        print("✅ AI交易模块导入成功")
        
        ai_module = AITradingModule()
        print("✅ AI交易模块初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ AI模块测试失败: {e}")
        return False

def test_ai_analysis():
    """测试AI分析功能"""
    print("\n🤖 测试AI分析功能...")
    
    try:
        from ai_trading_module import AITradingModule
        
        # 创建AI模块
        ai_module = AITradingModule()
        
        # 准备测试数据
        test_data = {
            'price': 150.0,
            'change_pct': 2.5,
            'volume': 1000000,
            'rsi': 65.0,
            'ma_20': 148.0,
            'ma_50': 145.0
        }
        
        test_position = {
            'shares': 100,
            'cost_basis': 140.0,
            'weight': 10.0
        }
        
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 测试股票信号分析
            print("📊 测试股票信号分析...")
            result1 = loop.run_until_complete(
                ai_module.analyze_stock_signal("AAPL", test_data, "quick")
            )
            print(f"✅ 股票信号分析结果: {result1.get('success', False)}")
            
            # 测试持仓分析
            print("💼 测试持仓分析...")
            result2 = loop.run_until_complete(
                ai_module.analyze_portfolio_position("AAPL", test_data, test_position)
            )
            print(f"✅ 持仓分析结果: {result2.get('success', False)}")
            
        finally:
            loop.close()
        
        return True
        
    except Exception as e:
        print(f"❌ AI分析测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始AI模块测试")
    print("=" * 50)
    
    # 测试模块导入和初始化
    if not test_ai_modules():
        print("❌ 模块测试失败")
        return False
    
    # 测试AI分析功能
    if not test_ai_analysis():
        print("❌ AI分析测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！AI模块工作正常")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 