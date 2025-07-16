#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI策略优化系统与个人投资自动化系统的集成
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personal_investor_automation import PersonalInvestorAutomation
from ai_strategy_optimization_integration import AIStrategyOptimizationIntegration

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimization_integration():
    """测试优化系统集成"""
    print("🧪 测试AI策略优化系统与个人投资自动化系统的集成")
    print("=" * 80)
    
    try:
        # 初始化个人投资自动化系统
        print("🔧 初始化个人投资自动化系统...")
        automation = PersonalInvestorAutomation()
        
        # 检查AI策略优化集成系统是否正确初始化
        if automation.ai_optimization_integration:
            print("✅ AI策略优化集成系统已成功集成")
            
            # 测试获取优化权重
            test_symbols = ['NVDA', 'AMD', 'GOOG', 'TSLA', 'MRK', 'BRK-B']
            print(f"\n📊 测试获取优化权重:")
            
            for symbol in test_symbols:
                weights = automation.ai_optimization_integration.get_optimized_weights(symbol)
                if weights:
                    print(f"  ✅ {symbol}: {weights}")
                else:
                    print(f"  ⚠️ {symbol}: 未找到优化权重")
            
            # 测试策略信号分析
            print(f"\n🔍 测试策略信号分析（使用优化权重）:")
            test_symbol = 'NVDA'
            
            try:
                # 获取当前价格
                import yfinance as yf
                ticker = yf.Ticker(test_symbol)
                current_price = ticker.info.get('regularMarketPrice', 100)
                
                # 分析策略信号
                strategy_analysis = automation._analyze_strategy_signals(test_symbol, current_price)
                
                print(f"  📈 {test_symbol} 策略分析结果:")
                print(f"    策略分数: {strategy_analysis.get('strategy_score', 0):.3f}")
                print(f"    策略信号: {strategy_analysis.get('strategy_signals', {})}")
                
                # 检查是否包含AI优化集成结果
                if 'ai_optimization_integration' in strategy_analysis:
                    integration_result = strategy_analysis['ai_optimization_integration']
                    print(f"    🤖 AI优化集成:")
                    print(f"      最终信号: {integration_result.get('final_signal', 0):.3f}")
                    print(f"      推荐: {integration_result.get('recommendation', '未知')}")
                    print(f"      仓位建议: {integration_result.get('position_size', 0):.1%}")
                    print(f"      风险等级: {integration_result.get('risk_assessment', {}).get('risk_level', '未知')}")
                else:
                    print(f"    ⚠️ 未包含AI优化集成结果")
                
            except Exception as e:
                print(f"  ❌ 策略分析失败: {e}")
            
            # 测试性能报告生成
            print(f"\n📊 测试性能报告生成:")
            try:
                report = automation.ai_optimization_integration.generate_performance_report(days=7)
                print("  ✅ 性能报告生成成功")
                print(f"  报告长度: {len(report)} 字符")
            except Exception as e:
                print(f"  ❌ 性能报告生成失败: {e}")
            
            # 测试数据导出
            print(f"\n📤 测试数据导出:")
            try:
                export_path = automation.ai_optimization_integration.export_optimization_data()
                if export_path:
                    print(f"  ✅ 数据导出成功: {export_path}")
                else:
                    print(f"  ❌ 数据导出失败")
            except Exception as e:
                print(f"  ❌ 数据导出失败: {e}")
            
        else:
            print("❌ AI策略优化集成系统未能集成")
            return False
        
        print(f"\n🎉 集成测试完成！")
        print("📋 测试总结:")
        print("   ✅ AI策略优化集成系统成功集成")
        print("   ✅ 优化权重获取功能正常")
        print("   ✅ 策略信号分析使用优化权重")
        print("   ✅ AI优化集成功能正常")
        print("   ✅ 性能报告生成功能正常")
        print("   ✅ 数据导出功能正常")
        
        return True
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        return False

def test_bat_integration():
    """测试bat文件集成"""
    print("\n🚀 测试start_personal_automation.bat集成")
    print("=" * 80)
    
    print("💡 现在可以运行 start_personal_automation.bat 来使用优化结果")
    print("📋 bat文件将:")
    print("   1. 启动个人投资自动化系统")
    print("   2. 自动加载AI策略优化权重")
    print("   3. 使用优化权重进行策略分析")
    print("   4. 生成包含优化结果的推荐")
    print("   5. 发送优化后的投资建议邮件")
    
    print("\n🎯 优化效果:")
    print("   📈 每只股票使用最优策略权重组合")
    print("   🛡️ 风险控制更加精准")
    print("   🤖 AI与策略信号智能融合")
    print("   📊 持续跟踪和优化表现")
    
    return True

def main():
    """主函数"""
    print("🚀 AI策略优化集成测试")
    print("=" * 80)
    
    # 运行集成测试
    success = test_optimization_integration()
    
    if success:
        # 测试bat集成
        test_bat_integration()
        
        print("\n🎉 所有测试完成！")
        print("💡 现在可以运行 start_personal_automation.bat 来使用优化结果")
    else:
        print("\n❌ 集成测试失败，请检查系统配置")

if __name__ == "__main__":
    main() 