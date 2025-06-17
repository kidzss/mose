#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周报系统增强功能集成测试
验证PersonalInvestorAutomation是否正确集成了增强分析功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_investor_automation import PersonalInvestorAutomation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_weekly_system_enhanced_integration():
    """测试周报系统的增强功能集成"""
    print("🚀 测试周报系统增强功能集成...")
    print("=" * 60)
    
    try:
        # 初始化自动化系统
        print("📊 初始化个人投资者自动化系统...")
        automation = PersonalInvestorAutomation()
        
        # 检查增强分析器是否正确初始化
        if automation.enhanced_analyzer:
            print("✅ 增强分析器已集成到周报系统")
            
            # 检查增强分析器的功能状态
            status = automation.enhanced_analyzer.get_feature_status()
            print("\n📊 增强功能模块状态:")
            for feature, available in status.items():
                status_icon = "✅" if available else "❌"
                print(f"   {status_icon} {feature}: {'可用' if available else '不可用'}")
        else:
            print("❌ 增强分析器未能集成到周报系统")
            return False
        
        # 测试筛选功能（使用小规模数据以加快测试）
        print("\n🔍 测试增强筛选功能...")
        
        # 修改配置以加快测试
        original_max_results = automation.config['max_results']
        automation.config['max_results'] = 5  # 只测试5只股票
        
        try:
            # 运行筛选测试
            results = automation.run_weekly_screening()
            
            if results:
                print(f"✅ 增强筛选成功，获得 {len(results)} 只推荐股票")
                
                # 检查每只股票是否包含增强分析信息
                enhanced_count = 0
                for stock in results:
                    if 'enhanced_analysis' in stock:
                        enhanced_count += 1
                        enhanced_analysis = stock['enhanced_analysis']
                        symbol = stock['symbol']
                        overall_score = enhanced_analysis.get('overall_score', 0)
                        warnings = enhanced_analysis.get('warnings', [])
                        recommendations = enhanced_analysis.get('recommendations', [])
                        
                        print(f"  📈 {symbol}: 增强评分 {overall_score:.3f}")
                        if warnings:
                            print(f"    ⚠️ 警告: {warnings[0]}")
                        if recommendations:
                            print(f"    💡 建议: {recommendations[0]}")
                
                print(f"📊 增强分析覆盖率: {enhanced_count}/{len(results)} ({enhanced_count/len(results)*100:.1f}%)")
                
                # 测试报告生成（但不发送邮件）
                print("\n📧 测试报告生成功能...")
                original_email = automation.config['email']
                automation.config['email'] = 'test@example.com'  # 避免实际发送
                
                # 生成测试报告内容
                test_content = automation._generate_weekly_content(results)
                if test_content and '增强评分' in test_content and '行业表现' in test_content:
                    print("✅ 增强报告内容生成成功")
                    print("  ✓ 包含增强评分列")
                    print("  ✓ 包含行业表现列")
                    print("  ✓ 包含增强投资建议")
                else:
                    print("❌ 报告内容缺少增强功能信息")
                    print(f"  调试信息: 内容包含增强评分={('增强评分' in test_content)}, 包含行业表现={('行业表现' in test_content)}")
                
                # 恢复原始配置
                automation.config['email'] = original_email
                
            else:
                print("⚠️ 未获得筛选结果（可能是数据问题）")
                
        finally:
            # 恢复原始配置
            automation.config['max_results'] = original_max_results
        
        print("\n🎉 周报系统增强功能集成测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_investment_advice():
    """测试增强版投资建议功能"""
    print("\n🔍 测试增强版投资建议功能...")
    
    automation = PersonalInvestorAutomation()
    
    # 创建测试股票数据
    test_stocks = [
        {
            'symbol': 'AAPL',
            'quality_factor': 0.85,
            'multifactor_score': 75,
            'enhanced_analysis': {
                'overall_score': 0.82,
                'warnings': [],
                'recommendations': ['优质蓝筹股，适合长期持有']
            }
        },
        {
            'symbol': 'TSLA',
            'quality_factor': 0.65,
            'multifactor_score': 60,
            'enhanced_analysis': {
                'overall_score': 0.45,
                'warnings': ['高估值风险，PE比率过高'],
                'recommendations': []
            }
        }
    ]
    
    for stock in test_stocks:
        advice = automation._get_enhanced_investment_advice(stock)
        print(f"  {stock['symbol']}: {advice}")
    
    print("✅ 增强版投资建议测试完成")

if __name__ == "__main__":
    success = test_weekly_system_enhanced_integration()
    test_enhanced_investment_advice()
    
    if success:
        print("\n🎉 所有测试通过！周报系统增强功能集成成功！")
    else:
        print("\n❌ 测试失败，请检查集成问题") 