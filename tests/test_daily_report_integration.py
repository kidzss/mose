#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日报系统增强功能集成测试
验证SmartDailyReportGenerator是否正确集成了增强分析功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.smart_daily_report import SmartDailyReportGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_daily_report_enhanced_integration():
    """测试日报系统的增强功能集成"""
    print("🚀 测试日报系统增强功能集成...")
    print("=" * 60)
    
    try:
        # 初始化日报生成器
        print("📊 初始化日报生成器...")
        generator = SmartDailyReportGenerator(
            watchlist=['AAPL', 'MSFT'],  # 只测试2只股票
            auto_update_data=False  # 禁用数据更新以加快测试
        )
        
        # 检查增强分析器是否正确初始化
        if generator.enhanced_analyzer:
            print("✅ 增强分析器已集成到日报系统")
            feature_status = generator.enhanced_analyzer.get_feature_status()
            print("📋 增强功能状态:")
            for feature, available in feature_status.items():
                status_icon = "✅" if available else "❌"
                print(f"   {status_icon} {feature}: {'可用' if available else '不可用'}")
        else:
            print("⚠️ 增强分析器未成功集成")
            return False
        
        # 测试股票分析功能
        print("\n🔍 测试股票分析功能...")
        test_symbol = 'AAPL'
        
        try:
            print(f"分析 {test_symbol}...")
            result = generator._analyze_stock(test_symbol)
            
            if result:
                print(f"✅ {test_symbol} 分析成功")
                
                # 检查基本分析结果
                print("📊 基本分析结果:")
                print(f"   当前价格: ${result.get('current_price', 'N/A'):.2f}")
                print(f"   价格变化: {result.get('price_change', 0):.2f}%")
                print(f"   RSI: {result.get('rsi', 'N/A'):.1f}")
                
                # 检查传统财务分析
                if 'financial_analysis' in result:
                    fa = result['financial_analysis']
                    print(f"   财务评分: {fa.get('total_score', 'N/A'):.3f}")
                    print(f"   财务评级: {fa.get('overall_rating', 'N/A')}")
                
                # 检查增强分析结果
                if 'enhanced_analysis' in result:
                    ea = result['enhanced_analysis']
                    print("🔧 增强分析结果:")
                    print(f"   总体评分: {ea.get('overall_score', 'N/A'):.3f}")
                    print(f"   总体评级: {ea.get('overall_rating', 'N/A')}")
                    print(f"   成长性评分: {ea.get('growth_score', 'N/A'):.3f}")
                    print(f"   行业比较评分: {ea.get('industry_score', 'N/A'):.3f}")
                    
                    # 显示增强功能数据
                    enhanced_features = ea.get('enhanced_features', {})
                    if enhanced_features:
                        print("   增强功能数据:")
                        for feature, data in enhanced_features.items():
                            if isinstance(data, dict):
                                print(f"     • {feature}: {len(data)} 项数据")
                            else:
                                print(f"     • {feature}: 已获取")
                
                # 检查增强建议
                if 'enhanced_recommendations' in result:
                    recommendations = result['enhanced_recommendations']
                    print(f"💡 增强建议 ({len(recommendations)}个):")
                    for rec in recommendations[:3]:  # 显示前3个
                        print(f"   • {rec}")
                
                # 检查增强警告
                if 'enhanced_warnings' in result:
                    warnings = result['enhanced_warnings']
                    print(f"⚠️ 增强警告 ({len(warnings)}个):")
                    for warning in warnings[:3]:  # 显示前3个
                        print(f"   • {warning}")
                
                print(f"✅ {test_symbol} 集成测试通过")
                return True
                
            else:
                print(f"❌ {test_symbol} 分析失败 - 无结果")
                return False
                
        except Exception as e:
            print(f"❌ {test_symbol} 分析失败: {e}")
            logger.error(f"股票分析错误: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"集成测试错误: {e}")
        return False

def test_report_generation():
    """测试报告生成功能"""
    print("\n📄 测试报告生成功能...")
    
    try:
        # 创建一个轻量级的报告生成器
        generator = SmartDailyReportGenerator(
            watchlist=['AAPL'],  # 只测试1只股票
            auto_update_data=False
        )
        
        # 测试报告生成
        print("生成测试报告...")
        html_content = generator.generate_report()
        
        if html_content and len(html_content) > 1000:  # 检查是否有实质内容
            print("✅ 报告生成成功")
            
            # 检查是否包含增强分析内容
            enhanced_keywords = ['增强分析', '成长性', '行业比较', '总体评分']
            found_enhanced = any(keyword in html_content for keyword in enhanced_keywords)
            
            if found_enhanced:
                print("✅ 报告包含增强分析内容")
            else:
                print("⚠️ 报告可能未包含增强分析内容")
            
            return True
        else:
            print("❌ 报告生成失败或内容过少")
            return False
            
    except Exception as e:
        print(f"❌ 报告生成测试失败: {e}")
        logger.error(f"报告生成错误: {e}")
        return False

if __name__ == "__main__":
    print("🧪 日报系统增强功能集成测试")
    print("=" * 80)
    
    # 测试增强功能集成
    integration_success = test_daily_report_enhanced_integration()
    
    # 测试报告生成
    if integration_success:
        report_success = test_report_generation()
        
        if integration_success and report_success:
            print("\n🎉 所有测试通过！增强功能已成功集成到日报系统")
        else:
            print("\n⚠️ 部分测试失败，请检查系统配置")
    else:
        print("\n❌ 集成测试失败，跳过报告生成测试") 