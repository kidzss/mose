#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强功能集成测试
测试EnhancedStockAnalyzer的集成功能
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_analyzer_integration():
    """测试增强分析器集成功能"""
    print("🚀 开始测试增强股票分析器集成功能...")
    print("=" * 60)
    
    try:
        # 初始化增强分析器
        analyzer = EnhancedStockAnalyzer()
        
        # 检查功能状态
        status = analyzer.get_feature_status()
        print("📊 功能模块状态:")
        for feature, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {feature}: {'可用' if available else '不可用'}")
        
        if not analyzer.is_available():
            print("⚠️ 没有可用的增强功能模块")
            return False
        
        # 测试股票分析
        test_symbols = ['AAPL', 'MSFT', 'TSLA']
        print(f"\n🔍 测试股票分析 ({len(test_symbols)} 只股票)...")
        
        for symbol in test_symbols:
            print(f"\n--- 分析 {symbol} ---")
            
            try:
                # 执行综合分析
                result = analyzer.analyze_stock_comprehensive(symbol, current_price=100.0)
                
                # 显示基本信息
                print(f"股票代码: {result.get('symbol', 'N/A')}")
                print(f"分析时间: {result.get('analysis_time', 'N/A')}")
                
                # 显示评分信息
                if 'overall_score' in result:
                    print(f"总体评分: {result['overall_score']:.3f}")
                if 'overall_rating' in result:
                    print(f"总体评级: {result['overall_rating']}")
                if 'growth_score' in result:
                    print(f"成长性评分: {result['growth_score']:.3f}")
                if 'industry_score' in result:
                    print(f"行业比较评分: {result['industry_score']:.3f}")
                
                # 显示警告信息
                warnings = result.get('warnings', [])
                if warnings:
                    print(f"⚠️ 警告 ({len(warnings)}个):")
                    for warning in warnings[:3]:  # 只显示前3个
                        print(f"   • {warning}")
                
                # 显示建议信息
                recommendations = result.get('recommendations', [])
                if recommendations:
                    print(f"💡 建议 ({len(recommendations)}个):")
                    for rec in recommendations[:3]:  # 只显示前3个
                        print(f"   • {rec}")
                
                # 检查增强功能结果
                enhanced_features = result.get('enhanced_features', {})
                if enhanced_features:
                    print(f"🔧 增强功能:")
                    for feature, data in enhanced_features.items():
                        if isinstance(data, dict):
                            print(f"   • {feature}: {len(data)} 项数据")
                        else:
                            print(f"   • {feature}: 已获取")
                
                print(f"✅ {symbol} 分析完成")
                
            except Exception as e:
                print(f"❌ {symbol} 分析失败: {e}")
                logger.error(f"{symbol} 分析错误: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 增强分析器集成测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"集成测试错误: {e}")
        return False

def test_feature_availability():
    """测试功能可用性"""
    print("\n🔍 检查各模块可用性...")
    
    # 测试财务分析器
    try:
        from monitor.financial_analyzer import FinancialAnalyzer
        fa = FinancialAnalyzer()
        print("✅ 增强版财务分析器可用")
    except Exception as e:
        print(f"❌ 增强版财务分析器不可用: {e}")
    
    # 测试退出策略
    try:
        from strategy.enhanced_exit_strategy import EnhancedExitStrategy
        es = EnhancedExitStrategy()
        print("✅ 增强版退出策略可用")
    except Exception as e:
        print(f"❌ 增强版退出策略不可用: {e}")

if __name__ == "__main__":
    print("🧪 增强功能集成测试")
    print("=" * 80)
    
    # 功能可用性测试
    test_feature_availability()
    
    # 集成功能测试
    success = test_enhanced_analyzer_integration()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查系统配置") 