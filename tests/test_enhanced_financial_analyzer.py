#!/usr/bin/env python3
"""
增强财务分析器测试脚本
测试新增的成长性指标、行业对比和预警功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.financial_analyzer import FinancialAnalyzer
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_analyzer():
    """测试增强的财务分析器"""
    print("🔍 测试增强财务分析器功能")
    print("=" * 50)
    
    # 初始化分析器
    analyzer = FinancialAnalyzer()
    
    # 测试股票列表
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for symbol in test_symbols:
        print(f"\n📊 分析股票: {symbol}")
        print("-" * 30)
        
        try:
            # 执行完整分析
            result = analyzer.analyze_stock(symbol)
            
            if result:
                print(f"✅ 分析成功")
                print(f"📈 综合评分: {result['total_score']:.3f}")
                print(f"🏆 总体评级: {result['overall_rating']}")
                print(f"🏭 所属行业: {result['basic_info']['sector']}")
                
                # 显示行业对比结果
                industry_comp = result['dimensions']['industry_comparison']
                print(f"🔍 行业对比得分: {industry_comp['industry_adjusted_score']:.3f}")
                print(f"📊 行业内表现: {industry_comp['summary']}")
                
                # 显示成长性分析
                growth = result['dimensions']['growth']
                print(f"📈 成长性得分: {growth['score']:.3f}")
                print(f"🌱 成长性评级: {growth['summary']}")
                
                # 显示预警信息
                alerts = result['warning_alerts']
                print(f"⚠️  预警等级: {alerts['alert_level']}")
                
                if alerts['valuation_alerts']:
                    print("📊 估值预警:")
                    for alert in alerts['valuation_alerts']:
                        print(f"   - {alert['message']}")
                
                if alerts['fundamental_alerts']:
                    print("📉 基本面预警:")
                    for alert in alerts['fundamental_alerts']:
                        print(f"   - {alert['message']}")
                
                if alerts['risk_alerts']:
                    print("⚠️  风险预警:")
                    for alert in alerts['risk_alerts']:
                        print(f"   - {alert['message']}")
                
                # 显示详细的行业对比指标
                if industry_comp['relative_metrics']:
                    print("\n🔍 详细行业对比:")
                    for metric, info in industry_comp['relative_metrics'].items():
                        print(f"   {metric}: {info['relative_position']} (得分: {info['score']})")
                
            else:
                print(f"❌ 分析失败: 无法获取 {symbol} 的数据")
                
        except Exception as e:
            print(f"❌ 分析出错: {e}")
            logger.error(f"分析 {symbol} 时出错: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 测试完成！")

def test_industry_benchmarks():
    """测试行业基准值功能"""
    print("\n🏭 测试行业基准值功能")
    print("=" * 50)
    
    analyzer = FinancialAnalyzer()
    
    # 测试不同行业的基准值
    test_sectors = ['Technology', 'Financial Services', 'Healthcare', 'Energy', 'Consumer Cyclical']
    
    for sector in test_sectors:
        benchmarks = analyzer._get_industry_benchmarks(sector)
        print(f"\n📊 {sector} 行业基准:")
        print(f"   PE比率 - 优秀: {benchmarks['pe_ratio']['excellent']}, 良好: {benchmarks['pe_ratio']['good']}")
        print(f"   ROE - 优秀: {benchmarks['roe']['excellent']:.1%}, 良好: {benchmarks['roe']['good']:.1%}")
        print(f"   市净率 - 优秀: {benchmarks['pb_ratio']['excellent']}, 良好: {benchmarks['pb_ratio']['good']}")

def compare_traditional_vs_enhanced():
    """对比传统分析与增强分析的差异"""
    print("\n⚖️  传统分析 vs 增强分析对比")
    print("=" * 50)
    
    analyzer = FinancialAnalyzer()
    
    # 测试几个代表性股票
    test_symbols = ['AAPL', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n📊 对比分析: {symbol}")
        print("-" * 30)
        
        try:
            result = analyzer.analyze_stock(symbol)
            if result:
                # 模拟传统评分（不包含行业对比）
                traditional_score = (
                    result['dimensions']['valuation']['score'] * 0.25 +
                    result['dimensions']['profitability']['score'] * 0.25 +
                    result['dimensions']['growth']['score'] * 0.20 +
                    result['dimensions']['financial_health']['score'] * 0.20 +
                    result['dimensions']['analyst_sentiment']['score'] * 0.10
                )
                
                enhanced_score = result['total_score']
                
                print(f"🔺 传统评分: {traditional_score:.3f}")
                print(f"🔻 增强评分: {enhanced_score:.3f}")
                print(f"📊 差异: {enhanced_score - traditional_score:.3f}")
                print(f"🏭 行业调整影响: {result['dimensions']['industry_comparison']['industry_adjusted_score']:.3f}")
                
        except Exception as e:
            print(f"❌ 对比分析失败: {e}")

if __name__ == "__main__":
    # 运行测试
    test_enhanced_analyzer()
    test_industry_benchmarks()
    compare_traditional_vs_enhanced()
    
    print("\n🎉 所有测试完成！")
    print("\n📝 **功能增强总结:**")
    print("✅ 1. 新增EPS增长率和自由现金流分析")
    print("✅ 2. 实现行业对比分析功能")
    print("✅ 3. 添加智能预警系统")
    print("✅ 4. 调整评分权重，强化成长性考量")
    print("✅ 5. 提供详细的投资决策支持信息") 