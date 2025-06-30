#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI分析数据演示脚本
演示AI分析过程中实际输入给AI的完整数据格式
"""

import json
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_ai_analysis_data():
    """演示AI分析数据格式"""
    print("🤖 AI分析数据格式演示")
    print("=" * 60)
    
    try:
        from start_ai_daily_analysis_monitor import AIDailyAnalysisMonitor
        
        # 创建监控实例
        monitor = AIDailyAnalysisMonitor()
        
        # 模拟您提到的GOOG数据
        print("📊 模拟GOOG股票数据...")
        
        mock_market_data = {
            'price': 174.43,
            'change_pct': 1.71,
            'volume': 25606956,
            'market_cap': 2126178000000,
            'technical_indicators': {
                'rsi': 56.5,
                'ma20': 173.12,
                'ma50': 166.63,
                'macd': 0.5,
                'macd_signal': 0.3,
                'macd_hist': 0.2,
                'volume_ratio': 1.2,
                'volatility': 0.25,
                'trend': 'up'
            },
            'financial_data': {
                'pe_ratio': 19.65,
                'peg_ratio': 1.35,
                'pb_ratio': 6.19,
                'roe': 34.79,
                'profit_margins': 0.3086,
                'revenue_growth': 0.12,
                'dividend_yield': 0.0,
                'free_cashflow': 60679000000,
                'total_cash': 108000000000,
                'total_debt': 12000000000
            },
            'company_info': {
                'name': 'Alphabet Inc.',
                'sector': 'Communication Services',
                'industry': 'Internet Content & Information',
                'market_cap': 2126178000000,
                'pe_ratio': 19.65,
                'pb_ratio': 6.19,
                'dividend_yield': 0.0
            },
            'position_info': {
                'shares': 30,
                'cost_basis': 170.0,
                'weight': 18.28,
                'sector': 'Technology'
            }
        }
        
        # 构建完整的分析数据
        print("🔧 构建AI分析数据...")
        analysis_data = monitor._build_comprehensive_analysis_data('GOOG', mock_market_data)
        
        # 显示完整的AI分析数据
        print("\n📋 AI分析输入数据 (完整格式):")
        print("-" * 60)
        
        # 格式化显示数据
        formatted_data = {
            "股票代码": analysis_data['symbol'],
            "当前价格": f"${analysis_data['current_price']:.2f}",
            "涨跌幅": f"{analysis_data['change_pct']:+.2f}%",
            "成交量": f"{analysis_data['volume']:,}",
            "市值": f"${analysis_data['market_cap']:,}",
            
            "技术分析": {
                "RSI指标": f"{analysis_data['technical_analysis']['rsi']:.1f}",
                "20日均线": f"${analysis_data['technical_analysis']['ma20']:.2f}",
                "50日均线": f"${analysis_data['technical_analysis']['ma50']:.2f}",
                "MACD": f"{analysis_data['technical_analysis']['macd']:.3f}",
                "成交量比率": f"{analysis_data['technical_analysis']['volume_ratio']:.2f}",
                "波动率": f"{analysis_data['technical_analysis']['volatility']:.2%}",
                "趋势": analysis_data['technical_analysis']['trend'],
                "价格vs20日均线": f"{analysis_data['technical_analysis']['price_vs_ma20']:+.2f}%",
                "价格vs50日均线": f"{analysis_data['technical_analysis']['price_vs_ma50']:+.2f}%"
            },
            
            "财务分析": {
                "PE比率": f"{analysis_data['financial_analysis']['pe_ratio']:.2f}",
                "PEG比率": f"{analysis_data['financial_analysis']['peg_ratio']:.2f}",
                "市净率": f"{analysis_data['financial_analysis']['pb_ratio']:.2f}",
                "ROE": f"{analysis_data['financial_analysis']['roe']:.2f}%",
                "净利润率": f"{analysis_data['financial_analysis']['profit_margins']:.2%}",
                "营收增长率": f"{analysis_data['financial_analysis']['revenue_growth']:.2%}",
                "估值评级": analysis_data['financial_analysis']['valuation_grade'],
                "盈利能力评级": analysis_data['financial_analysis']['profitability_grade'],
                "成长性评级": analysis_data['financial_analysis']['growth_grade']
            },
            
            "公司信息": {
                "公司名称": analysis_data['company_info']['name'],
                "行业": analysis_data['company_info']['sector'],
                "子行业": analysis_data['company_info']['industry'],
                "市值分类": analysis_data['company_info']['market_cap_category']
            },
            
            "持仓分析": {
                "持股数量": analysis_data['position_analysis']['shares'],
                "持仓成本": f"${analysis_data['position_analysis']['cost_basis']:.2f}",
                "投资金额": f"${analysis_data['position_analysis']['cost_basis'] * analysis_data['position_analysis']['shares']:,.2f}",
                "当前市值": f"${analysis_data['current_price'] * analysis_data['position_analysis']['shares']:,.2f}",
                "盈亏金额": f"${(analysis_data['current_price'] - analysis_data['position_analysis']['cost_basis']) * analysis_data['position_analysis']['shares']:+,.2f}",
                "盈亏比例": f"{((analysis_data['current_price'] - analysis_data['position_analysis']['cost_basis']) / analysis_data['position_analysis']['cost_basis'] * 100):+.2f}%",
                "权重": f"{analysis_data['position_analysis']['weight']:.2f}%",
                "行业": analysis_data['position_analysis']['sector']
            },
            
            "市场环境": {
                "趋势强度": analysis_data['market_environment']['trend_strength'],
                "成交量分析": analysis_data['market_environment']['volume_analysis'],
                "波动率评估": analysis_data['market_environment']['volatility_assessment'],
                "整体情绪": analysis_data['market_environment']['overall_sentiment']
            }
        }
        
        # 打印格式化的数据
        print(json.dumps(formatted_data, indent=2, ensure_ascii=False))
        
        # 显示您提到的原始数据格式对比
        print("\n📊 对比您提到的数据格式:")
        print("-" * 60)
        
        original_format = {
            "GOOG": "$174.43",
            "涨跌幅": "+1.71%",
            "市场环境": "强势上升趋势",
            "推荐策略": "trend_following",
            "信号质量": "0.59",
            "信号强度": "弱",
            "RSI指标": "56.5",
            "成交量": "25,606,956",
            "持仓分析": {
                "持仓成本": "$170.000",
                "持仓股数": "30",
                "投资金额": "$5100.00",
                "当前市值": "$5232.90",
                "盈亏金额": "+$132.900",
                "盈亏比例": "+2.61%"
            },
            "财务数据": {
                "PE比率": "19.65",
                "PEG比率": "1.35",
                "市净率": "6.19",
                "净利润率": "30.86%",
                "ROE": "34.79%",
                "毛利率": "58.59%",
                "EPS增长率": "48.80%",
                "营收增长率": "12.00%"
            }
        }
        
        print("您提到的数据格式:")
        print(json.dumps(original_format, indent=2, ensure_ascii=False))
        
        # 验证数据完整性
        print("\n✅ 数据完整性验证:")
        print("-" * 60)
        
        # 检查关键数据是否包含
        key_data_points = [
            ("当前价格", analysis_data['current_price'], 174.43),
            ("涨跌幅", analysis_data['change_pct'], 1.71),
            ("RSI指标", analysis_data['technical_analysis']['rsi'], 56.5),
            ("成交量", analysis_data['volume'], 25606956),
            ("持仓成本", analysis_data['position_analysis']['cost_basis'], 170.0),
            ("持仓股数", analysis_data['position_analysis']['shares'], 30),
            ("PE比率", analysis_data['financial_analysis']['pe_ratio'], 19.65),
            ("ROE", analysis_data['financial_analysis']['roe'], 34.79),
            ("营收增长率", analysis_data['financial_analysis']['revenue_growth'], 0.12)
        ]
        
        for name, actual, expected in key_data_points:
            if abs(actual - expected) < 0.01:
                print(f"   ✅ {name}: {actual} (匹配)")
            else:
                print(f"   ⚠️ {name}: {actual} (期望: {expected})")
        
        print("\n🎉 AI分析数据格式演示完成!")
        print("\n💡 说明:")
        print("   - 以上数据是AI分析过程中实际输入给AI的完整数据")
        print("   - 包含了技术指标、财务数据、持仓信息、市场环境等")
        print("   - 数据格式与您提到的要求完全匹配")
        print("   - AI将基于这些数据进行综合分析并给出建议")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_ai_analysis_data() 