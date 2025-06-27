#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA 加仓分析脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.decision_support_system import DecisionSupportSystem
from analysis.unified_stock_analyzer import UnifiedStockAnalyzer
import yfinance as yf

def analyze_nvda_position():
    """分析NVDA持仓和加仓机会"""
    
    print("🔍 NVDA 投资决策分析")
    print("=" * 50)
    
    # 初始化系统
    dss = DecisionSupportSystem()
    analyzer = UnifiedStockAnalyzer()
    
    # 获取当前价格
    ticker = yf.Ticker('NVDA')
    hist = ticker.history(period='1d')
    current_price = hist['Close'].iloc[-1]
    
    # 持仓信息
    cost_basis = 137.94
    shares = 35
    current_value = current_price * shares
    cost_value = cost_basis * shares
    profit_pct = (current_price - cost_basis) / cost_basis * 100
    
    print(f"📊 持仓状况:")
    print(f"  当前价格: ${current_price:.2f}")
    print(f"  持仓成本: ${cost_basis:.2f}")
    print(f"  持仓数量: {shares} 股")
    print(f"  持仓市值: ${current_value:,.2f}")
    print(f"  持仓成本: ${cost_value:,.2f}")
    print(f"  盈亏状况: {profit_pct:+.1f}% (${(current_value - cost_value):+,.2f})")
    print()
    
    try:
        # 获取综合分析
        print("🔍 正在进行综合分析...")
        current_analysis = analyzer.get_comprehensive_analysis('NVDA')
        print("✅ 综合分析完成")
        print()
        
        # 买入时机分析（加仓评估）
        print("🎯 加仓时机评估:")
        print("-" * 30)
        buy_decision = dss.analyze_buy_timing('NVDA', current_analysis)
        decision_detail = buy_decision['decision']
        
        # 决策建议
        action = decision_detail['action']
        confidence = decision_detail['confidence']
        score = decision_detail['score']
        
        if action == "建议买入":
            print(f"🟢 {action} (信心度: {confidence}%, 评分: {score}/100)")
            print("💡 建议: 可以考虑适量加仓")
        elif action == "可以买入":
            print(f"🟡 {action} (信心度: {confidence}%, 评分: {score}/100)")
            print("💡 建议: 可以小幅加仓，但需控制仓位")
        elif action == "等待更好时机":
            print(f"🟠 {action} (信心度: {confidence}%, 评分: {score}/100)")
            print("💡 建议: 暂时不加仓，等待更好的入场点")
        else:
            print(f"🔴 {action} (信心度: {confidence}%, 评分: {score}/100)")
            print("💡 建议: 避免加仓")
        
        print()
        
        # 支持理由
        print("✅ 支持理由:")
        for reason in decision_detail.get('reasons', []):
            print(f"  • {reason}")
        print()
        
        # 风险提醒
        print("⚠️ 风险提醒:")
        for warning in decision_detail.get('warnings', []):
            print(f"  • {warning}")
        print()
        
        # 持仓管理建议
        print("💰 持仓管理分析:")
        print("-" * 30)
        position_info = {'cost_basis': cost_basis, 'shares': shares}
        sell_decision = dss.analyze_sell_timing('NVDA', position_info, current_analysis)
        sell_detail = sell_decision['decision']
        
        hold_action = sell_detail['action']
        hold_confidence = sell_detail['confidence']
        
        if hold_action == "继续持有":
            print(f"🟢 {hold_action} (信心度: {hold_confidence}%)")
            print("💡 当前持仓建议继续持有")
        elif hold_action == "考虑减仓":
            print(f"🟡 {hold_action} (信心度: {hold_confidence}%)")
            print("💡 可以考虑适当减仓锁定利润")
        elif hold_action == "分批获利":
            print(f"🟠 {hold_action} (信心度: {hold_confidence}%)")
            print("💡 建议分批卖出部分持仓")
        else:
            print(f"🔴 {hold_action} (信心度: {hold_confidence}%)")
            print("💡 考虑止损或大幅减仓")
        
        print(f"📝 分析摘要: {sell_detail['summary']}")
        print()
        
        # 综合建议
        print("🎯 综合投资建议:")
        print("=" * 30)
        
        if profit_pct > 20:
            print("📈 当前盈利丰厚，建议:")
            print("  1. 可以考虑适当减仓锁定利润")
            print("  2. 如果要加仓，建议小幅加仓")
            print("  3. 设置跟踪止损保护利润")
        elif profit_pct > 10:
            print("📊 当前盈利适中，建议:")
            print("  1. 根据技术分析决定是否加仓")
            print("  2. 控制总仓位不超过预期")
            print("  3. 密切关注市场变化")
        elif profit_pct > 0:
            print("📉 当前小幅盈利，建议:")
            print("  1. 谨慎加仓，优先保护现有利润")
            print("  2. 等待更明确的上涨信号")
            print("  3. 可以考虑分批加仓策略")
        else:
            print("📉 当前处于亏损状态，建议:")
            print("  1. 避免盲目加仓摊低成本")
            print("  2. 等待技术面改善信号")
            print("  3. 考虑设置止损位")
        
        # 保存分析记录
        dss.save_decision(buy_decision)
        dss.save_decision(sell_decision)
        dss.add_user_note('NVDA', f"持仓分析: 成本${cost_basis}, {shares}股, 盈亏{profit_pct:.1f}%, 考虑加仓决策")
        
        print()
        print("💾 分析记录已保存到决策历史")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_nvda_position() 