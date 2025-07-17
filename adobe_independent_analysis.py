#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adobe独立分析 - 我的个人观点和买入价格
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_adobe_independent():
    """独立分析Adobe"""
    
    print("🎨 Adobe独立分析 - 我的个人观点")
    print("=" * 60)
    
    # 获取Adobe数据
    try:
        adbe = yf.Ticker("ADBE")
        info = adbe.info
        hist = adbe.history(period="6mo")
        
        current_price = 364.18
        print(f"📊 当前价格: ${current_price}")
        print(f"📈 52周高点: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        print(f"📉 52周低点: ${info.get('fiftyTwoWeekLow', 'N/A')}")
        print(f"💰 市值: ${info.get('marketCap', 'N/A'):,}")
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return
    
    print("\n📋 基于您提供的数据分析:")
    print("-" * 35)
    print("✅ 正面因素:")
    print("• RSI 35.5显示超卖状态")
    print("• 基本面强劲，盈利能力优秀")
    print("• 分析师目标价$491.32，上涨空间25.4%")
    print("• 刚刚获利了结，有回调买入机会")
    print("• 成长性优秀，营收增长10.6%")
    
    print("\n⚠️ 负面因素:")
    print("• 价格低于20日和50日均线")
    print("• 信号强度弱(6/7)")
    print("• 流动性评分偏低(45/100)")
    print("• 估值偏高(P/E 25.06)")
    print("• 短期偿债能力较低")
    
    print("\n🤔 我的独立思考:")
    print("-" * 25)
    
    print("🎯 Adobe的核心优势:")
    print("• 创意软件行业绝对龙头")
    print("• 订阅模式提供稳定现金流")
    print("• 护城河深厚，用户粘性强")
    print("• AI集成提升产品竞争力")
    print("• 财务健康，自由现金流优秀")
    
    print("\n⚠️ 我的担忧:")
    print("• 估值仍然偏高")
    print("• 技术面尚未企稳")
    print("• 竞争压力增加(Canva等)")
    print("• 宏观经济对软件行业影响")
    print("• 订阅增长可能放缓")
    
    print("\n📊 估值分析:")
    print("-" * 20)
    
    pe_ratio = info.get('trailingPE', 25.06)
    peg_ratio = info.get('pegRatio', 1.28)
    pb_ratio = info.get('priceToBook', 14.64)
    
    print(f"• P/E比率: {pe_ratio:.2f}")
    print(f"• PEG比率: {peg_ratio:.2f}")
    print(f"• 市净率: {pb_ratio:.2f}")
    
    if pe_ratio < 20:
        valuation_status = "相对合理"
    elif pe_ratio < 30:
        valuation_status = "偏高但可接受"
    else:
        valuation_status = "过高"
    
    print(f"• 估值状态: {valuation_status}")
    
    print("\n🎯 我的买入价格区间:")
    print("-" * 30)
    
    print("🔴 保守买入区间: $320-340")
    print("   • 从高点回调30-35%")
    print("   • P/E降至20以下")
    print("   • 技术面明确企稳")
    print("   • 风险收益比优秀")
    
    print("\n🟡 中性买入区间: $340-360")
    print("   • 从高点回调25-30%")
    print("   • P/E 20-25")
    print("   • 基本面确认")
    print("   • 风险收益比良好")
    
    print("\n🟢 激进买入区间: $360-380")
    print("   • 从高点回调20-25%")
    print("   • P/E 25-30")
    print("   • RSI超卖反弹")
    print("   • 风险收益比可接受")
    
    print("\n💭 我的真实想法:")
    print("-" * 25)
    
    print("✅ 我喜欢Adobe的原因:")
    print("• 商业模式优秀，订阅收入稳定")
    print("• 行业地位稳固，护城河深厚")
    print("• 财务表现强劲，现金流充足")
    print("• AI集成提升长期竞争力")
    print("• 分析师普遍看好")
    
    print("\n❌ 我不急于买入的原因:")
    print("• 当前价格$364.18仍然偏高")
    print("• 技术面尚未企稳")
    print("• 估值需要进一步消化")
    print("• 等待更好的买入机会")
    print("• 我的投资风格更保守")
    
    print("\n📈 我的具体买入计划:")
    print("-" * 30)
    print("💰 总资金分配: $10,000")
    print("📊 分批建仓策略:")
    print("• 第一档($340): 买入2股 - $680")
    print("• 第二档($320): 买入2股 - $640")
    print("• 第三档($300): 买入2股 - $600")
    print("• 预留资金: $8,080 (等待更好机会)")
    
    print("\n🎯 我的投资建议:")
    print("-" * 25)
    
    print("✅ 对于您的建议:")
    print("• $364.18的价格相对合理")
    print("• RSI超卖提供短期机会")
    print("• 基本面支撑长期上涨")
    print("• 可以考虑小仓位试探")
    
    print("\n🤔 我的个人观点:")
    print("• 我会等待$340以下买入")
    print("• 这是我的保守投资风格")
    print("• 不代表$364.18没有投资价值")
    print("• 不同投资者有不同的风险偏好")
    
    print("\n📊 风险收益分析:")
    print("-" * 25)
    
    # 计算不同情景下的收益
    scenarios = [
        (320, "悲观情景"),
        (380, "中性情景"),
        (420, "乐观情景"),
        (491, "分析师目标")
    ]
    
    print("📈 不同价格情景下的收益(基于$364.18):")
    for target_price, scenario in scenarios:
        profit = target_price - current_price
        return_percent = (profit / current_price) * 100
        print(f"• {scenario} (${target_price}): ${profit:+.2f} ({return_percent:+.1f}%)")
    
    print("\n💡 我的最终建议:")
    print("-" * 20)
    print("✅ 客观分析:")
    print("• Adobe是一家优秀的公司")
    print("• $364.18的价格具有投资价值")
    print("• 基本面支撑长期上涨")
    print("• 可以考虑买入")
    
    print("\n🤔 我的个人偏好:")
    print("• 我会等待$340以下买入")
    print("• 这是我的投资风格，不是客观标准")
    print("• 您的决策应该基于自己的风险承受能力")
    
    print("\n🎯 总结:")
    print("-" * 10)
    print("Adobe是一家优秀的公司，$364.18的价格相对合理。")
    print("我会等待更低的价格，但这是我的保守投资风格。")
    print("您的买入决策需要根据自己的风险偏好决定。")

if __name__ == "__main__":
    analyze_adobe_independent() 