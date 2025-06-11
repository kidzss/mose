#!/usr/bin/env python3
"""
CRM (Salesforce) 基本面与市场环境深度分析
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_crm_fundamentals():
    """CRM基本面分析"""
    print("🏢 CRM (Salesforce) 基本面深度分析")
    print("="*60)
    
    ticker = yf.Ticker('CRM')
    info = ticker.info
    
    # 1. 公司基本信息
    print("📋 公司概况:")
    print(f"   公司名称: {info.get('longName', 'N/A')}")
    print(f"   行业: {info.get('industry', 'N/A')}")
    print(f"   市值: ${info.get('marketCap', 0)/1e9:.1f}B")
    print(f"   员工数: {info.get('fullTimeEmployees', 'N/A'):,}")
    
    # 2. 财务健康度分析
    print(f"\n💰 财务健康度:")
    
    # 盈利能力
    revenue = info.get('totalRevenue', 0)
    gross_margin = info.get('grossMargins', 0) * 100
    profit_margin = info.get('profitMargins', 0) * 100
    roe = info.get('returnOnEquity', 0) * 100
    roa = info.get('returnOnAssets', 0) * 100
    
    print(f"   年收入: ${revenue/1e9:.1f}B")
    print(f"   毛利率: {gross_margin:.1f}%")
    print(f"   净利率: {profit_margin:.1f}%")
    print(f"   ROE: {roe:.1f}%")
    print(f"   ROA: {roa:.1f}%")
    
    # 财务稳定性
    debt_to_equity = info.get('debtToEquity', 0)
    current_ratio = info.get('currentRatio', 0)
    cash = info.get('totalCash', 0)
    debt = info.get('totalDebt', 0)
    
    print(f"\n⚖️ 财务稳定性:")
    print(f"   负债权益比: {debt_to_equity:.2f}")
    print(f"   流动比率: {current_ratio:.2f}")
    print(f"   现金: ${cash/1e9:.1f}B")
    print(f"   总债务: ${debt/1e9:.1f}B")
    
    # 3. 成长性分析
    print(f"\n📈 成长性指标:")
    revenue_growth = info.get('revenueGrowth', 0) * 100
    earnings_growth = info.get('earningsGrowthRate', 0) * 100
    
    print(f"   营收增长率: {revenue_growth:.1f}%")
    print(f"   盈利增长率: {earnings_growth:.1f}%")
    
    # 4. 估值分析
    print(f"\n💸 估值指标:")
    pe_ratio = info.get('trailingPE', 0)
    forward_pe = info.get('forwardPE', 0)
    pb_ratio = info.get('priceToBook', 0)
    ps_ratio = info.get('priceToSalesTrailing12Months', 0)
    peg_ratio = info.get('pegRatio', 0)
    
    print(f"   P/E 比率: {pe_ratio:.2f}")
    print(f"   前瞻P/E: {forward_pe:.2f}")
    print(f"   P/B 比率: {pb_ratio:.2f}")
    print(f"   P/S 比率: {ps_ratio:.2f}")
    print(f"   PEG 比率: {peg_ratio:.2f}")
    
    # 基本面评分
    fundamentals_score = calculate_fundamental_score({
        'profit_margin': profit_margin,
        'roe': roe,
        'revenue_growth': revenue_growth,
        'debt_to_equity': debt_to_equity,
        'pe_ratio': pe_ratio,
        'current_ratio': current_ratio
    })
    
    return {
        'revenue': revenue,
        'profit_margin': profit_margin,
        'roe': roe,
        'revenue_growth': revenue_growth,
        'pe_ratio': pe_ratio,
        'debt_to_equity': debt_to_equity,
        'fundamentals_score': fundamentals_score
    }

def analyze_saas_industry():
    """SaaS行业分析"""
    print(f"\n🏭 SaaS行业环境分析")
    print("="*40)
    
    # 获取同行对比数据
    competitors = ['CRM', 'MSFT', 'ORCL', 'ADBE', 'NOW']
    
    print("📊 同行对比 (P/E比率):")
    pe_ratios = {}
    
    for symbol in competitors:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            pe = info.get('trailingPE', 0)
            pe_ratios[symbol] = pe
            print(f"   {symbol}: {pe:.2f}")
        except:
            print(f"   {symbol}: 获取失败")
    
    # CRM在行业中的位置
    if 'CRM' in pe_ratios:
        crm_pe = pe_ratios['CRM']
        avg_pe = np.mean([pe for pe in pe_ratios.values() if pe > 0])
        
        print(f"\n📈 行业分析:")
        print(f"   CRM P/E: {crm_pe:.2f}")
        print(f"   行业平均: {avg_pe:.2f}")
        
        if crm_pe < avg_pe:
            valuation_status = "🟢 相对便宜"
        elif crm_pe < avg_pe * 1.2:
            valuation_status = "🟡 合理估值"
        else:
            valuation_status = "🔴 相对昂贵"
        
        print(f"   估值状态: {valuation_status}")
    
    # 行业趋势
    print(f"\n🌐 SaaS行业趋势:")
    print("   ✅ 数字化转型持续")
    print("   ✅ 云计算需求增长")
    print("   ✅ 企业软件订阅模式成熟")
    print("   ⚠️  竞争激烈")
    print("   ⚠️  利率敏感（科技股特征）")
    
    return pe_ratios

def analyze_market_environment():
    """当前市场环境分析"""
    print(f"\n🌍 市场环境分析")
    print("="*40)
    
    # 获取大盘数据
    spy = yf.Ticker('SPY')
    qqq = yf.Ticker('QQQ')
    
    spy_hist = spy.history(period='3mo')
    qqq_hist = qqq.history(period='3mo')
    
    # 大盘趋势
    spy_return_3m = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
    qqq_return_3m = (qqq_hist['Close'].iloc[-1] / qqq_hist['Close'].iloc[0] - 1) * 100
    
    print(f"📊 大盘表现 (3个月):")
    print(f"   SPY: {spy_return_3m:+.2f}%")
    print(f"   QQQ (科技股): {qqq_return_3m:+.2f}%")
    
    # VIX恐慌指数
    try:
        vix = yf.Ticker('^VIX')
        vix_current = vix.history(period='2d')['Close'].iloc[-1]
        print(f"   VIX恐慌指数: {vix_current:.2f}")
        
        if vix_current < 20:
            market_sentiment = "🟢 低波动，相对平静"
        elif vix_current < 30:
            market_sentiment = "🟡 中等波动"
        else:
            market_sentiment = "🔴 高波动，市场恐慌"
        
        print(f"   市场情绪: {market_sentiment}")
    except:
        print("   VIX: 获取失败")
    
    # 利率环境 (通过TLT债券ETF反映)
    try:
        tlt = yf.Ticker('TLT')
        tlt_hist = tlt.history(period='1mo')
        tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[0] - 1) * 100
        
        print(f"   长期债券(TLT): {tlt_return:+.2f}%")
        
        if tlt_return > 2:
            rate_environment = "🟢 利率下行，利好成长股"
        elif tlt_return > -2:
            rate_environment = "🟡 利率稳定"
        else:
            rate_environment = "🔴 利率上行，压制成长股"
        
        print(f"   利率环境: {rate_environment}")
    except:
        print("   利率环境: 获取失败")
    
    return {
        'spy_return_3m': spy_return_3m,
        'qqq_return_3m': qqq_return_3m
    }

def calculate_fundamental_score(metrics):
    """计算基本面综合评分"""
    score = 0
    max_score = 100
    
    # 盈利能力 (30分)
    if metrics['profit_margin'] > 15:
        score += 30
    elif metrics['profit_margin'] > 10:
        score += 20
    elif metrics['profit_margin'] > 5:
        score += 10
    
    # 成长性 (25分)
    if metrics['revenue_growth'] > 10:
        score += 25
    elif metrics['revenue_growth'] > 5:
        score += 15
    elif metrics['revenue_growth'] > 0:
        score += 10
    
    # 财务健康 (25分)
    if metrics['debt_to_equity'] < 30:
        score += 15
    elif metrics['debt_to_equity'] < 50:
        score += 10
    
    if metrics['current_ratio'] > 1:
        score += 10
    
    # 估值合理性 (20分)
    if metrics['pe_ratio'] < 30:
        score += 20
    elif metrics['pe_ratio'] < 50:
        score += 10
    
    return score

def generate_comprehensive_opinion():
    """生成综合投资观点"""
    print(f"\n🎯 CRM综合投资观点")
    print("="*50)
    
    # 运行所有分析
    fundamentals = analyze_crm_fundamentals()
    industry_data = analyze_saas_industry()
    market_env = analyze_market_environment()
    
    # 综合评分
    total_score = fundamentals['fundamentals_score']
    
    print(f"\n📊 综合评分: {total_score}/100")
    
    if total_score >= 80:
        grade = "🟢 优秀"
        investment_view = "强烈看好"
    elif total_score >= 60:
        grade = "🟡 良好"
        investment_view = "谨慎看好"
    elif total_score >= 40:
        grade = "🟠 一般"
        investment_view = "中性观望"
    else:
        grade = "🔴 较差"
        investment_view = "不建议投资"
    
    print(f"   评级: {grade}")
    print(f"   投资观点: {investment_view}")
    
    # 风险提示
    print(f"\n⚠️ 主要风险:")
    
    risks = []
    if fundamentals['pe_ratio'] > 40:
        risks.append("估值偏高，PE比率超过40")
    
    if market_env.get('qqq_return_3m', 0) < -10:
        risks.append("科技股整体表现疲弱")
    
    if fundamentals['revenue_growth'] < 5:
        risks.append("收入增长放缓")
    
    if not risks:
        risks.append("总体风险可控")
    
    for risk in risks:
        print(f"   - {risk}")
    
    # 针对用户的建议
    print(f"\n💡 针对你的投资风格建议:")
    print(f"   - 作为不熟悉股票，建议仓位: 5-10%")
    print(f"   - 分2-3批建仓，降低时机风险")
    print(f"   - 密切关注季报表现")
    print(f"   - 设置止损位$255，严格执行")
    
    return {
        'total_score': total_score,
        'grade': grade,
        'investment_view': investment_view,
        'risks': risks
    }

def main():
    """主函数"""
    print("🚀 开始CRM基本面与市场环境分析")
    print("="*60)
    print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = generate_comprehensive_opinion()
    
    print(f"\n✅ 分析完成！")
    
if __name__ == "__main__":
    main() 