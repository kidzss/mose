#!/usr/bin/env python3
"""
市场情景推演分析脚本
基于当前投资组合，分析标普500不同走势下的应对策略
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_sp500_current():
    """获取标普500当前价格"""
    try:
        spy = yf.Ticker('^GSPC')
        current_price = spy.info.get('regularMarketPrice', 0)
        return current_price
    except Exception as e:
        print(f"获取标普500数据失败: {e}")
        return 5600  # 默认值

def load_portfolio_config():
    """加载投资组合配置"""
    try:
        with open('personal_investor_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def analyze_portfolio_beta():
    """分析投资组合的Beta值"""
    portfolio_stocks = ['GOOG', 'AMD', 'NVDA', 'MRK', 'BRK-B']
    betas = {}
    
    for symbol in portfolio_stocks:
        try:
            stock = yf.Ticker(symbol)
            # 获取Beta值，如果没有则估算
            beta = stock.info.get('beta', 1.0)
            betas[symbol] = beta
        except:
            betas[symbol] = 1.0
    
    return betas

def calculate_portfolio_beta(portfolio_config, stock_betas):
    """计算投资组合加权Beta"""
    detailed_holdings = portfolio_config['current_portfolio']['detailed_holdings']
    total_allocation = sum([h['allocation'] for h in detailed_holdings.values()])
    
    weighted_beta = 0
    for symbol, holding in detailed_holdings.items():
        allocation = holding['allocation']
        beta = stock_betas.get(symbol, 1.0)
        weighted_beta += (allocation / total_allocation) * beta
    
    return weighted_beta

def analyze_market_scenarios():
    """分析不同市场情景"""
    print("🌍 市场情景推演分析")
    print("="*60)
    
    # 获取当前标普500价格
    current_sp500 = get_sp500_current()
    print(f"\n📊 当前市场状况:")
    print(f"  标普500当前价格: {current_sp500:.2f}")
    
    # 加载投资组合配置
    config = load_portfolio_config()
    if not config:
        return
    
    portfolio = config['current_portfolio']
    detailed_holdings = portfolio.get('detailed_holdings', {})
    
    # 分析投资组合Beta
    stock_betas = analyze_portfolio_beta()
    portfolio_beta = calculate_portfolio_beta(config, stock_betas)
    
    print(f"  投资组合Beta: {portfolio_beta:.2f}")
    print(f"  总资产: ${portfolio['total_assets']:,.2f}")
    print(f"  股票配置: {portfolio['stock_allocation']:.2f}%")
    print(f"  现金比例: {portfolio['fund_allocation']:.2f}%")
    
    # 情景分析
    scenarios = [
        {"name": "继续拉高", "sp500": 6350, "change": (6350 - current_sp500) / current_sp500 * 100},
        {"name": "小幅回调", "sp500": 5700, "change": (5700 - current_sp500) / current_sp500 * 100},
        {"name": "中等回调", "sp500": 5500, "change": (5500 - current_sp500) / current_sp500 * 100},
        {"name": "深度回调", "sp500": 5300, "change": (5300 - current_sp500) / current_sp500 * 100}
    ]
    
    print(f"\n📈 情景推演分析:")
    print("="*60)
    
    for scenario in scenarios:
        print(f"\n🎯 情景: {scenario['name']}")
        print(f"  标普500目标: {scenario['sp500']:.0f}")
        print(f"  标普500变化: {scenario['change']:+.2f}%")
        
        # 估算投资组合表现
        portfolio_change = scenario['change'] * portfolio_beta
        portfolio_value_change = portfolio['total_assets'] * (portfolio['stock_allocation'] / 100) * (portfolio_change / 100)
        
        print(f"  投资组合预期变化: {portfolio_change:+.2f}%")
        print(f"  股票部分价值变化: ${portfolio_value_change:+,.2f}")
        
        # 分析各股票表现
        print(f"  📊 个股预期表现:")
        for symbol, holding in detailed_holdings.items():
            beta = stock_betas.get(symbol, 1.0)
            stock_change = scenario['change'] * beta
            shares = holding['shares']
            cost_basis = holding['cost_basis']
            current_value = shares * cost_basis
            value_change = current_value * (stock_change / 100)
            
            print(f"    {symbol}: {stock_change:+.2f}% (Beta: {beta:.2f})")
            print(f"      价值变化: ${value_change:+,.2f}")
        
        # 应对策略
        print(f"  💡 应对策略:")
        if scenario['change'] > 10:  # 大幅上涨
            print(f"    ✅ 市场强势，可考虑:")
            print(f"      - 持有现有仓位，享受上涨")
            print(f"      - 考虑部分获利了结")
            print(f"      - 关注高估值风险")
        elif scenario['change'] > 0:  # 小幅上涨
            print(f"    📊 市场温和上涨，建议:")
            print(f"      - 继续持有")
            print(f"      - 关注个股基本面")
            print(f"      - 保持现金比例")
        elif scenario['change'] > -5:  # 小幅回调
            print(f"    ⚠️ 市场小幅回调，建议:")
            print(f"      - 保持冷静，不要恐慌")
            print(f"      - 关注优质股票加仓机会")
            print(f"      - 利用现金分批建仓")
        elif scenario['change'] > -10:  # 中等回调
            print(f"    🔶 市场中等回调，建议:")
            print(f"      - 严格执行分批建仓计划")
            print(f"      - 重点关注科技股机会")
            print(f"      - 考虑增加防御性股票")
        else:  # 深度回调
            print(f"    🔴 市场深度回调，建议:")
            print(f"      - 保持充足现金")
            print(f"      - 等待恐慌情绪释放")
            print(f"      - 关注TSLA等波段机会")
            print(f"      - 考虑对冲策略")

def analyze_risk_management():
    """风险管理分析"""
    print(f"\n⚠️ 风险管理策略:")
    print("="*60)
    
    config = load_portfolio_config()
    if not config:
        return
    
    portfolio = config['current_portfolio']
    
    print(f"  📊 当前风险状况:")
    print(f"    现金比例: {portfolio['fund_allocation']:.2f}%")
    print(f"    股票配置: {portfolio['stock_allocation']:.2f}%")
    
    # 风险控制建议
    print(f"  💡 风险控制建议:")
    
    if portfolio['fund_allocation'] >= 30:
        print(f"    ✅ 现金充足，有足够缓冲")
        print(f"    ✅ 可以应对市场回调")
    else:
        print(f"    ⚠️ 现金比例偏低，建议增加现金")
    
    # 分批建仓策略
    print(f"  📈 分批建仓策略:")
    print(f"    - 第一档: 标普回调5-8%时，加仓10%")
    print(f"    - 第二档: 标普回调10-15%时，加仓15%")
    print(f"    - 第三档: 标普回调15%以上时，加仓20%")
    
    # 止损策略
    print(f"  🛡️ 止损策略:")
    print(f"    - 单股止损: -15%")
    print(f"    - 组合止损: -20%")
    print(f"    - 市场止损: 标普跌破关键支撑")

def analyze_opportunities():
    """机会分析"""
    print(f"\n🎯 投资机会分析:")
    print("="*60)
    
    print(f"  📈 当前机会:")
    print(f"    - TSLA回调至$270-$235区间")
    print(f"    - 科技股整体回调机会")
    print(f"    - 大盘回调时的优质股票")
    
    print(f"  💰 资金使用计划:")
    print(f"    - 33.63%现金可用于加仓")
    print(f"    - 重点关注TSLA波段机会")
    print(f"    - 考虑增加防御性股票")
    
    print(f"  ⏰ 时机把握:")
    print(f"    - 等待技术面确认信号")
    print(f"    - 关注市场情绪指标")
    print(f"    - 分批建仓，不要一次性重仓")

def main():
    """主函数"""
    print("🚀 市场情景推演分析开始")
    print("="*60)
    
    # 1. 市场情景分析
    analyze_market_scenarios()
    
    # 2. 风险管理分析
    analyze_risk_management()
    
    # 3. 机会分析
    analyze_opportunities()
    
    # 4. 综合建议
    print(f"\n🎯 综合建议:")
    print("="*60)
    print(f"  📊 当前策略:")
    print(f"    - 保持现有仓位，等待机会")
    print(f"    - 利用33.63%现金分批建仓")
    print(f"    - 重点关注TSLA回调机会")
    print(f"    - 严格执行风险控制")
    
    print(f"  ⚠️ 风险提示:")
    print(f"    - 市场估值偏高，回调风险存在")
    print(f"    - 科技股波动性较大")
    print(f"    - 需要耐心等待机会")
    print(f"    - 不要追高，分批建仓")
    
    print("="*60)
    print("✅ 分析完成")
    print("="*60)

if __name__ == "__main__":
    main() 