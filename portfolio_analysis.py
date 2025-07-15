#!/usr/bin/env python3
"""
投资组合分析脚本
显示当前持仓状况、成本分析、风险分布等
"""

import json
import yfinance as yf
from datetime import datetime

def load_portfolio_config():
    """加载投资组合配置"""
    try:
        with open('personal_investor_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def get_current_prices(symbols):
    """获取当前价格"""
    prices = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.info.get('regularMarketPrice', 0)
            prices[symbol] = current_price
        except Exception as e:
            print(f"获取{symbol}价格失败: {e}")
            prices[symbol] = 0
    return prices

def analyze_portfolio():
    """分析投资组合"""
    config = load_portfolio_config()
    if not config:
        return
    
    portfolio = config['current_portfolio']
    detailed_holdings = portfolio.get('detailed_holdings', {})
    
    print("📊 投资组合分析")
    print("="*60)
    
    # 基本信息
    print(f"\n💰 基本信息:")
    print(f"  总资产: ${portfolio['total_assets']:,.2f}")
    print(f"  股票配置: {portfolio['stock_allocation']:.2f}%")
    print(f"  货币基金: {portfolio['fund_allocation']:.2f}%")
    
    # 持仓详情
    print(f"\n📈 持仓详情:")
    print(f"{'股票':<8} {'股数':<6} {'成本':<10} {'当前价':<10} {'盈亏':<10} {'占比':<8}")
    print("-" * 60)
    
    total_value = 0
    total_cost = 0
    
    for symbol, holding in detailed_holdings.items():
        shares = holding['shares']
        cost_basis = holding['cost_basis']
        allocation = holding['allocation']
        
        # 获取当前价格
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.info.get('regularMarketPrice', cost_basis)
        except:
            current_price = cost_basis
        
        # 计算盈亏
        cost_value = shares * cost_basis
        current_value = shares * current_price
        profit_loss = current_value - cost_value
        profit_loss_pct = (profit_loss / cost_value) * 100 if cost_value > 0 else 0
        
        total_value += current_value
        total_cost += cost_value
        
        print(f"{symbol:<8} {shares:<6} ${cost_basis:<9.2f} ${current_price:<9.2f} ${profit_loss:<9.2f} {allocation:<7.2f}%")
        
        # 显示盈亏状态
        if profit_loss > 0:
            print(f"  ✅ 盈利: {profit_loss_pct:.2f}%")
        elif profit_loss < 0:
            print(f"  ❌ 亏损: {profit_loss_pct:.2f}%")
        else:
            print(f"  📊 持平")
    
    # 总体分析
    print(f"\n📊 总体分析:")
    print(f"  总成本: ${total_cost:,.2f}")
    print(f"  当前市值: ${total_value:,.2f}")
    print(f"  总盈亏: ${total_value - total_cost:,.2f}")
    print(f"  总盈亏率: {((total_value - total_cost) / total_cost * 100):.2f}%" if total_cost > 0 else "0.00%")
    
    # 风险分析
    print(f"\n⚠️ 风险分析:")
    print(f"  最大单股占比: {max([h['allocation'] for h in detailed_holdings.values()]):.2f}%")
    print(f"  科技股占比: {sum([h['allocation'] for symbol, h in detailed_holdings.items() if symbol in ['GOOG', 'NVDA', 'AMD']]):.2f}%")
    print(f"  现金比例: {portfolio['fund_allocation']:.2f}%")
    
    # 投资建议
    print(f"\n💡 投资建议:")
    if portfolio['fund_allocation'] < 20:
        print(f"  ⚠️ 现金比例偏低，建议保持20%以上现金")
    else:
        print(f"  ✅ 现金比例合理，有足够资金应对机会")
    
    max_allocation = max([h['allocation'] for h in detailed_holdings.values()])
    if max_allocation > 20:
        print(f"  ⚠️ 最大单股占比({max_allocation:.2f}%)超过20%，建议分散风险")
    else:
        print(f"  ✅ 单股占比合理，风险分散良好")
    
    tech_allocation = sum([h['allocation'] for symbol, h in detailed_holdings.items() if symbol in ['GOOG', 'NVDA', 'AMD']])
    if tech_allocation > 60:
        print(f"  ⚠️ 科技股占比({tech_allocation:.2f}%)过高，建议增加其他行业")
    else:
        print(f"  ✅ 行业配置相对均衡")
    
    print("="*60)

if __name__ == "__main__":
    analyze_portfolio() 