#!/usr/bin/env python3
"""
修正后的投资组合分析
使用准确的股数信息重新计算
"""

def analyze_corrected_portfolio():
    """分析修正后的投资组合"""
    
    print("📊 修正后的投资组合分析")
    print("=" * 50)
    
    # 最新资产状况
    total_assets = 27714.64
    cash_percentage = 8.38
    stocks_percentage = 70.02
    funds_percentage = 21.6
    
    cash_amount = total_assets * cash_percentage / 100
    stocks_amount = total_assets * stocks_percentage / 100
    funds_amount = total_assets * funds_percentage / 100
    
    print("💰 最新资产配置:")
    print(f"   总资产: ${total_assets:,.2f}")
    print(f"   现金: ${cash_amount:,.2f} ({cash_percentage}%)")
    print(f"   股票: ${stocks_amount:,.2f} ({stocks_percentage}%)")
    print(f"   基金: ${funds_amount:,.2f} ({funds_percentage}%)")
    print()
    
    # 修正后的具体持仓分析
    print("📈 修正后的持仓分析:")
    print("-" * 30)
    
    # 准确的股票持仓详情
    holdings = {
        'NVDA': {'percentage': 21.0, 'shares': 40, 'amount': total_assets * 21.0 / 100},
        'AMD': {'percentage': 18.3, 'shares': 40, 'amount': total_assets * 18.3 / 100},
        'GOOG': {'percentage': 18.83, 'shares': 30, 'amount': total_assets * 18.83 / 100},
        'PFE': {'percentage': 6.89, 'shares': 80, 'amount': total_assets * 6.89 / 100},
        'TSLA': {'percentage': 4.65, 'shares': 4, 'amount': total_assets * 4.65 / 100}
    }
    
    # 根据实际股数计算当前股价
    current_prices = {}
    for symbol, holding in holdings.items():
        current_prices[symbol] = holding['amount'] / holding['shares']
    
    print("股票持仓详情（修正后）:")
    total_stock_weight = 0
    for symbol, holding in holdings.items():
        total_stock_weight += holding['percentage']
        print(f"   {symbol}: {holding['percentage']}% (${holding['amount']:,.2f}) - {holding['shares']}股 @ ${current_prices[symbol]:.2f}")
    
    print(f"   股票总权重: {total_stock_weight:.2f}%")
    print()
    
    # 重新计算当前股价
    print("📊 当前股价分析:")
    print("-" * 20)
    for symbol, price in current_prices.items():
        print(f"   {symbol}: ${price:.2f}/股")
    print()
    
    # 基于混合策略的精确调整建议
    print("🎯 精确调整建议:")
    print("-" * 20)
    
    # 目标配置 (每只主要股票15%上限)
    target_weight = 15.0  # 15%
    target_amount_per_stock = total_assets * target_weight / 100
    
    print(f"目标配置（每只主要股票≤{target_weight}%）:")
    
    total_reduction = 0
    adjustments = {}
    
    for symbol, holding in holdings.items():
        current_amount = holding['amount']
        current_shares = holding['shares']
        current_price = current_prices[symbol]
        
        if symbol in ['NVDA', 'AMD', 'GOOGL']:  # 主要持仓需要调整
            if current_amount > target_amount_per_stock:
                target_shares = int(target_amount_per_stock / current_price)
                shares_to_sell = current_shares - target_shares
                reduction_amount = shares_to_sell * current_price
                
                adjustments[symbol] = {
                    'current_shares': current_shares,
                    'target_shares': target_shares,
                    'shares_to_sell': shares_to_sell,
                    'reduction_amount': reduction_amount,
                    'current_weight': holding['percentage'],
                    'target_weight': (target_shares * current_price / total_assets) * 100
                }
                total_reduction += reduction_amount
                
                print(f"   {symbol}: {current_shares}股 → {target_shares}股 (减持{shares_to_sell}股)")
                print(f"      权重: {holding['percentage']:.1f}% → {adjustments[symbol]['target_weight']:.1f}%")
                print(f"      释放资金: ${reduction_amount:,.2f}")
        else:
            print(f"   {symbol}: 保持{current_shares}股不变")
        print()
    
    print(f"总释放资金: ${total_reduction:,.2f}")
    print()
    
    # 卫星资产策略
    print("🚀 卫星资产策略:")
    print("-" * 20)
    
    satellite_target = total_assets * 0.15  # 15%卫星资产
    current_tsla_amount = holdings['TSLA']['amount']
    
    print(f"目标卫星资产总额: ${satellite_target:,.2f} (15%)")
    print(f"当前TSLA价值: ${current_tsla_amount:,.2f}")
    print(f"还需投入: ${satellite_target - current_tsla_amount:,.2f}")
    print()
    print("建议:")
    print("   • 保留TSLA 4股作为卫星资产核心")
    print("   • 用释放的资金选择1-2只高成长股票")
    print("   • 采用您成功的TSLA操作策略")
    print("   • 心理预期：这部分资金可能全亏")
    print()
    
    # 流动性分析
    liquid_assets = cash_amount + funds_amount
    liquid_percentage = (liquid_assets / total_assets) * 100
    
    print("💧 流动性优势:")
    print("-" * 15)
    print(f"当前流动性资产: ${liquid_assets:,.2f} ({liquid_percentage:.1f}%)")
    print("优势分析:")
    print("   ✅ 货币基金T+0赎回，等同现金")
    print("   ✅ 总流动性资产接近30%")
    print("   ✅ 可随时应对市场机会和风险")
    print("   ✅ 为卫星资产投资提供充足弹药")
    print()
    
    # 执行计划
    print("🎯 具体执行计划:")
    print("-" * 20)
    
    print("第一步：减持超重股票")
    for symbol, adj in adjustments.items():
        print(f"   • {symbol}: 卖出{adj['shares_to_sell']}股")
        print(f"     预计获得: ${adj['reduction_amount']:,.2f}")
    
    print(f"\n第二步：资金配置")
    print(f"   • 释放资金总计: ${total_reduction:,.2f}")
    print(f"   • 暂时保持现金状态")
    print(f"   • 等待右侧交易信号")
    
    print(f"\n第三步：建立卫星仓位")
    print(f"   • 选择高成长标的")
    print(f"   • 投资金额: ${satellite_target - current_tsla_amount:,.2f}")
    print(f"   • 采用集中投资策略")
    print()
    
    # 风险收益预期
    print("📈 调整后预期:")
    print("-" * 20)
    print("风险控制:")
    print("   • 单股最大权重: 21% → 15%")
    print("   • 前三大集中度: 58.1% → 45%")
    print("   • 科技股集中度: 62.8% → 50%左右")
    print("   • 整体风险等级: 高 → 中等")
    print()
    
    print("收益预期:")
    print("   • 核心资产(75%): 年化10-15%")
    print("   • 卫星资产(15%): 年化30-50%或全亏")
    print("   • 现金储备(10%): 年化4%")
    print("   • 综合预期: 年化12-18%")
    print()
    
    # 心理建设
    print("🧠 心理建设:")
    print("-" * 15)
    print("这样调整的好处:")
    print("   ✅ 保持了75%的稳健投资基础")
    print("   ✅ 用15%满足您的暴富心理")
    print("   ✅ 即使卫星资产全亏也不伤筋骨")
    print("   ✅ 核心资产继续稳定增长")
    print("   ✅ 睡眠质量和投资心态都会更好")
    
    return {
        'adjustments': adjustments,
        'total_reduction': total_reduction,
        'current_prices': current_prices
    }

if __name__ == "__main__":
    result = analyze_corrected_portfolio() 