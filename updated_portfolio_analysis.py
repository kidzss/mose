#!/usr/bin/env python3
"""
更新后的投资组合分析
基于最新持仓信息重新评估风险和配置建议
"""

def analyze_updated_portfolio():
    """分析更新后的投资组合"""
    
    print("📊 最新投资组合分析")
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
    
    # 具体持仓分析
    print("📈 具体持仓分析:")
    print("-" * 25)
    
    # 股票持仓详情
    holdings = {
        'NVDA': {'percentage': 21.0, 'amount': total_assets * 21.0 / 100},
        'AMD': {'percentage': 18.3, 'amount': total_assets * 18.3 / 100},
        'GOOGL': {'percentage': 18.83, 'amount': total_assets * 18.83 / 100},
        'PFE': {'percentage': 6.89, 'amount': total_assets * 6.89 / 100},
        'TSLA': {'percentage': 4.65, 'amount': total_assets * 4.65 / 100}
    }
    
    # 计算当前股价和股数（估算）
    current_prices = {
        'NVDA': 145.48,
        'AMD': 126.79,
        'GOOGL': 173.32,
        'PFE': 26.85,
        'TSLA': 196.00  # 基于您之前的信息
    }
    
    print("股票持仓详情:")
    total_stock_weight = 0
    for symbol, holding in holdings.items():
        estimated_shares = holding['amount'] / current_prices[symbol]
        total_stock_weight += holding['percentage']
        print(f"   {symbol}: {holding['percentage']}% (${holding['amount']:,.2f}) - 约{estimated_shares:.0f}股")
    
    print(f"   股票总权重: {total_stock_weight:.2f}%")
    print()
    
    # 风险分析
    print("⚠️ 风险分析:")
    print("-" * 15)
    
    # 集中度风险
    top3_concentration = holdings['NVDA']['percentage'] + holdings['AMD']['percentage'] + holdings['GOOGL']['percentage']
    tech_concentration = top3_concentration + holdings['TSLA']['percentage']  # 科技股集中度
    
    print(f"集中度风险:")
    print(f"   前三大持仓: {top3_concentration:.1f}% ", end="")
    if top3_concentration > 60:
        print("🔴 极高风险")
    elif top3_concentration > 50:
        print("🟡 高风险")
    else:
        print("🟢 中等风险")
    
    print(f"   科技股集中度: {tech_concentration:.1f}% ", end="")
    if tech_concentration > 70:
        print("🔴 极高风险")
    elif tech_concentration > 60:
        print("🟡 高风险")
    else:
        print("🟢 中等风险")
    
    # 单股权重风险
    print(f"单股权重风险:")
    high_weight_stocks = []
    for symbol, holding in holdings.items():
        if holding['percentage'] > 20:
            high_weight_stocks.append(f"{symbol}({holding['percentage']:.1f}%)")
            print(f"   {symbol}: {holding['percentage']:.1f}% 🔴 超重")
        elif holding['percentage'] > 15:
            print(f"   {symbol}: {holding['percentage']:.1f}% 🟡 偏重")
        else:
            print(f"   {symbol}: {holding['percentage']:.1f}% 🟢 合理")
    
    if high_weight_stocks:
        print(f"   ⚠️ 超重股票: {', '.join(high_weight_stocks)}")
    print()
    
    # 流动性分析
    liquid_assets = cash_amount + funds_amount
    liquid_percentage = (liquid_assets / total_assets) * 100
    
    print("💧 流动性分析:")
    print(f"   流动性资产: ${liquid_assets:,.2f} ({liquid_percentage:.1f}%)")
    if liquid_percentage > 30:
        print("   流动性状态: 🟢 优秀 - 有充足资金应对机会和风险")
    elif liquid_percentage > 20:
        print("   流动性状态: 🟡 良好 - 基本够用")
    else:
        print("   流动性状态: 🔴 偏低 - 建议增加现金比例")
    print()
    
    # 与之前对比
    print("📊 与之前配置对比:")
    print("-" * 25)
    print("积极变化:")
    print("   ✅ 现金比例提升至8.38%（之前约3%）")
    print("   ✅ 股票总权重降至70.02%（之前约78%）")
    print("   ✅ 整体风险有所降低")
    print()
    
    print("仍需改进:")
    print("   🔴 NVDA仍然超重(21.0%)")
    print("   🔴 前三大持仓仍然过于集中(58.13%)")
    print("   🔴 科技股依然占主导(62.78%)")
    print()
    
    # 基于混合策略的建议调整
    print("🎯 基于混合策略的调整建议:")
    print("-" * 35)
    
    # 核心资产目标配置 (75%)
    core_target = total_assets * 0.75
    satellite_target = total_assets * 0.15  # 卫星资产 (15%)
    cash_target = total_assets * 0.10       # 现金储备 (10%)
    
    print(f"目标配置:")
    print(f"   💰 核心资产: ${core_target:,.2f} (75%)")
    print(f"   🚀 卫星资产: ${satellite_target:,.2f} (15%)")
    print(f"   💵 现金储备: ${cash_target:,.2f} (10%)")
    print()
    
    # 具体调整建议
    print("具体调整建议:")
    
    # 核心资产配置（每只股票不超过15%）
    core_per_stock = total_assets * 0.15  # 15%上限
    
    adjustments = {}
    
    for symbol, holding in holdings.items():
        current_amount = holding['amount']
        if symbol in ['NVDA', 'AMD', 'GOOGL']:  # 主要持仓
            target_amount = core_per_stock
            if current_amount > target_amount:
                reduction = current_amount - target_amount
                adjustments[symbol] = {
                    'action': 'reduce',
                    'current_amount': current_amount,
                    'target_amount': target_amount,
                    'change_amount': -reduction,
                    'current_shares': int(current_amount / current_prices[symbol]),
                    'target_shares': int(target_amount / current_prices[symbol]),
                    'change_shares': int(target_amount / current_prices[symbol]) - int(current_amount / current_prices[symbol])
                }
        else:  # PFE和TSLA保持当前水平
            adjustments[symbol] = {
                'action': 'maintain',
                'current_amount': current_amount,
                'target_amount': current_amount,
                'change_amount': 0
            }
    
    total_reduction = 0
    print("调整方案:")
    for symbol, adj in adjustments.items():
        if adj['action'] == 'reduce':
            print(f"   {symbol}: 减持{-adj['change_shares']}股 (${-adj['change_amount']:,.2f})")
            total_reduction += -adj['change_amount']
        else:
            print(f"   {symbol}: 保持当前持仓")
    
    print(f"   总释放资金: ${total_reduction:,.2f}")
    print()
    
    # 卫星资产建议
    current_satellite_candidates = holdings['TSLA']['amount']  # TSLA可以作为卫星资产
    additional_satellite_needed = satellite_target - current_satellite_candidates
    
    print("🚀 卫星资产配置:")
    print(f"   当前候选: TSLA (${holdings['TSLA']['amount']:,.2f})")
    print(f"   目标总额: ${satellite_target:,.2f}")
    print(f"   还需投入: ${additional_satellite_needed:,.2f}")
    print(f"   建议: 选择1-2只高成长股票，采用您的TSLA成功策略")
    print()
    
    # 执行优先级
    print("📋 执行优先级:")
    print("-" * 15)
    print("第一优先级（立即执行）:")
    print("   1. NVDA减持约8股 (21%→15%)")
    print("   2. AMD减持约3股 (18.3%→15%)")
    print("   3. GOOGL减持约6股 (18.83%→15%)")
    print("   4. 释放约$3,000现金")
    print()
    
    print("第二优先级（择机执行）:")
    print("   1. 选择新的卫星资产标的")
    print("   2. 投入约$3,000建立卫星仓位")
    print("   3. 保持10%现金储备")
    print()
    
    # 风险收益预期
    print("📈 调整后预期:")
    print("-" * 20)
    print("风险降低:")
    print("   • 单股最大权重: 21%→15%")
    print("   • 前三大集中度: 58.13%→45%")
    print("   • 整体风险等级: 高→中等")
    print()
    
    print("收益预期:")
    print("   • 核心资产: 年化10-15%")
    print("   • 卫星资产: 年化30-50%（或全亏）")
    print("   • 综合预期: 年化12-18%")
    print()
    
    return {
        'total_assets': total_assets,
        'adjustments': adjustments,
        'total_reduction': total_reduction,
        'risk_level': 'medium_high'
    }

if __name__ == "__main__":
    result = analyze_updated_portfolio() 