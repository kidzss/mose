import yfinance as yf
import pandas as pd
from datetime import datetime

class UpdatedPortfolioStatus:
    """更新后的投资组合状况分析"""
    
    def __init__(self):
        # 更新后的持仓 - AMD已减持5股
        self.current_positions = {
            'AMD': {'shares': 35, 'avg_price': 128.9, 'current_weight': 16.36},
            'GOOG': {'shares': 30, 'price': 170.27},
            'NVDA': {'shares': 35, 'price': 144.13},
            'PFE': {'shares': 80, 'price': 23.84}
        }
        
        # 估算总资产（基于AMD权重反推）
        amd_value = 35 * 128.9  # $4,511.5
        self.estimated_total_assets = amd_value / (16.36 / 100)  # ~$27,578
        
        # AMD减持收益
        self.amd_sale_proceeds = 5 * 128.9  # $644.5
        
        # 目标配置保持不变
        self.target_allocation = {
            'NVDA': 15.0, 'GOOG': 12.0, 'AMD': 10.0,
            'MSFT': 10.0, 'META': 8.0, 'TSLA': 5.0, 'PLTR': 3.0,
            'JPM': 8.0, 'BRK-B': 8.0, 'MRK': 6.0, 'PFE': 8.0
        }
    
    def calculate_current_portfolio(self):
        """计算当前投资组合状况"""
        print("📊 更新后的投资组合分析")
        print("=" * 60)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"预估总资产: ${self.estimated_total_assets:,.0f}")
        
        print(f"\n✅ AMD减持操作已完成:")
        print(f"  减持数量: 5股 @ $128.9")
        print(f"  减持收益: ${self.amd_sale_proceeds:,.1f}")
        print(f"  剩余持仓: 35股")
        print(f"  当前权重: 16.36%")
        
        # 计算其他股票的当前权重
        current_portfolio = {}
        
        # AMD (已知权重)
        amd_value = 35 * 128.9
        current_portfolio['AMD'] = {
            'shares': 35,
            'value': amd_value,
            'weight': 16.36,
            'target_weight': 10.0,
            'status': '已部分减仓'
        }
        
        # 其他股票（估算）
        goog_value = 30 * 170.27
        goog_weight = (goog_value / self.estimated_total_assets) * 100
        current_portfolio['GOOG'] = {
            'shares': 30,
            'value': goog_value,
            'weight': goog_weight,
            'target_weight': 12.0,
            'status': '需减仓'
        }
        
        nvda_value = 35 * 144.13
        nvda_weight = (nvda_value / self.estimated_total_assets) * 100
        current_portfolio['NVDA'] = {
            'shares': 35,
            'value': nvda_value,
            'weight': nvda_weight,
            'target_weight': 15.0,
            'status': '需小幅减仓'
        }
        
        pfe_value = 80 * 23.84
        pfe_weight = (pfe_value / self.estimated_total_assets) * 100
        current_portfolio['PFE'] = {
            'shares': 80,
            'value': pfe_value,
            'weight': pfe_weight,
            'target_weight': 8.0,
            'status': '权重合适'
        }
        
        print(f"\n📈 当前持仓结构:")
        print("-" * 50)
        
        total_stock_value = 0
        for symbol, data in current_portfolio.items():
            diff = data['weight'] - data['target_weight']
            status_icon = "🔴" if diff > 2 else "🟡" if diff > 0 else "🟢"
            
            print(f"{symbol:6} | {data['weight']:5.1f}% | 目标:{data['target_weight']:4.1f}% | 差额:{diff:+5.1f}% | {status_icon} {data['status']}")
            total_stock_value += data['value']
        
        cash_value = self.estimated_total_assets - total_stock_value + self.amd_sale_proceeds
        cash_weight = (cash_value / self.estimated_total_assets) * 100
        
        print(f"CASH   | {cash_weight:5.1f}% | 目标: 7.0% | 差额:{cash_weight-7:+5.1f}% | 💰 可投资现金")
        
        return current_portfolio, cash_value
    
    def update_rebalancing_plan(self, current_portfolio, available_cash):
        """更新再平衡计划"""
        print(f"\n🎯 更新后的再平衡计划")
        print("=" * 60)
        
        print(f"💰 可用资金: ${available_cash:,.0f}")
        
        # 重新计算还需减持的数量
        print(f"\n🔴 剩余减仓需求:")
        print("-" * 30)
        
        # AMD: 从16.36%降至10%
        amd_current_value = current_portfolio['AMD']['value']
        amd_target_value = self.estimated_total_assets * 0.10
        amd_excess = amd_current_value - amd_target_value
        amd_shares_to_sell = int(amd_excess / 128.9)
        
        print(f"AMD  | 还需减持: {amd_shares_to_sell}股 (${amd_excess:,.0f})")
        
        # GOOG: 从当前权重降至12%
        goog_current_value = current_portfolio['GOOG']['value']
        goog_target_value = self.estimated_total_assets * 0.12
        goog_excess = goog_current_value - goog_target_value
        goog_shares_to_sell = int(goog_excess / 170.27)
        
        print(f"GOOG | 还需减持: {goog_shares_to_sell}股 (${goog_excess:,.0f})")
        
        # NVDA: 从当前权重降至15%
        nvda_current_value = current_portfolio['NVDA']['value']
        nvda_target_value = self.estimated_total_assets * 0.15
        nvda_excess = nvda_current_value - nvda_target_value
        nvda_shares_to_sell = max(0, int(nvda_excess / 144.13))
        
        if nvda_shares_to_sell > 0:
            print(f"NVDA | 还需减持: {nvda_shares_to_sell}股 (${nvda_excess:,.0f})")
        else:
            print(f"NVDA | 权重合适，无需减仓")
        
        # 计算总的额外资金
        total_additional_proceeds = amd_excess + goog_excess + max(0, nvda_excess)
        total_available = available_cash + total_additional_proceeds
        
        print(f"\n💵 资金规划:")
        print(f"  当前可用现金: ${available_cash:,.0f}")
        print(f"  预期减仓收益: ${total_additional_proceeds:,.0f}")
        print(f"  总可投资金额: ${total_available:,.0f}")
        
        return {
            'amd_to_sell': amd_shares_to_sell,
            'goog_to_sell': goog_shares_to_sell,
            'nvda_to_sell': nvda_shares_to_sell,
            'total_proceeds': total_additional_proceeds,
            'total_available': total_available
        }
    
    def immediate_action_plan(self, rebalancing_data):
        """制定即时行动计划"""
        print(f"\n⚡ 即时行动计划")
        print("=" * 60)
        
        print("🎯 第一优先级 (本周内执行):")
        
        # AMD继续减仓
        if rebalancing_data['amd_to_sell'] > 0:
            print(f"1. AMD继续减仓 {rebalancing_data['amd_to_sell']}股")
            print(f"   • 当前RSI: 73.8 (超买)")
            print(f"   • 建议价位: $130以上")
            print(f"   • 预期收益: ~${rebalancing_data['amd_to_sell'] * 129:,.0f}")
        
        # BRK-B立即建仓
        brk_target_value = self.estimated_total_assets * 0.08
        brk_shares = int(brk_target_value / 485)
        print(f"\n2. BRK-B立即建仓 {brk_shares}股")
        print(f"   • 当前价格: $485.43")
        print(f"   • RSI: 27.8 (超卖，绝佳买点)")
        print(f"   • 投资金额: ~${brk_shares * 485:,.0f}")
        
        # TSLA谨慎建仓
        tsla_target_value = self.estimated_total_assets * 0.05
        tsla_shares = int(tsla_target_value / 324)
        print(f"\n3. TSLA谨慎建仓 {tsla_shares}股")
        print(f"   • 当前价格: $324.27")
        print(f"   • RSI: 42.3 (健康)")
        print(f"   • 投资金额: ~${tsla_shares * 324:,.0f}")
        
        print(f"\n🎯 第二优先级 (2-4周内):")
        print("4. GOOG减仓10股 (等待反弹至$175)")
        print("5. 关注JPM回调至$270以下")
        print("6. 等待MSFT、META更大幅度回调")
        
        # 风险控制
        print(f"\n⚠️ 风险控制要点:")
        print("• AMD已减持5股，剩余减仓压力减轻")
        print("• BRK-B超卖是难得机会，优先建仓")
        print("• 保持30%现金比例应对市场变化")
        print("• 严格执行止损，单股损失不超过8%")
        
        return {
            'brk_shares': brk_shares,
            'tsla_shares': tsla_shares,
            'immediate_investment': (brk_shares * 485) + (tsla_shares * 324)
        }
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("🎯 AMD减持后投资组合更新分析")
        print("=" * 80)
        
        # 计算当前组合
        current_portfolio, available_cash = self.calculate_current_portfolio()
        
        # 更新再平衡计划
        rebalancing_data = self.update_rebalancing_plan(current_portfolio, available_cash)
        
        # 即时行动计划
        action_plan = self.immediate_action_plan(rebalancing_data)
        
        # 总结
        print(f"\n📋 执行总结")
        print("=" * 40)
        print("✅ 已完成: AMD减持5股，收益$644.5")
        print(f"🎯 下一步: BRK-B建仓{action_plan['brk_shares']}股 + TSLA建仓{action_plan['tsla_shares']}股")
        print(f"💰 所需资金: ${action_plan['immediate_investment']:,.0f}")
        print(f"💵 剩余现金: ${available_cash - action_plan['immediate_investment']:,.0f}")
        
        return {
            'current_portfolio': current_portfolio,
            'rebalancing_data': rebalancing_data,
            'action_plan': action_plan
        }

if __name__ == "__main__":
    analyzer = UpdatedPortfolioStatus()
    result = analyzer.comprehensive_analysis() 