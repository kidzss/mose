import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioAdjustmentAnalysis:
    """投资组合调整分析"""
    
    def __init__(self):
        # 当前持仓数据
        self.current_portfolio = {
            'total_assets': 27673.00,
            'stock_allocation': 17347.00,  # 62.68%
            'cash_allocation': 4342.00,    # 15.69%
            'money_fund': 5987.00,         # 21.64%
            'hk_stock': 97.00              # 小米港股
        }
        
        self.positions = {
            'AMD': {
                'shares': 40,
                'weight': 18.72,
                'current_value': 5174.40,
                'cost_basis': 126.156,
                'unrealized_pnl': 128.16
            },
            'GOOG': {
                'shares': 30,
                'weight': 18.47,
                'current_value': 5095.20,
                'cost_basis': 170.00,
                'unrealized_pnl': -4.80
            },
            'NVDA': {
                'shares': 35,
                'weight': 18.23,
                'current_value': 5043.15,
                'cost_basis': 138.843,
                'unrealized_pnl': 183.64
            },
            'PFE': {
                'shares': 80,
                'weight': 7.76,
                'current_value': 1907.20,
                'cost_basis': 25.899,
                'unrealized_pnl': -164.72
            }
        }
        
        # 交易记录
        self.transactions = {
            'TSLA_exit': {
                'shares': 4,
                'exit_price': 324.00,
                'cost_basis': 179.841,
                'profit': 576.64,
                'return_pct': 79.6
            },
            'NVDA_reduction': {
                'shares_sold': 5,
                'estimated_price': 145.00,  # 估算减持价格
                'profit_estimate': 30.78    # 估算利润 (145-138.843)*5
            }
        }
    
    def analyze_portfolio_structure(self):
        """分析投资组合结构"""
        print("📊 投资组合结构分析")
        print("=" * 60)
        
        print(f"💰 总资产: ${self.current_portfolio['total_assets']:,.2f}")
        print()
        
        # 资产配置分析
        stock_pct = (self.current_portfolio['stock_allocation'] / self.current_portfolio['total_assets']) * 100
        cash_pct = (self.current_portfolio['cash_allocation'] / self.current_portfolio['total_assets']) * 100
        fund_pct = (self.current_portfolio['money_fund'] / self.current_portfolio['total_assets']) * 100
        
        print("📈 资产配置:")
        print(f"  股票投资: ${self.current_portfolio['stock_allocation']:,.2f} ({stock_pct:.1f}%)")
        print(f"  现金储备: ${self.current_portfolio['cash_allocation']:,.2f} ({cash_pct:.1f}%)")
        print(f"  货币基金: ${self.current_portfolio['money_fund']:,.2f} ({fund_pct:.1f}%)")
        print(f"  港股小米: ${self.current_portfolio['hk_stock']:,.2f} (0.4%)")
        
        # 股票持仓分析
        print(f"\n🏆 股票持仓排名:")
        sorted_positions = sorted(self.positions.items(), key=lambda x: x[1]['weight'], reverse=True)
        
        for i, (symbol, data) in enumerate(sorted_positions, 1):
            pnl_status = "📈" if data['unrealized_pnl'] > 0 else "📉"
            print(f"  {i}. {symbol}: {data['weight']:.1f}% | ${data['current_value']:,.0f} | {pnl_status} {data['unrealized_pnl']:+.0f}")
        
        return sorted_positions
    
    def analyze_recent_transactions(self):
        """分析近期交易"""
        print(f"\n💼 近期交易分析")
        print("=" * 60)
        
        total_realized_profit = 0
        
        print("🔄 TSLA清仓:")
        tsla = self.transactions['TSLA_exit']
        print(f"  卖出: {tsla['shares']}股 @ ${tsla['exit_price']:.2f}")
        print(f"  成本: ${tsla['cost_basis']:.2f}")
        print(f"  利润: ${tsla['profit']:.2f} ({tsla['return_pct']:.1f}%)")
        print(f"  策略: 右侧交易获利了结")
        total_realized_profit += tsla['profit']
        
        print(f"\n📉 NVDA减持:")
        nvda = self.transactions['NVDA_reduction']
        print(f"  减持: {nvda['shares_sold']}股 (40→35股)")
        print(f"  估算价格: ~${nvda['estimated_price']:.2f}")
        print(f"  估算利润: ~${nvda['profit_estimate']:.2f}")
        print(f"  策略: 仓位管理，降低集中度风险")
        total_realized_profit += nvda['profit_estimate']
        
        print(f"\n💰 总实现利润: ${total_realized_profit:.2f}")
        
        return total_realized_profit
    
    def analyze_risk_management(self):
        """分析风险管理效果"""
        print(f"\n🛡️ 风险管理分析")
        print("=" * 60)
        
        # 集中度风险分析
        print("📊 持仓集中度:")
        top3_weight = sum([data['weight'] for _, data in list(self.positions.items())[:3]])
        print(f"  前三大持仓占比: {top3_weight:.1f}%")
        
        if top3_weight > 60:
            print("  ⚠️ 集中度偏高，建议进一步分散")
        elif top3_weight > 50:
            print("  🟡 集中度适中，可适当分散")
        else:
            print("  ✅ 集中度合理")
        
        # 行业分散度
        tech_weight = sum([data['weight'] for symbol, data in self.positions.items() 
                          if symbol in ['AMD', 'GOOG', 'NVDA']])
        print(f"  科技股占比: {tech_weight:.1f}%")
        
        if tech_weight > 60:
            print("  ⚠️ 科技股过度集中，建议增加其他行业")
        else:
            print("  ✅ 行业配置需要优化，但风险可控")
        
        # 现金比例分析
        cash_ratio = (self.current_portfolio['cash_allocation'] + self.current_portfolio['money_fund']) / self.current_portfolio['total_assets']
        print(f"\n💵 流动性分析:")
        print(f"  现金+基金占比: {cash_ratio:.1%}")
        
        if cash_ratio > 0.4:
            print("  📈 流动性充足，有较强的投资灵活性")
        elif cash_ratio > 0.3:
            print("  ✅ 流动性良好，平衡了风险和收益")
        elif cash_ratio > 0.2:
            print("  🟡 流动性适中，关注市场机会")
        else:
            print("  ⚠️ 流动性偏低，建议保留更多现金")
    
    def generate_optimization_suggestions(self):
        """生成优化建议"""
        print(f"\n💡 投资组合优化建议")
        print("=" * 60)
        
        available_cash = self.current_portfolio['cash_allocation']
        
        print(f"💰 可用资金: ${available_cash:,.2f}")
        print()
        
        print("🎯 配置建议 (按优先级排序):")
        
        print("\n1️⃣ 防御性配置 (高优先级)")
        print(f"   目标: 增加防御性资产，降低波动性")
        print(f"   建议:")
        print(f"   • JPM: $1,100 (4股) - 金融龙头，估值合理")
        print(f"   • MRK: $400 (5股) - 医疗股，4.1%股息")
        print(f"   • BRK-B: $485 (1股) - 巴菲特旗舰，防御性强")
        print(f"   小计: $1,985")
        
        print("\n2️⃣ 价值成长配置 (中优先级)")
        print(f"   目标: 平衡成长和价值")
        print(f"   建议:")
        print(f"   • ORCL: $400 (2股) - 云计算转型成功")
        print(f"   • IBM: $540 (2股) - AI转型，分红稳定")
        print(f"   小计: $940")
        
        print("\n3️⃣ 消费防御配置 (中优先级)")
        print(f"   目标: 增加消费必需品")
        print(f"   建议:")
        print(f"   • COST: $955 (1股) - 会员制零售龙头")
        print(f"   • PG: $450 (3股) - 消费必需品，稳定分红")
        print(f"   小计: $1,405")
        
        print(f"\n💼 分批投资策略:")
        print(f"   • 第一批: $2,000 (防御性为主)")
        print(f"   • 第二批: $1,500 (等待更好时机)")
        print(f"   • 保留现金: $842 (机动资金)")
        
        print(f"\n⏰ 时机选择:")
        print(f"   • 不急于一次性投入")
        print(f"   • 关注技术指标和基本面")
        print(f"   • 财报季后可能有更好机会")
        print(f"   • 保持15%现金比例的灵活性")
    
    def analyze_performance_attribution(self):
        """分析业绩归因"""
        print(f"\n📈 业绩归因分析")
        print("=" * 60)
        
        total_unrealized = sum([data['unrealized_pnl'] for data in self.positions.values()])
        total_realized = self.analyze_recent_transactions()
        
        print(f"💰 收益分解:")
        print(f"  已实现收益: ${total_realized:.2f}")
        print(f"  未实现收益: ${total_unrealized:.2f}")
        print(f"  总收益: ${total_realized + total_unrealized:.2f}")
        
        print(f"\n🏆 最佳表现:")
        best_performer = max(self.positions.items(), key=lambda x: x[1]['unrealized_pnl'])
        print(f"  {best_performer[0]}: ${best_performer[1]['unrealized_pnl']:+.2f}")
        
        print(f"\n📉 待改善:")
        worst_performer = min(self.positions.items(), key=lambda x: x[1]['unrealized_pnl'])
        print(f"  {worst_performer[0]}: ${worst_performer[1]['unrealized_pnl']:+.2f}")
        
        # 策略执行评估
        print(f"\n🎯 策略执行评估:")
        print(f"  ✅ TSLA: 成功实现79.6%收益")
        print(f"  ✅ NVDA: 适度减持，降低集中度")
        print(f"  ✅ 现金比例: 提升至37.3%，增强灵活性")
        print(f"  ⚠️ 科技股占比: 仍较高(55.4%)，需要分散")
    
    def comprehensive_analysis(self):
        """综合分析报告"""
        print("🎯 投资组合调整综合分析")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 执行各项分析
        self.analyze_portfolio_structure()
        realized_profit = self.analyze_recent_transactions()
        self.analyze_risk_management()
        self.analyze_performance_attribution()
        self.generate_optimization_suggestions()
        
        # 总结评价
        print(f"\n📋 调整效果评估")
        print("=" * 60)
        
        score = 0
        
        # 风险控制 (30%)
        cash_ratio = (self.current_portfolio['cash_allocation'] + self.current_portfolio['money_fund']) / self.current_portfolio['total_assets']
        if cash_ratio > 0.35:
            risk_score = 9
        elif cash_ratio > 0.25:
            risk_score = 7
        else:
            risk_score = 5
        score += risk_score * 0.3
        
        # 收益实现 (40%)
        if realized_profit > 500:
            profit_score = 10
        elif realized_profit > 300:
            profit_score = 8
        else:
            profit_score = 6
        score += profit_score * 0.4
        
        # 配置优化 (30%)
        config_score = 7  # 基于分散度改善
        score += config_score * 0.3
        
        print(f"💯 调整综合评分: {score:.1f}/10")
        
        if score >= 8.5:
            grade = "A+ 优秀"
        elif score >= 7.5:
            grade = "A 良好"
        elif score >= 6.5:
            grade = "B+ 中上"
        else:
            grade = "B 中等"
        
        print(f"🏆 调整等级: {grade}")
        
        print(f"\n✅ 调整成果:")
        print("  • 成功获利了结，实现超额收益")
        print("  • 降低了持仓集中度风险")
        print("  • 大幅提升了资金灵活性")
        print("  • 为后续投资创造了条件")
        
        print(f"\n🎯 下一步重点:")
        print("  • 适度增加防御性资产")
        print("  • 分散科技股集中风险")
        print("  • 保持合理现金比例")
        print("  • 等待更好的投资时机")

if __name__ == "__main__":
    analyzer = PortfolioAdjustmentAnalysis()
    analyzer.comprehensive_analysis() 