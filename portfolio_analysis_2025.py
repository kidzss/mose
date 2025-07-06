#!/usr/bin/env python3
"""
投资组合独立分析 - 2025年1月
基于当前市场行情进行深度分析
"""

import json
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

class PortfolioAnalyzer:
    def __init__(self):
        """初始化分析器"""
        self.current_prices = {
            'PFE': 25.380,
            'NVDA': 159.340,
            'MRK': 80.93,
            'GOOG': 180.55,
            'BRK-B': 485.0,
            'AMD': 137.910
        }
        
        self.positions = {
            'PFE': {'shares': 20, 'cost': 27.095, 'weight': 1.76},
            'NVDA': {'shares': 35, 'cost': 137.942, 'weight': 19.35},
            'MRK': {'shares': 18, 'cost': 80.454, 'weight': 5.05},
            'GOOG': {'shares': 30, 'cost': 170.0, 'weight': 18.79},
            'BRK-B': {'shares': 3, 'cost': 485.083, 'weight': 5.05},
            'AMD': {'shares': 25, 'cost': 125.276, 'weight': 11.96}
        }
        
        self.total_assets = 28821.33
        
    def calculate_portfolio_metrics(self):
        """计算投资组合指标"""
        print("📊 投资组合实时分析")
        print("=" * 60)
        
        total_market_value = 0
        total_unrealized_pnl = 0
        sector_allocation = {}
        
        print("📈 持仓详情:")
        print("-" * 60)
        
        for symbol, pos in self.positions.items():
            shares = pos['shares']
            cost = pos['cost']
            current_price = self.current_prices[symbol]
            market_value = shares * current_price
            unrealized_pnl = market_value - (shares * cost)
            pnl_percentage = (unrealized_pnl / (shares * cost)) * 100
            
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl
            
            # 确定行业分类
            if symbol in ['NVDA', 'GOOG', 'AMD']:
                sector = 'Technology'
            elif symbol in ['PFE', 'MRK']:
                sector = 'Healthcare'
            elif symbol == 'BRK-B':
                sector = 'Financial'
            
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += market_value
            
            status = "🟢" if pnl_percentage > 0 else "🔴" if pnl_percentage < 0 else "🟡"
            
            print(f"{status} {symbol}:")
            print(f"   持仓: {shares}股 | 成本: ${cost:.3f} | 现价: ${current_price:.3f}")
            print(f"   市值: ${market_value:,.2f} | 盈亏: ${unrealized_pnl:,.2f} ({pnl_percentage:+.2f}%)")
            print()
        
        # 计算总体指标
        total_cost = sum(pos['shares'] * pos['cost'] for pos in self.positions.values())
        total_pnl_percentage = (total_unrealized_pnl / total_cost) * 100
        
        print("💰 投资组合总览:")
        print("-" * 60)
        print(f"股票总市值: ${total_market_value:,.2f}")
        print(f"股票总成本: ${total_cost:,.2f}")
        print(f"未实现盈亏: ${total_unrealized_pnl:,.2f} ({total_pnl_percentage:+.2f}%)")
        print(f"现金+基金: ${28821.33 - total_market_value:,.2f}")
        print()
        
        # 行业配置分析
        print("🏭 行业配置:")
        print("-" * 60)
        for sector, value in sector_allocation.items():
            percentage = (value / total_market_value) * 100
            print(f"{sector}: ${value:,.2f} ({percentage:.1f}%)")
        print()
        
        return {
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'sector_allocation': sector_allocation
        }
    
    def analyze_individual_positions(self):
        """分析个别持仓"""
        print("🎯 个别持仓分析:")
        print("-" * 60)
        
        # PFE分析
        pfe_analysis = """
🔍 PFE (辉瑞) 分析:
   - 当前价格: $25.38 (成本: $27.095)
   - 亏损: -6.33%
   - 做T策略: 明智选择
   - 理由: 医药股近期承压，PFE基本面稳健但短期技术面偏弱
   - 建议: 在$24-26区间做T，目标盈利3-5%
"""
        print(pfe_analysis)
        
        # NVDA分析
        nvda_analysis = """
🔍 NVDA (英伟达) 分析:
   - 当前价格: $159.34 (成本: $137.942)
   - 盈利: +15.52%
   - 地位: 投资组合最大持仓，AI龙头
   - 优势: AI芯片需求强劲，技术领先
   - 风险: 估值较高，需关注回调风险
   - 建议: 继续持有，可考虑在$170+分批减仓
"""
        print(nvda_analysis)
        
        # GOOG分析
        goog_analysis = """
🔍 GOOG (谷歌) 分析:
   - 当前价格: $180.55 (成本: $170.00)
   - 盈利: +6.21%
   - 地位: 第二大持仓，科技巨头
   - 优势: AI布局全面，搜索广告业务稳定
   - 风险: 监管压力，竞争加剧
   - 建议: 稳健持有，目标价$200+
"""
        print(goog_analysis)
        
        # AMD分析
        amd_analysis = """
🔍 AMD (超微) 分析:
   - 当前价格: $137.91 (成本: $125.276)
   - 盈利: +10.08%
   - 地位: 第三大持仓，芯片制造商
   - 优势: 技术追赶Intel，AI芯片布局
   - 风险: 竞争激烈，周期性明显
   - 建议: 继续持有，技术面强势
"""
        print(amd_analysis)
        
        # MRK分析
        mrk_analysis = """
🔍 MRK (默克) 分析:
   - 当前价格: $80.93 (成本: $80.454)
   - 盈利: +0.59%
   - 地位: 防御性配置，医药龙头
   - 优势: 研发管线丰富，现金流稳定
   - 风险: 专利到期，监管风险
   - 建议: 长期持有，分红稳定
"""
        print(mrk_analysis)
        
        # BRK-B分析
        brkb_analysis = """
🔍 BRK-B (伯克希尔) 分析:
   - 当前价格: $485.00 (成本: $485.083)
   - 盈亏: -0.02%
   - 地位: 价值投资标杆，防御性配置
   - 优势: 多元化业务，现金充裕
   - 风险: 巴菲特年龄，接班问题
   - 建议: 长期持有，价值投资典范
"""
        print(brkb_analysis)
    
    def market_outlook_analysis(self):
        """市场展望分析"""
        print("🌍 市场展望分析:")
        print("-" * 60)
        
        outlook = """
📈 当前市场环境:
   - 美联储政策: 利率维持高位，但降息预期升温
   - 经济数据: 通胀放缓，就业市场稳健
   - 地缘政治: 中东局势紧张，但影响有限
   - 科技股: AI主题持续火热，但估值偏高

🎯 投资组合优势:
   ✅ 科技股占比合理(约50%)，把握AI趋势
   ✅ 防御性配置(医药+金融)提供稳定性
   ✅ 现金+基金占比38%，流动性充足
   ✅ 个股选择优质，基本面良好

⚠️ 需要注意的风险:
   - 科技股估值偏高，需关注回调风险
   - 集中度较高，NVDA+GOOG占比38%
   - 医药股短期承压，PFE做T策略合理

💡 投资建议:
   1. 继续持有核心科技股(NVDA, GOOG, AMD)
   2. PFE做T策略执行得当，可继续操作
   3. 考虑增加防御性股票，降低集中度
   4. 保持现金储备，等待更好买入机会
"""
        print(outlook)
    
    def risk_assessment(self):
        """风险评估"""
        print("⚠️ 风险评估:")
        print("-" * 60)
        
        # 计算集中度风险
        total_stock_value = sum(pos['shares'] * self.current_prices[symbol] 
                               for symbol, pos in self.positions.items())
        
        concentration_risk = {}
        for symbol, pos in self.positions.items():
            market_value = pos['shares'] * self.current_prices[symbol]
            concentration = (market_value / total_stock_value) * 100
            concentration_risk[symbol] = concentration
        
        print("📊 持仓集中度:")
        for symbol, concentration in sorted(concentration_risk.items(), 
                                         key=lambda x: x[1], reverse=True):
            risk_level = "🔴" if concentration > 20 else "🟡" if concentration > 15 else "🟢"
            print(f"{risk_level} {symbol}: {concentration:.1f}%")
        
        print(f"\n⚠️ 风险提示:")
        print(f"  - 最大持仓NVDA占比{concentration_risk['NVDA']:.1f}%，需关注回调风险")
        print(f"  - 科技股集中度较高，建议适当分散")
        print(f"  - 现金+基金占比38%，流动性充足，风险可控")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 开始投资组合分析...")
        print("=" * 60)
        
        # 1. 计算基础指标
        metrics = self.calculate_portfolio_metrics()
        
        # 2. 个别持仓分析
        self.analyze_individual_positions()
        
        # 3. 市场展望
        self.market_outlook_analysis()
        
        # 4. 风险评估
        self.risk_assessment()
        
        # 5. 总结
        print("\n🎯 投资组合总体评价:")
        print("-" * 60)
        
        if metrics['total_pnl_percentage'] > 5:
            print("✅ 投资组合表现优秀，整体盈利良好")
        elif metrics['total_pnl_percentage'] > 0:
            print("✅ 投资组合表现稳定，略有盈利")
        else:
            print("⚠️ 投资组合暂时亏损，但配置合理")
        
        print(f"📈 主要优势: 科技股布局合理，防御性配置到位")
        print(f"💡 改进建议: 可考虑增加更多防御性股票，降低集中度")
        print(f"🎯 总体评级: 良好 (B+)")

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.run_complete_analysis() 