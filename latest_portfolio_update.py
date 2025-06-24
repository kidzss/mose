import yfinance as yf
import pandas as pd
from datetime import datetime
import json

class LatestPortfolioUpdate:
    """最新投资组合持仓更新"""
    
    def __init__(self):
        # 最新持仓信息（截至2025年6月20日）
        self.updated_positions = {
            'AMD': {
                'shares': 35,  # 从40股减持5股
                'sale_price': 128.9,  # 减持价格
                'sale_proceeds': 5 * 128.9,  # $644.5
                'status': '已减持5股'
            },
            'GOOG': {
                'shares': 30,  # 保持不变
                'status': '持有中'
            },
            'NVDA': {
                'shares': 35,  # 保持不变  
                'status': '持有中'
            },
            'PFE': {
                'shares': 80,  # 保持不变
                'status': '持有中'
            },
            'BRK-B': {
                'shares': 2,  # 新买入
                'purchase_price': 485.36,
                'purchase_amount': 2 * 485.36,  # $970.72
                'status': '新建仓'
            }
        }
        
        # 预估总资产
        self.estimated_total_assets = 27578  # 基于AMD权重16.36%反推
        
        # 现金变化
        self.cash_changes = {
            'amd_sale_proceeds': 644.5,
            'brk_purchase_cost': 970.72,
            'net_cash_change': 644.5 - 970.72  # -$326.22
        }
    
    def get_current_prices(self):
        """获取当前股价"""
        symbols = ['AMD', 'GOOG', 'NVDA', 'PFE', 'BRK-B']
        current_prices = {}
        
        print("📡 获取最新股价...")
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period="1d")
                if len(data) > 0:
                    current_prices[symbol] = data['Close'].iloc[-1]
                else:
                    # 使用已知价格作为备用
                    fallback_prices = {
                        'AMD': 129.34, 'GOOG': 170.27, 'NVDA': 144.13, 
                        'PFE': 23.84, 'BRK-B': 485.36
                    }
                    current_prices[symbol] = fallback_prices.get(symbol, 0)
            except:
                # 使用已知价格作为备用
                fallback_prices = {
                    'AMD': 129.34, 'GOOG': 170.27, 'NVDA': 144.13, 
                    'PFE': 23.84, 'BRK-B': 485.36
                }
                current_prices[symbol] = fallback_prices.get(symbol, 0)
        
        return current_prices
    
    def calculate_updated_portfolio(self):
        """计算更新后的投资组合"""
        print("📊 最新投资组合持仓更新")
        print("=" * 60)
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        current_prices = self.get_current_prices()
        
        print(f"\n✅ 最近操作记录:")
        print(f"  AMD减持: 5股 @ $128.9，收益 $644.5")
        print(f"  BRK-B建仓: 2股 @ $485.36，投入 $970.72")
        print(f"  净现金变化: -$326.22")
        
        # 计算当前持仓价值
        portfolio_summary = {}
        total_stock_value = 0
        
        print(f"\n📈 当前持仓明细:")
        print("-" * 50)
        print(f"{'股票':<8} {'股数':<6} {'当前价':<8} {'市值':<10} {'权重':<8} {'状态':<12}")
        print("-" * 50)
        
        for symbol, position in self.updated_positions.items():
            shares = position['shares']
            current_price = current_prices.get(symbol, 0)
            market_value = shares * current_price
            weight = (market_value / self.estimated_total_assets) * 100
            status = position['status']
            
            portfolio_summary[symbol] = {
                'shares': shares,
                'current_price': current_price,
                'market_value': market_value,
                'weight': weight,
                'status': status
            }
            
            total_stock_value += market_value
            
            print(f"{symbol:<8} {shares:<6} ${current_price:<7.2f} ${market_value:<9.0f} {weight:<7.1f}% {status:<12}")
        
        # 计算现金余额
        original_cash = self.estimated_total_assets - (total_stock_value - portfolio_summary['BRK-B']['market_value'])
        updated_cash = original_cash + self.cash_changes['net_cash_change']
        cash_weight = (updated_cash / self.estimated_total_assets) * 100
        
        print("-" * 50)
        print(f"{'现金':<8} {'--':<6} {'--':<8} ${updated_cash:<9.0f} {cash_weight:<7.1f}% {'可投资':<12}")
        print("-" * 50)
        print(f"{'总计':<8} {'--':<6} {'--':<8} ${self.estimated_total_assets:<9.0f} {'100.0%':<7} {'--':<12}")
        
        return portfolio_summary, updated_cash
    
    def analyze_portfolio_changes(self, portfolio_summary):
        """分析投资组合变化"""
        print(f"\n📊 投资组合变化分析")
        print("-" * 40)
        
        # 权重变化分析
        print(f"权重变化:")
        print(f"  AMD: 18.7% → {portfolio_summary['AMD']['weight']:.1f}% (减持后)")
        print(f"  GOOG: {portfolio_summary['GOOG']['weight']:.1f}% (保持)")
        print(f"  NVDA: {portfolio_summary['NVDA']['weight']:.1f}% (保持)")
        print(f"  PFE: {portfolio_summary['PFE']['weight']:.1f}% (保持)")
        print(f"  BRK-B: 0% → {portfolio_summary['BRK-B']['weight']:.1f}% (新建仓)")
        
        # 行业配置分析
        tech_weight = (portfolio_summary['AMD']['weight'] + 
                      portfolio_summary['GOOG']['weight'] + 
                      portfolio_summary['NVDA']['weight'])
        
        print(f"\n行业配置:")
        print(f"  科技股权重: {tech_weight:.1f}% (AMD+GOOG+NVDA)")
        print(f"  医药股权重: {portfolio_summary['PFE']['weight']:.1f}% (PFE)")
        print(f"  价值股权重: {portfolio_summary['BRK-B']['weight']:.1f}% (BRK-B)")
        
        # 风险分散度
        max_weight = max([pos['weight'] for pos in portfolio_summary.values()])
        print(f"\n风险控制:")
        print(f"  最大单股权重: {max_weight:.1f}%")
        print(f"  集中度风险: {'偏高' if max_weight > 20 else '适中' if max_weight > 15 else '较低'}")
    
    def next_steps_recommendation(self, portfolio_summary, cash_balance):
        """下一步操作建议"""
        print(f"\n🎯 下一步操作建议")
        print("-" * 40)
        
        print(f"💰 可用现金: ${cash_balance:,.0f} ({(cash_balance/self.estimated_total_assets)*100:.1f}%)")
        
        # 计算科技股权重
        tech_weight = (portfolio_summary['AMD']['weight'] + 
                      portfolio_summary['GOOG']['weight'] + 
                      portfolio_summary['NVDA']['weight'])
        
        print(f"\n🔴 继续减仓建议:")
        if portfolio_summary['AMD']['weight'] > 12:
            amd_excess = portfolio_summary['AMD']['weight'] - 10
            print(f"  AMD: 还需减持约{amd_excess/portfolio_summary['AMD']['weight']*35:.0f}股 (目标10%)")
        
        if portfolio_summary['GOOG']['weight'] > 14:
            goog_excess = portfolio_summary['GOOG']['weight'] - 12
            print(f"  GOOG: 还需减持约{goog_excess/portfolio_summary['GOOG']['weight']*30:.0f}股 (目标12%)")
        
        print(f"\n🟢 建仓机会:")
        print(f"  TSLA: 等待$320以下建仓4股")
        print(f"  BRK-B: 如跌至$437可加仓1-2股")
        print(f"  MSFT: 等待$450以下建仓")
        print(f"  META: 等待$650以下建仓")
        
        print(f"\n⚠️ 风险提醒:")
        print(f"  • 科技股仍占{tech_weight:.1f}%，注意板块风险")
        print(f"  • BRK-B仓位较小，可考虑逐步加仓")
        print(f"  • 保持现金比例应对市场变化")
    
    def save_portfolio_config(self, portfolio_summary, cash_balance):
        """保存投资组合配置"""
        config = {
            'update_time': datetime.now().isoformat(),
            'total_assets': self.estimated_total_assets,
            'cash_balance': cash_balance,
            'positions': {},
            'recent_transactions': [
                {
                    'date': '2025-06-20',
                    'action': 'SELL',
                    'symbol': 'AMD',
                    'shares': 5,
                    'price': 128.9,
                    'amount': 644.5
                },
                {
                    'date': '2025-06-20',
                    'action': 'BUY',
                    'symbol': 'BRK-B',
                    'shares': 2,
                    'price': 485.36,
                    'amount': 970.72
                }
            ]
        }
        
        for symbol, data in portfolio_summary.items():
            config['positions'][symbol] = {
                'shares': data['shares'],
                'current_price': data['current_price'],
                'market_value': data['market_value'],
                'weight': data['weight']
            }
        
        # 保存到配置文件
        with open('config/portfolio_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 投资组合配置已更新至 config/portfolio_config.json")
    
    def comprehensive_update(self):
        """综合更新报告"""
        print("🎯 投资组合持仓综合更新")
        print("=" * 80)
        
        # 计算更新后组合
        portfolio_summary, cash_balance = self.calculate_updated_portfolio()
        
        # 分析变化
        self.analyze_portfolio_changes(portfolio_summary)
        
        # 下一步建议
        self.next_steps_recommendation(portfolio_summary, cash_balance)
        
        # 保存配置
        self.save_portfolio_config(portfolio_summary, cash_balance)
        
        return {
            'portfolio_summary': portfolio_summary,
            'cash_balance': cash_balance,
            'total_assets': self.estimated_total_assets
        }

if __name__ == "__main__":
    updater = LatestPortfolioUpdate()
    result = updater.comprehensive_update() 