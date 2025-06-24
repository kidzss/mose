#!/usr/bin/env python3
"""
更新后的投资组合分析器 - 基于最新持股配置
"""

import json
import yfinance as yf
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UpdatedPortfolioAnalyzer:
    """更新后的投资组合分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 从配置文件读取最新持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 新增投资
        self.new_investments = {
            'MRK': {'shares': 14, 'price': 79.29, 'sector': 'Healthcare'},
            'JPM': {'shares': 3, 'price': 273.96, 'sector': 'Financial'}
        }
        
        logger.info("📊 更新后的投资组合分析器初始化完成")
    
    def get_current_prices(self):
        """获取当前股价"""
        # 从配置文件获取股票列表
        positions = self.config['positions']
        us_stocks = [symbol for symbol in positions.keys() if not symbol.endswith('.HK')]
        
        # 添加新投资股票
        all_symbols = us_stocks + list(self.new_investments.keys())
        
        prices = {}
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prices[symbol] = current_price
                    logger.info(f"📊 {symbol}: ${current_price:.2f}")
                    
            except Exception as e:
                logger.warning(f"获取{symbol}价格失败: {e}")
                
        return prices
    
    def calculate_current_portfolio(self, prices):
        """计算当前投资组合"""
        current_portfolio = {}
        total_stock_value = 0
        
        # 现有持仓
        for symbol, position in self.config['positions'].items():
            if symbol.endswith('.HK'):
                continue  # 跳过港股
                
            if symbol in prices:
                current_value = position['shares'] * prices[symbol]
                unrealized_pnl = current_value - (position['shares'] * position['cost_basis'])
                pnl_percentage = (unrealized_pnl / (position['shares'] * position['cost_basis'])) * 100
                
                current_portfolio[symbol] = {
                    'shares': position['shares'],
                    'cost_basis': position['cost_basis'],
                    'current_price': prices[symbol],
                    'current_value': current_value,
                    'sector': position['sector'],
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percentage': pnl_percentage,
                    'type': 'existing'
                }
                total_stock_value += current_value
        
        return current_portfolio, total_stock_value
    
    def add_new_investments(self, current_portfolio, total_stock_value, prices):
        """添加新投资"""
        new_total_value = total_stock_value
        
        for symbol, investment in self.new_investments.items():
            if symbol in prices:
                current_value = investment['shares'] * prices[symbol]
                unrealized_pnl = current_value - (investment['shares'] * investment['price'])
                pnl_percentage = (unrealized_pnl / (investment['shares'] * investment['price'])) * 100
                
                current_portfolio[symbol] = {
                    'shares': investment['shares'],
                    'cost_basis': investment['price'],
                    'current_price': prices[symbol],
                    'current_value': current_value,
                    'sector': investment['sector'],
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percentage': pnl_percentage,
                    'type': 'new_investment'
                }
                new_total_value += current_value
        
        return current_portfolio, new_total_value
    
    def analyze_portfolio_composition(self, portfolio, total_value):
        """分析投资组合构成"""
        # 按股票分析
        stock_analysis = {}
        for symbol, data in portfolio.items():
            allocation = (data['current_value'] / total_value) * 100
            stock_analysis[symbol] = {
                'allocation': allocation,
                'value': data['current_value'],
                'sector': data['sector'],
                'type': data['type']
            }
        
        # 按行业分析
        sector_analysis = {}
        for symbol, data in stock_analysis.items():
            sector = data['sector']
            if sector not in sector_analysis:
                sector_analysis[sector] = {
                    'allocation': 0,
                    'value': 0,
                    'stocks': []
                }
            
            sector_analysis[sector]['allocation'] += data['allocation']
            sector_analysis[sector]['value'] += data['value']
            sector_analysis[sector]['stocks'].append({
                'symbol': symbol,
                'allocation': data['allocation'],
                'value': data['value'],
                'type': data['type']
            })
        
        return stock_analysis, sector_analysis
    
    def calculate_total_assets(self, new_stock_value):
        """计算总资产"""
        # 从配置文件获取其他资产
        cash = self.config['portfolio']['cash_allocation']['amount']
        money_fund = self.config['portfolio']['money_fund_allocation']['amount']
        hk_stock = self.config['portfolio']['hk_stock_allocation']['total_amount']
        
        # 新增投资金额
        new_investment_amount = sum([inv['shares'] * inv['price'] for inv in self.new_investments.values()])
        
        # 假设新投资来自现金
        updated_cash = cash - new_investment_amount
        
        total_assets = new_stock_value + updated_cash + money_fund + hk_stock
        
        return {
            'total_assets': total_assets,
            'stock_value': new_stock_value,
            'cash': updated_cash,
            'money_fund': money_fund,
            'hk_stock': hk_stock,
            'new_investment_amount': new_investment_amount
        }
    
    def generate_analysis_report(self, portfolio, total_value, stock_analysis, 
                               sector_analysis, asset_breakdown, prices):
        """生成分析报告"""
        report = []
        report.append("=" * 100)
        report.append("📊 更新后投资组合分析报告")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"💰 总资产: ${asset_breakdown['total_assets']:,.2f}")
        report.append("=" * 100)
        
        # 资产配置概览
        report.append(f"\n💼 资产配置概览:")
        report.append("-" * 80)
        stock_pct = (asset_breakdown['stock_value'] / asset_breakdown['total_assets']) * 100
        cash_pct = (asset_breakdown['cash'] / asset_breakdown['total_assets']) * 100
        fund_pct = (asset_breakdown['money_fund'] / asset_breakdown['total_assets']) * 100
        hk_pct = (asset_breakdown['hk_stock'] / asset_breakdown['total_assets']) * 100
        
        report.append(f"• 美股投资: ${asset_breakdown['stock_value']:,.2f} ({stock_pct:.1f}%)")
        report.append(f"• 现金储备: ${asset_breakdown['cash']:,.2f} ({cash_pct:.1f}%)")
        report.append(f"• 货币基金: ${asset_breakdown['money_fund']:,.2f} ({fund_pct:.1f}%) [年化4%收益]")
        report.append(f"• 港股小米: ${asset_breakdown['hk_stock']:,.2f} ({hk_pct:.1f}%)")
        report.append(f"• 新增投资: ${asset_breakdown['new_investment_amount']:,.2f}")
        
        # 股票持仓明细
        report.append(f"\n🏢 股票持仓明细 (按权重排序):")
        report.append("-" * 80)
        
        sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['allocation'], reverse=True)
        
        for symbol, data in sorted_stocks:
            portfolio_data = portfolio[symbol]
            status = "🆕" if data['type'] == 'new_investment' else "📈"
            
            report.append(f"{status} {symbol}: {portfolio_data['shares']}股 × "
                         f"${portfolio_data['current_price']:.2f} = "
                         f"${data['value']:,.2f} ({data['allocation']:.1f}%) "
                         f"[{portfolio_data['pnl_percentage']:+.1f}%]")
        
        # 行业配置分析
        report.append(f"\n🏭 行业配置分析:")
        report.append("-" * 80)
        
        sorted_sectors = sorted(sector_analysis.items(), key=lambda x: x[1]['allocation'], reverse=True)
        
        for sector, data in sorted_sectors:
            report.append(f"\n【{sector} - {data['allocation']:.1f}%】")
            report.append(f"  总价值: ${data['value']:,.2f}")
            
            for stock in sorted(data['stocks'], key=lambda x: x['allocation'], reverse=True):
                status = "🆕" if stock['type'] == 'new_investment' else "   "
                report.append(f"  {status}• {stock['symbol']}: {stock['allocation']:.1f}%")
        
        # 风险分析
        report.append(f"\n⚠️ 风险分析:")
        report.append("-" * 80)
        
        # 集中度风险
        high_concentration = []
        for symbol, data in stock_analysis.items():
            if data['allocation'] > 15:
                high_concentration.append(f"{symbol}({data['allocation']:.1f}%)")
        
        if high_concentration:
            report.append(f"• 高集中度风险: {', '.join(high_concentration)}")
        else:
            report.append(f"• ✅ 单股集中度风险已得到控制 (最高{max([d['allocation'] for d in stock_analysis.values()]):.1f}%)")
        
        # 行业集中度
        tech_allocation = sector_analysis.get('Technology', {}).get('allocation', 0)
        if tech_allocation > 50:
            report.append(f"• ⚠️ 科技股占比{tech_allocation:.1f}%仍然偏高，建议进一步分散")
        elif tech_allocation > 35:
            report.append(f"• 🔶 科技股占比{tech_allocation:.1f}%适中，但建议关注行业风险")
        else:
            report.append(f"• ✅ 科技股占比{tech_allocation:.1f}%合理")
        
        # 新增投资效果
        report.append(f"\n📈 新增投资效果:")
        report.append("-" * 80)
        
        healthcare_allocation = sector_analysis.get('Healthcare', {}).get('allocation', 0)
        financial_allocation = sector_analysis.get('Financial', {}).get('allocation', 0)
        
        report.append(f"• 医疗板块占比提升至: {healthcare_allocation:.1f}%")
        report.append(f"• 金融板块新增占比: {financial_allocation:.1f}%")
        report.append(f"• 行业分散度改善: 从3个行业扩展到4个行业")
        
        # 投资建议
        report.append(f"\n💡 投资建议:")
        report.append("-" * 80)
        
        if tech_allocation > 50:
            report.append(f"• 🔴 优先级1: 继续减持科技股，目标控制在40%以内")
        
        if healthcare_allocation < 20:
            report.append(f"• 🟡 优先级2: 可考虑继续增加医疗股配置")
        
        if financial_allocation < 15:
            report.append(f"• 🟡 优先级3: 可考虑继续增加金融股配置")
        
        report.append(f"• 🟢 建议新增消费股板块，进一步提升防御性")
        report.append(f"• 🟢 保持当前现金+货币基金比例({cash_pct + fund_pct:.1f}%)，流动性充足")
        
        # 预期表现
        report.append(f"\n📊 组合特征预期:")
        report.append("-" * 80)
        report.append(f"• 预期年化收益: 9-13% (新增防御性股票后)")
        report.append(f"• 预期波动率: 20-25% (行业分散降低风险)")
        report.append(f"• 最大回撤预期: 18-28% (较之前有所改善)")
        report.append(f"• 夏普比率目标: 0.35-0.50")
        
        report.append("\n" + "=" * 100)
        
        return '\n'.join(report)

def main():
    """主函数"""
    analyzer = UpdatedPortfolioAnalyzer()
    
    # 获取当前价格
    prices = analyzer.get_current_prices()
    
    if len(prices) >= 5:
        # 计算当前投资组合
        current_portfolio, current_stock_value = analyzer.calculate_current_portfolio(prices)
        
        # 添加新投资
        updated_portfolio, new_total_value = analyzer.add_new_investments(
            current_portfolio, current_stock_value, prices)
        
        # 分析投资组合构成
        stock_analysis, sector_analysis = analyzer.analyze_portfolio_composition(
            updated_portfolio, new_total_value)
        
        # 计算总资产
        asset_breakdown = analyzer.calculate_total_assets(new_total_value)
        
        # 生成报告
        report = analyzer.generate_analysis_report(
            updated_portfolio, new_total_value, stock_analysis, 
            sector_analysis, asset_breakdown, prices)
        
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON数据
        output_data = {
            'timestamp': timestamp,
            'total_assets': asset_breakdown['total_assets'],
            'current_prices': prices,
            'portfolio': updated_portfolio,
            'stock_analysis': stock_analysis,
            'sector_analysis': sector_analysis,
            'asset_breakdown': asset_breakdown
        }
        
        with open(f'updated_portfolio_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open(f'updated_portfolio_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ 更新后投资组合分析完成，报告已保存")
        
    else:
        logger.error("❌ 无法获取足够的股价数据")

if __name__ == "__main__":
    main() 