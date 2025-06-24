#!/usr/bin/env python3
"""
平衡配置计算器 - 精确计算MRK和JPM的最优投资组合
"""

import yfinance as yf
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedPortfolioCalculator:
    """平衡投资组合计算器"""
    
    def __init__(self, total_budget: float = 1940.0):
        """初始化计算器"""
        self.total_budget = total_budget
        self.target_allocation = {
            'MRK': 0.5,  # 50%
            'JPM': 0.5   # 50%
        }
        logger.info(f"💰 初始化平衡投资组合计算器，总预算: ${total_budget}")
    
    def get_current_prices(self):
        """获取当前股价"""
        prices = {}
        
        for symbol in ['MRK', 'JPM']:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prices[symbol] = current_price
                    logger.info(f"📊 {symbol} 当前价格: ${current_price:.2f}")
                else:
                    logger.error(f"无法获取{symbol}的价格数据")
                    
            except Exception as e:
                logger.error(f"获取{symbol}价格失败: {e}")
                
        return prices
    
    def calculate_optimal_allocation(self, prices):
        """计算最优配置"""
        logger.info("🎯 开始计算最优股票配置...")
        
        results = {}
        
        # 方案1: 严格50/50分配
        allocation_50_50 = self._calculate_50_50_allocation(prices)
        results['50_50_allocation'] = allocation_50_50
        
        # 方案2: 最大化投资金额（尽可能用完预算）
        max_investment = self._calculate_max_investment_allocation(prices)
        results['max_investment'] = max_investment
        
        # 方案3: 整数股优化（避免零股）
        integer_optimized = self._calculate_integer_optimized(prices)
        results['integer_optimized'] = integer_optimized
        
        return results
    
    def _calculate_50_50_allocation(self, prices):
        """严格50/50分配"""
        mrk_budget = self.total_budget * 0.5  # $970
        jpm_budget = self.total_budget * 0.5  # $970
        
        mrk_shares = int(mrk_budget / prices['MRK'])
        jpm_shares = int(jpm_budget / prices['JPM'])
        
        mrk_investment = mrk_shares * prices['MRK']
        jpm_investment = jpm_shares * prices['JPM']
        
        total_investment = mrk_investment + jpm_investment
        remaining_cash = self.total_budget - total_investment
        
        return {
            'strategy': '严格50/50分配',
            'MRK': {
                'shares': mrk_shares,
                'price_per_share': prices['MRK'],
                'investment': mrk_investment,
                'target_allocation': 50.0,
                'actual_allocation': (mrk_investment / total_investment) * 100 if total_investment > 0 else 0
            },
            'JPM': {
                'shares': jpm_shares,
                'price_per_share': prices['JPM'],
                'investment': jpm_investment,
                'target_allocation': 50.0,
                'actual_allocation': (jpm_investment / total_investment) * 100 if total_investment > 0 else 0
            },
            'total_investment': total_investment,
            'remaining_cash': remaining_cash,
            'budget_utilization': (total_investment / self.total_budget) * 100
        }
    
    def _calculate_max_investment_allocation(self, prices):
        """最大化投资金额配置"""
        # 尝试不同的股数组合，找到最接近预算且保持平衡的配置
        best_combination = None
        best_utilization = 0
        
        max_mrk_shares = int(self.total_budget / prices['MRK']) + 1
        max_jpm_shares = int(self.total_budget / prices['JPM']) + 1
        
        for mrk_shares in range(max_mrk_shares):
            for jpm_shares in range(max_jpm_shares):
                total_cost = mrk_shares * prices['MRK'] + jpm_shares * prices['JPM']
                
                if total_cost <= self.total_budget:
                    utilization = total_cost / self.total_budget
                    
                    # 计算配置偏差（理想是50/50）
                    if total_cost > 0:
                        mrk_allocation = (mrk_shares * prices['MRK']) / total_cost
                        allocation_deviation = abs(mrk_allocation - 0.5)
                        
                        # 优先考虑高利用率，其次考虑平衡性
                        score = utilization - allocation_deviation * 0.1
                        
                        if score > best_utilization:
                            best_utilization = score
                            best_combination = {
                                'mrk_shares': mrk_shares,
                                'jpm_shares': jpm_shares,
                                'total_cost': total_cost,
                                'utilization': utilization,
                                'allocation_deviation': allocation_deviation
                            }
        
        if best_combination:
            mrk_investment = best_combination['mrk_shares'] * prices['MRK']
            jpm_investment = best_combination['jpm_shares'] * prices['JPM']
            total_investment = best_combination['total_cost']
            
            return {
                'strategy': '最大化投资金额',
                'MRK': {
                    'shares': best_combination['mrk_shares'],
                    'price_per_share': prices['MRK'],
                    'investment': mrk_investment,
                    'target_allocation': 50.0,
                    'actual_allocation': (mrk_investment / total_investment) * 100
                },
                'JPM': {
                    'shares': best_combination['jpm_shares'],
                    'price_per_share': prices['JPM'],
                    'investment': jpm_investment,
                    'target_allocation': 50.0,
                    'actual_allocation': (jpm_investment / total_investment) * 100
                },
                'total_investment': total_investment,
                'remaining_cash': self.total_budget - total_investment,
                'budget_utilization': best_combination['utilization'] * 100
            }
        
        return None
    
    def _calculate_integer_optimized(self, prices):
        """整数股优化配置"""
        # 基于当前价格比例，计算最优整数股配置
        price_ratio = prices['MRK'] / prices['JPM']
        
        # 寻找最佳整数比例
        best_combination = None
        best_score = 0
        
        for mrk_shares in range(1, int(self.total_budget / prices['MRK']) + 1):
            # 根据价格比例计算对应的JPM股数
            ideal_jpm_shares = mrk_shares / price_ratio
            
            # 尝试上下取整
            for jpm_shares in [int(ideal_jpm_shares), int(ideal_jpm_shares) + 1]:
                if jpm_shares <= 0:
                    continue
                    
                total_cost = mrk_shares * prices['MRK'] + jpm_shares * prices['JPM']
                
                if total_cost <= self.total_budget:
                    utilization = total_cost / self.total_budget
                    
                    # 计算配置平衡性
                    mrk_allocation = (mrk_shares * prices['MRK']) / total_cost
                    balance_score = 1 - abs(mrk_allocation - 0.5) * 2  # 0-1分数
                    
                    # 综合评分
                    score = utilization * 0.7 + balance_score * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_combination = {
                            'mrk_shares': mrk_shares,
                            'jpm_shares': jpm_shares,
                            'total_cost': total_cost,
                            'utilization': utilization,
                            'balance_score': balance_score
                        }
        
        if best_combination:
            mrk_investment = best_combination['mrk_shares'] * prices['MRK']
            jpm_investment = best_combination['jpm_shares'] * prices['JPM']
            total_investment = best_combination['total_cost']
            
            return {
                'strategy': '整数股优化配置',
                'MRK': {
                    'shares': best_combination['mrk_shares'],
                    'price_per_share': prices['MRK'],
                    'investment': mrk_investment,
                    'target_allocation': 50.0,
                    'actual_allocation': (mrk_investment / total_investment) * 100
                },
                'JPM': {
                    'shares': best_combination['jpm_shares'],
                    'price_per_share': prices['JPM'],
                    'investment': jpm_investment,
                    'target_allocation': 50.0,
                    'actual_allocation': (jpm_investment / total_investment) * 100
                },
                'total_investment': total_investment,
                'remaining_cash': self.total_budget - total_investment,
                'budget_utilization': best_combination['utilization'] * 100
            }
        
        return None
    
    def generate_report(self, results, prices):
        """生成投资报告"""
        report = []
        report.append("=" * 80)
        report.append("💰 MRK + JPM 平衡配置投资方案")
        report.append(f"📅 计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"💵 投资预算: ${self.total_budget:,.2f}")
        report.append("=" * 80)
        
        report.append(f"\n📊 当前股价:")
        report.append(f"• MRK (默克制药): ${prices['MRK']:.2f}")
        report.append(f"• JPM (摩根大通): ${prices['JPM']:.2f}")
        
        # 推荐方案
        recommended = None
        max_utilization = 0
        
        for strategy_name, strategy_data in results.items():
            if strategy_data and strategy_data['budget_utilization'] > max_utilization:
                max_utilization = strategy_data['budget_utilization']
                recommended = strategy_data
        
        if recommended:
            report.append(f"\n🎯 推荐方案: {recommended['strategy']}")
            report.append(f"📈 预算利用率: {recommended['budget_utilization']:.1f}%")
            report.append(f"💰 总投资: ${recommended['total_investment']:.2f}")
            report.append(f"💵 剩余现金: ${recommended['remaining_cash']:.2f}")
            
            report.append(f"\n🏢 股票配置:")
            report.append(f"• MRK: {recommended['MRK']['shares']}股 × ${recommended['MRK']['price_per_share']:.2f} = ${recommended['MRK']['investment']:.2f} ({recommended['MRK']['actual_allocation']:.1f}%)")
            report.append(f"• JPM: {recommended['JPM']['shares']}股 × ${recommended['JPM']['price_per_share']:.2f} = ${recommended['JPM']['investment']:.2f} ({recommended['JPM']['actual_allocation']:.1f}%)")
        
        # 所有方案对比
        report.append(f"\n📋 所有方案对比:")
        report.append("-" * 60)
        
        for strategy_name, strategy_data in results.items():
            if strategy_data:
                report.append(f"\n【{strategy_data['strategy']}】")
                report.append(f"MRK: {strategy_data['MRK']['shares']}股 = ${strategy_data['MRK']['investment']:.2f}")
                report.append(f"JPM: {strategy_data['JPM']['shares']}股 = ${strategy_data['JPM']['investment']:.2f}")
                report.append(f"总计: ${strategy_data['total_investment']:.2f} (利用率: {strategy_data['budget_utilization']:.1f}%)")
                report.append(f"剩余: ${strategy_data['remaining_cash']:.2f}")
        
        report.append(f"\n💡 投资建议:")
        report.append(f"• 建议采用预算利用率最高的方案")
        report.append(f"• 可以分批买入，降低市场波动风险")
        report.append(f"• 定期关注两只股票的基本面变化")
        report.append(f"• 剩余现金可用于应急或等待更好的买入机会")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)

def main():
    """主函数"""
    calculator = BalancedPortfolioCalculator(1940.0)
    
    # 获取当前价格
    prices = calculator.get_current_prices()
    
    if len(prices) == 2:
        # 计算最优配置
        results = calculator.calculate_optimal_allocation(prices)
        
        # 生成报告
        report = calculator.generate_report(results, prices)
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON数据
        output_data = {
            'timestamp': timestamp,
            'budget': 1940.0,
            'current_prices': prices,
            'allocation_results': results
        }
        
        with open(f'balanced_portfolio_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存报告
        with open(f'balanced_portfolio_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ 平衡配置计算完成，报告已保存")
        
    else:
        logger.error("❌ 无法获取股价数据，请检查网络连接")

if __name__ == "__main__":
    main() 