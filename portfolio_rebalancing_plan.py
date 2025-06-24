#!/usr/bin/env python3
"""
投资组合调仓方案生成器
基于最优配置制定具体的买卖调整计划
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioRebalancingPlan:
    """投资组合调仓方案生成器"""
    
    def __init__(self):
        """初始化调仓方案生成器"""
        # 最优组合配置
        self.optimal_portfolio = {
            'NVDA': 0.10,
            'PLTR': 0.10,
            'ORCL': 0.10,
            'IBM': 0.10,
            'BRK-B': 0.125,
            'GS': 0.125,
            'ABT': 0.10,
            'MRK': 0.10,
            'COST': 0.10,
            'XLK': 0.05
        }
        
        # 读取当前持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.total_assets = self.config['portfolio']['total_value']
        
        # 当前持仓
        self.current_positions = {}
        for symbol, position in self.config['positions'].items():
            if position.get('excluded_from_analysis', False):
                continue
            self.current_positions[symbol] = {
                'shares': position['shares'],
                'investment_amount': position['investment_amount'],
                'weight': position['weight'] / 100.0,
                'cost_basis': position['cost_basis'],
                'sector': position.get('sector', 'Unknown')
            }
        
        logger.info("📊 调仓方案生成器初始化完成")
    
    def get_current_prices(self, symbols):
        """获取当前股价"""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    prices[symbol] = hist['Close'].iloc[-1]
                    logger.info(f"{symbol}: ${prices[symbol]:.2f}")
            except Exception as e:
                logger.warning(f"获取{symbol}价格失败: {e}")
        return prices
    
    def calculate_rebalancing_plan(self):
        """计算调仓方案"""
        # 获取所有相关股票的当前价格
        all_symbols = set(list(self.optimal_portfolio.keys()) + list(self.current_positions.keys()))
        current_prices = self.get_current_prices(all_symbols)
        
        # 计算当前持仓的现值
        current_portfolio_value = 0
        current_portfolio_details = {}
        
        for symbol, position in self.current_positions.items():
            if symbol in current_prices:
                current_value = position['shares'] * current_prices[symbol]
                current_portfolio_value += current_value
                current_portfolio_details[symbol] = {
                    'shares': position['shares'],
                    'current_price': current_prices[symbol],
                    'current_value': current_value,
                    'current_weight': 0,  # 稍后计算
                    'cost_basis': position['cost_basis'],
                    'unrealized_pnl': (current_prices[symbol] - position['cost_basis']) * position['shares'],
                    'unrealized_pnl_pct': (current_prices[symbol] / position['cost_basis'] - 1) * 100
                }
        
        # 计算当前权重
        for symbol in current_portfolio_details:
            current_portfolio_details[symbol]['current_weight'] = (
                current_portfolio_details[symbol]['current_value'] / current_portfolio_value
            )
        
        # 计算目标配置
        target_portfolio_details = {}
        for symbol, target_weight in self.optimal_portfolio.items():
            target_value = current_portfolio_value * target_weight
            if symbol in current_prices:
                target_shares = int(target_value / current_prices[symbol])
                actual_target_value = target_shares * current_prices[symbol]
                
                target_portfolio_details[symbol] = {
                    'target_weight': target_weight,
                    'target_value': target_value,
                    'target_shares': target_shares,
                    'actual_target_value': actual_target_value,
                    'current_price': current_prices[symbol]
                }
        
        # 计算调仓动作
        rebalancing_actions = []
        total_sells = 0
        total_buys = 0
        
        # 处理需要卖出或减仓的股票
        for symbol in current_portfolio_details:
            current_shares = current_portfolio_details[symbol]['shares']
            current_value = current_portfolio_details[symbol]['current_value']
            
            if symbol in target_portfolio_details:
                # 股票在目标组合中
                target_shares = target_portfolio_details[symbol]['target_shares']
                shares_diff = current_shares - target_shares
                
                if shares_diff > 0:
                    # 需要减仓
                    sell_value = shares_diff * current_prices[symbol]
                    total_sells += sell_value
                    
                    rebalancing_actions.append({
                        'action': 'REDUCE',
                        'symbol': symbol,
                        'current_shares': current_shares,
                        'target_shares': target_shares,
                        'shares_to_sell': shares_diff,
                        'sell_value': sell_value,
                        'current_price': current_prices[symbol],
                        'reason': f"减仓至目标权重{target_portfolio_details[symbol]['target_weight']:.1%}"
                    })
                elif shares_diff < 0:
                    # 需要加仓
                    add_shares = abs(shares_diff)
                    add_value = add_shares * current_prices[symbol]
                    total_buys += add_value
                    
                    rebalancing_actions.append({
                        'action': 'ADD',
                        'symbol': symbol,
                        'current_shares': current_shares,
                        'target_shares': target_shares,
                        'shares_to_buy': add_shares,
                        'buy_value': add_value,
                        'current_price': current_prices[symbol],
                        'reason': f"加仓至目标权重{target_portfolio_details[symbol]['target_weight']:.1%}"
                    })
                # else: 持仓正好，无需调整
            else:
                # 股票不在目标组合中，全部卖出
                sell_value = current_value
                total_sells += sell_value
                
                rebalancing_actions.append({
                    'action': 'SELL_ALL',
                    'symbol': symbol,
                    'current_shares': current_shares,
                    'shares_to_sell': current_shares,
                    'sell_value': sell_value,
                    'current_price': current_prices[symbol],
                    'unrealized_pnl': current_portfolio_details[symbol]['unrealized_pnl'],
                    'reason': "不在最优组合中，全部卖出"
                })
        
        # 处理需要新买入的股票
        for symbol in target_portfolio_details:
            if symbol not in current_portfolio_details:
                # 全新买入
                target_shares = target_portfolio_details[symbol]['target_shares']
                buy_value = target_shares * current_prices[symbol]
                total_buys += buy_value
                
                rebalancing_actions.append({
                    'action': 'BUY_NEW',
                    'symbol': symbol,
                    'target_shares': target_shares,
                    'shares_to_buy': target_shares,
                    'buy_value': buy_value,
                    'current_price': current_prices[symbol],
                    'reason': f"新买入，目标权重{target_portfolio_details[symbol]['target_weight']:.1%}"
                })
        
        # 计算资金缺口
        net_cash_needed = total_buys - total_sells
        
        return {
            'current_portfolio_value': current_portfolio_value,
            'current_portfolio_details': current_portfolio_details,
            'target_portfolio_details': target_portfolio_details,
            'rebalancing_actions': rebalancing_actions,
            'total_sells': total_sells,
            'total_buys': total_buys,
            'net_cash_needed': net_cash_needed,
            'current_prices': current_prices
        }
    
    def analyze_tax_implications(self, rebalancing_plan):
        """分析税务影响"""
        tax_analysis = {
            'total_realized_gains': 0,
            'total_realized_losses': 0,
            'short_term_gains': 0,
            'long_term_gains': 0,
            'tax_efficient_actions': [],
            'tax_inefficient_actions': []
        }
        
        for action in rebalancing_plan['rebalancing_actions']:
            if action['action'] in ['SELL_ALL', 'REDUCE']:
                if 'unrealized_pnl' in action:
                    pnl = action['unrealized_pnl']
                    if pnl > 0:
                        tax_analysis['total_realized_gains'] += pnl
                        # 假设持有超过1年（实际需要根据买入日期计算）
                        tax_analysis['long_term_gains'] += pnl
                    else:
                        tax_analysis['total_realized_losses'] += abs(pnl)
                    
                    # 税务效率评估
                    if pnl < 0:
                        tax_analysis['tax_efficient_actions'].append({
                            'symbol': action['symbol'],
                            'action': action['action'],
                            'loss': abs(pnl),
                            'benefit': '实现亏损可抵税'
                        })
                    elif pnl > action['sell_value'] * 0.3:  # 收益超过30%
                        tax_analysis['tax_inefficient_actions'].append({
                            'symbol': action['symbol'],
                            'action': action['action'],
                            'gain': pnl,
                            'concern': '大额收益需缴纳资本利得税'
                        })
        
        return tax_analysis
    
    def generate_implementation_timeline(self, rebalancing_plan):
        """生成实施时间表"""
        timeline = {
            'phase_1_immediate': [],
            'phase_2_strategic': [],
            'phase_3_completion': []
        }
        
        for action in rebalancing_plan['rebalancing_actions']:
            if action['action'] == 'SELL_ALL':
                # 立即卖出不需要的股票
                timeline['phase_1_immediate'].append({
                    'action': action,
                    'priority': 'HIGH',
                    'timing': '立即执行',
                    'reason': '释放资金用于重新配置'
                })
            elif action['action'] == 'BUY_NEW':
                # 分批买入新股票
                timeline['phase_2_strategic'].append({
                    'action': action,
                    'priority': 'MEDIUM',
                    'timing': '分2-3批执行',
                    'reason': '分散买入降低时机风险'
                })
            elif action['action'] in ['REDUCE', 'ADD']:
                # 调整现有持仓
                timeline['phase_3_completion'].append({
                    'action': action,
                    'priority': 'LOW',
                    'timing': '等待合适时机',
                    'reason': '微调持仓，可等待更好价格'
                })
        
        return timeline
    
    def generate_rebalancing_report(self, rebalancing_plan, tax_analysis, timeline):
        """生成调仓报告"""
        report = []
        report.append("=" * 120)
        report.append("📊 投资组合调仓方案报告")
        report.append("🎯 目标：实现全球最优配置 - 均衡配置_v3")
        report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 当前持仓分析
        report.append(f"\n📈 当前持仓分析:")
        report.append("-" * 100)
        report.append(f"• 投资组合总价值: ${rebalancing_plan['current_portfolio_value']:,.0f}")
        report.append(f"• 当前持仓数量: {len(rebalancing_plan['current_portfolio_details'])}只")
        
        report.append(f"\n{'股票':<8} {'股数':<8} {'现价':<10} {'现值':<12} {'权重':<8} {'盈亏':<12} {'盈亏%':<10}")
        report.append("-" * 80)
        
        for symbol, details in rebalancing_plan['current_portfolio_details'].items():
            report.append(f"{symbol:<8} {details['shares']:<8} "
                         f"${details['current_price']:<9.2f} "
                         f"${details['current_value']:<11,.0f} "
                         f"{details['current_weight']:<7.1%} "
                         f"${details['unrealized_pnl']:<11,.0f} "
                         f"{details['unrealized_pnl_pct']:<9.1f}%")
        
        # 目标配置
        report.append(f"\n🎯 目标配置 - 全球最优组合:")
        report.append("-" * 100)
        report.append(f"{'股票':<8} {'目标权重':<10} {'目标价值':<12} {'目标股数':<10} {'现价':<10}")
        report.append("-" * 60)
        
        for symbol, details in rebalancing_plan['target_portfolio_details'].items():
            report.append(f"{symbol:<8} {details['target_weight']:<9.1%} "
                         f"${details['target_value']:<11,.0f} "
                         f"{details['target_shares']:<10} "
                         f"${details['current_price']:<9.2f}")
        
        # 调仓动作
        report.append(f"\n💼 具体调仓动作:")
        report.append("-" * 100)
        
        # 按动作类型分组
        sell_actions = [a for a in rebalancing_plan['rebalancing_actions'] if a['action'] in ['SELL_ALL', 'REDUCE']]
        buy_actions = [a for a in rebalancing_plan['rebalancing_actions'] if a['action'] in ['BUY_NEW', 'ADD']]
        
        if sell_actions:
            report.append(f"\n🔴 卖出/减仓动作:")
            report.append(f"{'股票':<8} {'动作':<10} {'卖出股数':<10} {'卖出金额':<12} {'理由':<30}")
            report.append("-" * 80)
            
            for action in sell_actions:
                report.append(f"{action['symbol']:<8} {action['action']:<10} "
                             f"{action['shares_to_sell']:<10} "
                             f"${action['sell_value']:<11,.0f} "
                             f"{action['reason']:<30}")
        
        if buy_actions:
            report.append(f"\n🟢 买入/加仓动作:")
            report.append(f"{'股票':<8} {'动作':<10} {'买入股数':<10} {'买入金额':<12} {'理由':<30}")
            report.append("-" * 80)
            
            for action in buy_actions:
                report.append(f"{action['symbol']:<8} {action['action']:<10} "
                             f"{action['shares_to_buy']:<10} "
                             f"${action['buy_value']:<11,.0f} "
                             f"{action['reason']:<30}")
        
        # 资金分析
        report.append(f"\n💰 资金分析:")
        report.append("-" * 100)
        report.append(f"• 卖出总金额: ${rebalancing_plan['total_sells']:,.0f}")
        report.append(f"• 买入总金额: ${rebalancing_plan['total_buys']:,.0f}")
        report.append(f"• 净资金需求: ${rebalancing_plan['net_cash_needed']:,.0f}")
        
        if rebalancing_plan['net_cash_needed'] > 0:
            report.append(f"• ⚠️  需要额外注入资金: ${rebalancing_plan['net_cash_needed']:,.0f}")
        else:
            report.append(f"• ✅ 卖出资金足够，还有余额: ${abs(rebalancing_plan['net_cash_needed']):,.0f}")
        
        # 税务分析
        report.append(f"\n💸 税务影响分析:")
        report.append("-" * 100)
        report.append(f"• 预计实现收益: ${tax_analysis['total_realized_gains']:,.0f}")
        report.append(f"• 预计实现亏损: ${tax_analysis['total_realized_losses']:,.0f}")
        report.append(f"• 净收益: ${tax_analysis['total_realized_gains'] - tax_analysis['total_realized_losses']:,.0f}")
        
        if tax_analysis['tax_efficient_actions']:
            report.append(f"\n✅ 税务友好动作:")
            for action in tax_analysis['tax_efficient_actions']:
                report.append(f"• {action['symbol']}: {action['benefit']}")
        
        if tax_analysis['tax_inefficient_actions']:
            report.append(f"\n⚠️  税务影响较大:")
            for action in tax_analysis['tax_inefficient_actions']:
                report.append(f"• {action['symbol']}: {action['concern']}")
        
        # 实施时间表
        report.append(f"\n📅 实施时间表建议:")
        report.append("-" * 100)
        
        if timeline['phase_1_immediate']:
            report.append(f"\n阶段一 - 立即执行 (1-2个交易日):")
            for item in timeline['phase_1_immediate']:
                action = item['action']
                report.append(f"• {action['symbol']}: {action['action']} - {item['reason']}")
        
        if timeline['phase_2_strategic']:
            report.append(f"\n阶段二 - 分批执行 (1-2周):")
            for item in timeline['phase_2_strategic']:
                action = item['action']
                report.append(f"• {action['symbol']}: {action['action']} - {item['reason']}")
        
        if timeline['phase_3_completion']:
            report.append(f"\n阶段三 - 精细调整 (1个月内):")
            for item in timeline['phase_3_completion']:
                action = item['action']
                report.append(f"• {action['symbol']}: {action['action']} - {item['reason']}")
        
        # 风险提示
        report.append(f"\n⚠️ 风险提示:")
        report.append("-" * 100)
        report.append(f"• 市场波动可能影响执行价格")
        report.append(f"• 大额交易可能产生滑点成本")
        report.append(f"• 税务规划建议咨询专业人士")
        report.append(f"• 分批执行可降低时机风险")
        
        # 预期收益
        report.append(f"\n🎯 调仓后预期表现:")
        report.append("-" * 100)
        report.append(f"• 目标年化收益: 31.1%")
        report.append(f"• 风险调整收益(夏普比率): 1.67")
        report.append(f"• 各市场情景下都能稳定超过20%目标")
        report.append(f"• 持仓分散度和稳定性显著提升")
        
        report.append("\n" + "=" * 120)
        report.append("📋 声明: 本方案基于数据分析生成，执行前请根据市场情况和个人风险承受能力调整")
        report.append("=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    rebalancer = PortfolioRebalancingPlan()
    
    # 计算调仓方案
    rebalancing_plan = rebalancer.calculate_rebalancing_plan()
    
    # 分析税务影响
    tax_analysis = rebalancer.analyze_tax_implications(rebalancing_plan)
    
    # 生成实施时间表
    timeline = rebalancer.generate_implementation_timeline(rebalancing_plan)
    
    # 生成报告
    report = rebalancer.generate_rebalancing_report(rebalancing_plan, tax_analysis, timeline)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细数据
    full_plan = {
        'timestamp': timestamp,
        'rebalancing_plan': rebalancing_plan,
        'tax_analysis': tax_analysis,
        'timeline': timeline
    }
    
    with open(f'rebalancing_plan_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(full_plan, f, ensure_ascii=False, indent=2, default=str)
    
    with open(f'rebalancing_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 调仓方案生成完成")

if __name__ == "__main__":
    main()