#!/usr/bin/env python3
"""
独立投资组合评估系统
基于客观数据和专业判断，不迎合预设目标，给出真实的投资建议
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndependentPortfolioAssessment:
    """独立投资组合评估师"""
    
    def __init__(self):
        """初始化评估系统"""
        # 读取当前持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.total_assets = self.config['portfolio']['total_value']
        
        # 当前持仓分析
        self.current_holdings = {}
        for symbol, position in self.config['positions'].items():
            # 排除港股
            if position.get('excluded_from_analysis', False):
                continue
                
            self.current_holdings[symbol] = {
                'shares': position['shares'],
                'investment_amount': position['investment_amount'],
                'weight': position['weight'] / 100.0,  # 转换为小数
                'sector': position.get('sector', 'Unknown')
            }
        
        # 市场基准数据
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            'VTI': 'Total Stock Market',
            'VEA': 'Developed Markets',
            'VWO': 'Emerging Markets'
        }
        
        logger.info("🔍 独立投资组合评估系统初始化完成")
    
    def analyze_historical_performance(self, symbols, period="5y"):
        """分析历史表现，获取真实的收益和风险数据"""
        performance_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    # 计算年化收益率
                    total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                    years = len(hist) / 252  # 交易日换算年数
                    annual_return = (1 + total_return) ** (1/years) - 1
                    
                    # 计算年化波动率
                    daily_returns = hist['Close'].pct_change().dropna()
                    annual_volatility = daily_returns.std() * np.sqrt(252)
                    
                    # 计算最大回撤
                    cumulative = (1 + daily_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # 计算夏普比率 (假设无风险利率4%)
                    risk_free_rate = 0.04
                    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                    
                    # 计算近期表现
                    recent_3m = hist['Close'].iloc[-63:] if len(hist) >= 63 else hist['Close']
                    recent_1y = hist['Close'].iloc[-252:] if len(hist) >= 252 else hist['Close']
                    
                    performance_3m = (recent_3m.iloc[-1] / recent_3m.iloc[0]) - 1 if len(recent_3m) > 1 else 0
                    performance_1y = (recent_1y.iloc[-1] / recent_1y.iloc[0]) - 1 if len(recent_1y) > 1 else 0
                    
                    performance_data[symbol] = {
                        'annual_return': annual_return,
                        'annual_volatility': annual_volatility,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'performance_3m': performance_3m,
                        'performance_1y': performance_1y,
                        'current_price': hist['Close'].iloc[-1],
                        'period_analyzed': f"{years:.1f} years"
                    }
                    
            except Exception as e:
                logger.warning(f"无法获取{symbol}的历史数据: {e}")
        
        return performance_data
    
    def analyze_current_portfolio(self):
        """分析当前投资组合的真实表现"""
        current_symbols = list(self.current_holdings.keys())
        performance = self.analyze_historical_performance(current_symbols)
        
        # 计算加权组合表现
        total_weight = 0
        weighted_return = 0
        weighted_volatility = 0
        weighted_sharpe = 0
        
        portfolio_analysis = {
            'individual_stocks': {},
            'sector_allocation': {},
            'risk_metrics': {},
            'concentration_risk': {}
        }
        
        for symbol, data in performance.items():
            if symbol in self.current_holdings:
                weight = self.current_holdings[symbol]['weight']
                
                portfolio_analysis['individual_stocks'][symbol] = {
                    'weight': weight,
                    'annual_return': data['annual_return'],
                    'volatility': data['annual_volatility'],
                    'sharpe_ratio': data['sharpe_ratio'],
                    'max_drawdown': data['max_drawdown'],
                    'sector': self.current_holdings[symbol]['sector']
                }
                
                weighted_return += data['annual_return'] * weight
                weighted_volatility += (data['annual_volatility'] ** 2) * (weight ** 2)  # 简化计算
                weighted_sharpe += data['sharpe_ratio'] * weight
                total_weight += weight
        
        # 组合层面指标
        portfolio_analysis['portfolio_metrics'] = {
            'weighted_annual_return': weighted_return,
            'estimated_volatility': np.sqrt(weighted_volatility),
            'weighted_sharpe_ratio': weighted_sharpe,
            'number_of_holdings': len(current_symbols)
        }
        
        # 行业集中度分析
        sector_weights = {}
        for symbol, holding in self.current_holdings.items():
            sector = holding['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + holding['weight']
        
        portfolio_analysis['sector_allocation'] = sector_weights
        
        # 集中度风险
        max_single_weight = max([h['weight'] for h in self.current_holdings.values()])
        top3_weight = sum(sorted([h['weight'] for h in self.current_holdings.values()], reverse=True)[:3])
        
        portfolio_analysis['concentration_risk'] = {
            'max_single_stock': max_single_weight,
            'top_3_stocks': top3_weight,
            'herfindahl_index': sum([w**2 for w in [h['weight'] for h in self.current_holdings.values()]])
        }
        
        return portfolio_analysis
    
    def benchmark_comparison(self):
        """与市场基准对比"""
        benchmark_performance = self.analyze_historical_performance(list(self.benchmarks.keys()))
        
        benchmark_analysis = {}
        for symbol, name in self.benchmarks.items():
            if symbol in benchmark_performance:
                data = benchmark_performance[symbol]
                benchmark_analysis[symbol] = {
                    'name': name,
                    'annual_return': data['annual_return'],
                    'volatility': data['annual_volatility'],
                    'sharpe_ratio': data['sharpe_ratio'],
                    'max_drawdown': data['max_drawdown']
                }
        
        return benchmark_analysis
    
    def realistic_return_assessment(self, portfolio_analysis, benchmark_analysis):
        """基于历史数据的现实收益评估"""
        
        current_return = portfolio_analysis['portfolio_metrics']['weighted_annual_return']
        current_volatility = portfolio_analysis['portfolio_metrics']['estimated_volatility']
        current_sharpe = portfolio_analysis['portfolio_metrics']['weighted_sharpe_ratio']
        
        # 市场基准比较
        spy_return = benchmark_analysis.get('SPY', {}).get('annual_return', 0.10)
        spy_volatility = benchmark_analysis.get('SPY', {}).get('volatility', 0.16)
        spy_sharpe = benchmark_analysis.get('SPY', {}).get('sharpe_ratio', 0.375)
        
        assessment = {
            'current_portfolio_assessment': {
                'expected_return': current_return,
                'risk_level': current_volatility,
                'risk_adjusted_return': current_sharpe,
                'vs_spy_return': current_return - spy_return,
                'vs_spy_sharpe': current_sharpe - spy_sharpe
            },
            'realistic_scenarios': {
                'conservative': current_return * 0.7,  # 考虑未来可能的挑战
                'base_case': current_return * 0.85,    # 历史表现打折
                'optimistic': current_return * 1.0     # 维持历史表现
            },
            'risk_assessment': {
                'concentration_risk': 'HIGH' if portfolio_analysis['concentration_risk']['top_3_stocks'] > 0.6 else 'MEDIUM',
                'sector_risk': 'HIGH' if max(portfolio_analysis['sector_allocation'].values()) > 0.5 else 'MEDIUM',
                'volatility_risk': 'HIGH' if current_volatility > 0.25 else 'MEDIUM' if current_volatility > 0.18 else 'LOW'
            }
        }
        
        return assessment
    
    def generate_independent_report(self, portfolio_analysis, benchmark_analysis, realistic_assessment):
        """生成独立的专业评估报告"""
        
        report = []
        report.append("=" * 100)
        report.append("📊 独立投资组合专业评估报告")
        report.append("🎯 基于历史数据的客观分析，不迎合预设目标")
        report.append(f"📅 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 100)
        
        # 当前持仓分析
        report.append(f"\n📈 当前持仓客观分析:")
        report.append("-" * 80)
        report.append(f"• 总资产: ${self.total_assets:,.0f}")
        report.append(f"• 持仓数量: {portfolio_analysis['portfolio_metrics']['number_of_holdings']}只")
        
        # 个股表现
        report.append(f"\n📋 个股历史表现分析 (基于5年数据):")
        report.append("-" * 80)
        report.append(f"{'股票':<6} {'权重':<8} {'年化收益':<10} {'波动率':<8} {'夏普比率':<10} {'最大回撤':<10}")
        report.append("-" * 70)
        
        for symbol, data in portfolio_analysis['individual_stocks'].items():
            report.append(f"{symbol:<6} {data['weight']:<7.1%} {data['annual_return']:<9.1%} "
                         f"{data['volatility']:<7.1%} {data['sharpe_ratio']:<9.2f} {data['max_drawdown']:<9.1%}")
        
        # 组合层面分析
        portfolio_metrics = portfolio_analysis['portfolio_metrics']
        report.append(f"\n📊 组合层面分析:")
        report.append("-" * 80)
        report.append(f"• 加权年化收益率: {portfolio_metrics['weighted_annual_return']:.1%}")
        report.append(f"• 估计波动率: {portfolio_metrics['estimated_volatility']:.1%}")
        report.append(f"• 加权夏普比率: {portfolio_metrics['weighted_sharpe_ratio']:.2f}")
        
        # 与基准对比
        report.append(f"\n📈 市场基准对比:")
        report.append("-" * 80)
        report.append(f"{'基准':<15} {'年化收益':<10} {'波动率':<8} {'夏普比率':<10}")
        report.append("-" * 50)
        
        for symbol, data in benchmark_analysis.items():
            report.append(f"{data['name']:<15} {data['annual_return']:<9.1%} "
                         f"{data['volatility']:<7.1%} {data['sharpe_ratio']:<9.2f}")
        
        # 现实收益评估
        assessment = realistic_assessment['current_portfolio_assessment']
        scenarios = realistic_assessment['realistic_scenarios']
        
        report.append(f"\n🎯 现实收益预期 (独立专业判断):")
        report.append("-" * 80)
        report.append(f"• 历史加权收益率: {assessment['expected_return']:.1%}")
        report.append(f"• 相对SPY超额收益: {assessment['vs_spy_return']:+.1%}")
        report.append(f"• 风险调整后优势: {assessment['vs_spy_sharpe']:+.2f}")
        
        report.append(f"\n📊 未来收益情景分析:")
        report.append("-" * 80)
        report.append(f"• 保守情景 (70%历史表现): {scenarios['conservative']:.1%}")
        report.append(f"• 基础情景 (85%历史表现): {scenarios['base_case']:.1%}")
        report.append(f"• 乐观情景 (100%历史表现): {scenarios['optimistic']:.1%}")
        
        # 风险评估
        risks = realistic_assessment['risk_assessment']
        report.append(f"\n⚠️ 风险评估:")
        report.append("-" * 80)
        report.append(f"• 集中度风险: {risks['concentration_risk']}")
        report.append(f"• 行业集中风险: {risks['sector_risk']}")
        report.append(f"• 波动率风险: {risks['volatility_risk']}")
        
        # 行业分配
        report.append(f"\n🏭 行业配置分析:")
        report.append("-" * 80)
        for sector, weight in sorted(portfolio_analysis['sector_allocation'].items(), 
                                   key=lambda x: x[1], reverse=True):
            report.append(f"• {sector}: {weight:.1%}")
        
        # 专业结论
        report.append(f"\n🏆 独立专业结论:")
        report.append("-" * 80)
        
        base_case_return = scenarios['base_case']
        
        if base_case_return >= 0.20:
            report.append(f"✅ 基于历史数据，该组合有能力实现20%+年化收益")
            report.append(f"• 基础情景预期收益: {base_case_return:.1%}")
            report.append(f"• 但需要注意高波动率风险: {assessment['risk_level']:.1%}")
            report.append(f"• 建议适度降低集中度以控制风险")
            
        elif base_case_return >= 0.15:
            report.append(f"⚠️ 该组合预期收益处于中等水平")
            report.append(f"• 基础情景预期收益: {base_case_return:.1%}")
            report.append(f"• 距离20%目标尚有差距: {0.20 - base_case_return:.1%}")
            report.append(f"• 需要优化配置或提高风险承受能力")
            
        else:
            report.append(f"❌ 坦率地说，该组合难以实现20%年化收益")
            report.append(f"• 基础情景预期收益: {base_case_return:.1%}")
            report.append(f"• 建议调整预期或重新配置组合")
            report.append(f"• 市场基准SPY历史收益: {benchmark_analysis.get('SPY', {}).get('annual_return', 0.10):.1%}")
        
        # 改进建议
        report.append(f"\n💡 客观改进建议:")
        report.append("-" * 80)
        
        if risks['concentration_risk'] == 'HIGH':
            report.append(f"• 降低单一股票权重，提高分散化程度")
        
        if risks['sector_risk'] == 'HIGH':
            report.append(f"• 增加行业多样性，减少行业集中风险")
        
        if assessment['vs_spy_return'] < 0:
            report.append(f"• 当前组合跑输市场基准，考虑增加指数基金配置")
        
        if assessment['risk_level'] > 0.25:
            report.append(f"• 波动率较高，考虑增加防御性资产")
        
        report.append(f"• 定期回顾和再平衡，避免情绪化决策")
        
        report.append("\n" + "=" * 100)
        report.append("📋 声明: 本报告基于历史数据客观分析，不保证未来表现")
        report.append("投资有风险，需要根据个人情况做出独立判断")
        report.append("=" * 100)
        
        return '\n'.join(report)

def main():
    """主函数"""
    assessor = IndependentPortfolioAssessment()
    
    # 分析当前组合
    portfolio_analysis = assessor.analyze_current_portfolio()
    
    # 基准对比
    benchmark_analysis = assessor.benchmark_comparison()
    
    # 现实收益评估
    realistic_assessment = assessor.realistic_return_assessment(portfolio_analysis, benchmark_analysis)
    
    # 生成独立报告
    report = assessor.generate_independent_report(portfolio_analysis, benchmark_analysis, realistic_assessment)
    
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细数据
    full_analysis = {
        'timestamp': timestamp,
        'portfolio_analysis': portfolio_analysis,
        'benchmark_analysis': benchmark_analysis,
        'realistic_assessment': realistic_assessment
    }
    
    with open(f'independent_portfolio_assessment_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(full_analysis, f, ensure_ascii=False, indent=2, default=str)
    
    with open(f'independent_assessment_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 独立投资组合评估完成")

if __name__ == "__main__":
    main()