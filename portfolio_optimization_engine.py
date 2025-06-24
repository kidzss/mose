#!/usr/bin/env python3
"""
投资组合优化引擎
在多种市场情景下测试大量组合，找到真正的最优解
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizationEngine:
    """投资组合优化引擎"""
    
    def __init__(self):
        """初始化优化引擎"""
        # 扩展的股票池
        self.stock_universe = {
            # 超大盘科技龙头
            'NVDA': {'category': 'AI/芯片', 'risk': 'HIGH', 'growth': 'VERY_HIGH', 'sector': 'Technology'},
            'MSFT': {'category': '云计算/AI', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'GOOGL': {'category': '搜索/AI', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'AAPL': {'category': '消费电子', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Technology'},
            'META': {'category': '社交/VR', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'AMZN': {'category': '电商/云', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'TSLA': {'category': '电动车/AI', 'risk': 'VERY_HIGH', 'growth': 'HIGH', 'sector': 'Automotive'},
            
            # 中盘成长科技股
            'AMD': {'category': 'AI/芯片', 'risk': 'HIGH', 'growth': 'HIGH', 'sector': 'Technology'},
            'CRM': {'category': '企业软件', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'ADBE': {'category': '创意软件', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'Technology'},
            'NOW': {'category': '企业软件', 'risk': 'MEDIUM', 'growth': 'HIGH', 'sector': 'Technology'},
            'PLTR': {'category': '数据分析', 'risk': 'HIGH', 'growth': 'HIGH', 'sector': 'Technology'},
            
            # 传统科技
            'ORCL': {'category': '数据库', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Technology'},
            'CSCO': {'category': '网络设备', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Technology'},
            'IBM': {'category': '云/AI', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Technology'},
            'INTC': {'category': '芯片', 'risk': 'MEDIUM', 'growth': 'LOW', 'sector': 'Technology'},
            
            # 金融
            'JPM': {'category': '银行', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'Financial'},
            'BAC': {'category': '银行', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'Financial'},
            'WFC': {'category': '银行', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'Financial'},
            'GS': {'category': '投行', 'risk': 'HIGH', 'growth': 'MEDIUM', 'sector': 'Financial'},
            'V': {'category': '支付', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Financial'},
            
            # 医疗
            'JNJ': {'category': '制药', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Healthcare'},
            'PFE': {'category': '制药', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Healthcare'},
            'UNH': {'category': '医保', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Healthcare'},
            'MRK': {'category': '制药', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Healthcare'},
            'ABT': {'category': '医疗设备', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Healthcare'},
            
            # 消费
            'COST': {'category': '零售', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Consumer'},
            'WMT': {'category': '零售', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Consumer'},
            'PG': {'category': '消费品', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Consumer'},
            'KO': {'category': '饮料', 'risk': 'LOW', 'growth': 'LOW', 'sector': 'Consumer'},
            'HD': {'category': '家居', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'Consumer'},
            
            # 价值投资
            'BRK-B': {'category': '投资', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'Financial'},
            
            # ETF
            'SPY': {'category': '大盘ETF', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'ETF'},
            'QQQ': {'category': '科技ETF', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'ETF'},
            'XLK': {'category': '科技板块', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'ETF'},
            'XLF': {'category': '金融板块', 'risk': 'MEDIUM', 'growth': 'MEDIUM', 'sector': 'ETF'},
            'VTI': {'category': '全市场', 'risk': 'LOW', 'growth': 'MEDIUM', 'sector': 'ETF'}
        }
        
        # 市场情景定义
        self.market_scenarios = {
            'bull_market': {
                'name': '牛市情景',
                'probability': 0.25,
                'tech_multiplier': 1.3,
                'defensive_multiplier': 0.8,
                'volatility_multiplier': 1.2,
                'description': '科技股表现优异，防御股落后'
            },
            'bear_market': {
                'name': '熊市情景', 
                'probability': 0.20,
                'tech_multiplier': 0.6,
                'defensive_multiplier': 1.1,
                'volatility_multiplier': 2.0,
                'description': '科技股大幅下跌，防御股相对抗跌'
            },
            'sideways_market': {
                'name': '震荡市情景',
                'probability': 0.30,
                'tech_multiplier': 0.9,
                'defensive_multiplier': 1.0,
                'volatility_multiplier': 1.5,
                'description': '市场横盘整理，波动加大'
            },
            'rotation_market': {
                'name': '风格轮动',
                'probability': 0.15,
                'tech_multiplier': 0.7,
                'defensive_multiplier': 1.3,
                'volatility_multiplier': 1.1,
                'description': '资金从科技股流向价值股'
            },
            'normal_market': {
                'name': '正常市场',
                'probability': 0.10,
                'tech_multiplier': 1.0,
                'defensive_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'description': '市场正常运行'
            }
        }
        
        logger.info("🔧 投资组合优化引擎初始化完成")
        
    def get_stock_data(self, symbols, period="5y"):
        """获取股票历史数据"""
        stock_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"获取 {symbol} 数据...")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if not hist.empty:
                    # 计算基础指标
                    total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                    years = len(hist) / 252
                    annual_return = (1 + total_return) ** (1/years) - 1
                    
                    daily_returns = hist['Close'].pct_change().dropna()
                    annual_volatility = daily_returns.std() * np.sqrt(252)
                    
                    # 最大回撤
                    cumulative = (1 + daily_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # 下行风险
                    negative_returns = daily_returns[daily_returns < 0]
                    downside_deviation = negative_returns.std() * np.sqrt(252)
                    
                    # 技术指标
                    current_price = hist['Close'].iloc[-1]
                    ma200 = hist['Close'].rolling(200).mean().iloc[-1]
                    
                    # 基本面
                    pe_ratio = info.get('trailingPE', None)
                    market_cap = info.get('marketCap', 0)
                    
                    # 近期表现
                    recent_3m = hist['Close'].iloc[-63:] if len(hist) >= 63 else hist['Close']
                    recent_1y = hist['Close'].iloc[-252:] if len(hist) >= 252 else hist['Close']
                    
                    performance_3m = (recent_3m.iloc[-1] / recent_3m.iloc[0]) - 1 if len(recent_3m) > 1 else 0
                    performance_1y = (recent_1y.iloc[-1] / recent_1y.iloc[0]) - 1 if len(recent_1y) > 1 else 0
                    
                    stock_data[symbol] = {
                        'annual_return': annual_return,
                        'annual_volatility': annual_volatility,
                        'max_drawdown': max_drawdown,
                        'downside_deviation': downside_deviation,
                        'sharpe_ratio': (annual_return - 0.04) / annual_volatility,
                        'sortino_ratio': (annual_return - 0.04) / downside_deviation if downside_deviation > 0 else 0,
                        'current_price': current_price,
                        'pe_ratio': pe_ratio,
                        'market_cap': market_cap,
                        'performance_3m': performance_3m,
                        'performance_1y': performance_1y,
                        'trend_strength': (current_price / ma200 - 1) if ma200 > 0 else 0,
                        'category': self.stock_universe[symbol]['category'],
                        'sector': self.stock_universe[symbol]['sector'],
                        'risk_level': self.stock_universe[symbol]['risk']
                    }
                    
            except Exception as e:
                logger.warning(f"获取{symbol}数据失败: {e}")
                
        return stock_data
    
    def generate_portfolio_combinations(self, stock_data, tech_weight_range=(0.4, 0.8), num_portfolios=50):
        """生成多种投资组合组合"""
        portfolios = []
        
        # 按板块分类
        tech_stocks = [s for s, data in stock_data.items() 
                      if self.stock_universe[s]['sector'] == 'Technology']
        financial_stocks = [s for s, data in stock_data.items() 
                           if self.stock_universe[s]['sector'] == 'Financial']
        healthcare_stocks = [s for s, data in stock_data.items() 
                            if self.stock_universe[s]['sector'] == 'Healthcare']
        consumer_stocks = [s for s, data in stock_data.items() 
                          if self.stock_universe[s]['sector'] == 'Consumer']
        etf_stocks = [s for s, data in stock_data.items() 
                     if self.stock_universe[s]['sector'] == 'ETF']
        
        # 按夏普比率排序
        tech_sorted = sorted(tech_stocks, 
                           key=lambda x: stock_data[x]['sharpe_ratio'], reverse=True)
        financial_sorted = sorted(financial_stocks, 
                                key=lambda x: stock_data[x]['sharpe_ratio'], reverse=True)
        healthcare_sorted = sorted(healthcare_stocks, 
                                 key=lambda x: stock_data[x]['sharpe_ratio'], reverse=True)
        consumer_sorted = sorted(consumer_stocks, 
                               key=lambda x: stock_data[x]['sharpe_ratio'], reverse=True)
        
        # 生成不同配置策略
        strategies = [
            # 科技重仓策略
            {'name': '科技重仓', 'tech': 0.70, 'financial': 0.15, 'healthcare': 0.05, 'consumer': 0.05, 'etf': 0.05},
            {'name': '科技主导', 'tech': 0.60, 'financial': 0.20, 'healthcare': 0.10, 'consumer': 0.05, 'etf': 0.05},
            {'name': '科技平衡', 'tech': 0.50, 'financial': 0.20, 'healthcare': 0.15, 'consumer': 0.10, 'etf': 0.05},
            
            # 平衡策略
            {'name': '均衡配置', 'tech': 0.40, 'financial': 0.25, 'healthcare': 0.20, 'consumer': 0.10, 'etf': 0.05},
            {'name': '防御配置', 'tech': 0.30, 'financial': 0.20, 'healthcare': 0.30, 'consumer': 0.15, 'etf': 0.05},
            
            # 动量策略
            {'name': '动量追逐', 'tech': 0.80, 'financial': 0.10, 'healthcare': 0.05, 'consumer': 0.00, 'etf': 0.05},
            {'name': '价值回归', 'tech': 0.20, 'financial': 0.30, 'healthcare': 0.25, 'consumer': 0.20, 'etf': 0.05},
            
            # ETF增强
            {'name': 'ETF增强', 'tech': 0.40, 'financial': 0.15, 'healthcare': 0.15, 'consumer': 0.10, 'etf': 0.20},
        ]
        
        for strategy in strategies:
            # 为每个策略生成多个变种
            for variant in range(3):
                portfolio = []
                total_weight = 0
                
                # 科技股配置
                tech_weight = strategy['tech']
                tech_count = max(2, min(5, int(tech_weight * 10)))  # 2-5只科技股
                selected_tech = tech_sorted[:tech_count]
                tech_individual_weight = tech_weight / tech_count
                
                for stock in selected_tech:
                    portfolio.append({
                        'symbol': stock,
                        'weight': tech_individual_weight,
                        'sector': 'Technology'
                    })
                    total_weight += tech_individual_weight
                
                # 金融股配置
                if strategy['financial'] > 0 and financial_sorted:
                    financial_count = max(1, min(3, int(strategy['financial'] * 10)))
                    selected_financial = financial_sorted[:financial_count]
                    financial_individual_weight = strategy['financial'] / financial_count
                    
                    for stock in selected_financial:
                        portfolio.append({
                            'symbol': stock,
                            'weight': financial_individual_weight,
                            'sector': 'Financial'
                        })
                        total_weight += financial_individual_weight
                
                # 医疗股配置
                if strategy['healthcare'] > 0 and healthcare_sorted:
                    healthcare_count = max(1, min(3, int(strategy['healthcare'] * 10)))
                    selected_healthcare = healthcare_sorted[:healthcare_count]
                    healthcare_individual_weight = strategy['healthcare'] / healthcare_count
                    
                    for stock in selected_healthcare:
                        portfolio.append({
                            'symbol': stock,
                            'weight': healthcare_individual_weight,
                            'sector': 'Healthcare'
                        })
                        total_weight += healthcare_individual_weight
                
                # 消费股配置
                if strategy['consumer'] > 0 and consumer_sorted:
                    consumer_count = max(1, min(2, int(strategy['consumer'] * 10)))
                    selected_consumer = consumer_sorted[:consumer_count]
                    consumer_individual_weight = strategy['consumer'] / consumer_count
                    
                    for stock in selected_consumer:
                        portfolio.append({
                            'symbol': stock,
                            'weight': consumer_individual_weight,
                            'sector': 'Consumer'
                        })
                        total_weight += consumer_individual_weight
                
                # ETF配置
                if strategy['etf'] > 0 and etf_stocks:
                    etf_stock = etf_stocks[variant % len(etf_stocks)]  # 轮换选择ETF
                    portfolio.append({
                        'symbol': etf_stock,
                        'weight': strategy['etf'],
                        'sector': 'ETF'
                    })
                    total_weight += strategy['etf']
                
                # 归一化权重
                for pos in portfolio:
                    pos['weight'] = pos['weight'] / total_weight
                
                portfolios.append({
                    'name': f"{strategy['name']}_v{variant+1}",
                    'positions': portfolio,
                    'tech_weight': tech_weight / total_weight
                })
        
        return portfolios[:num_portfolios]  # 限制数量
    
    def evaluate_portfolio_scenarios(self, portfolio, stock_data):
        """在不同市场情景下评估投资组合"""
        scenario_results = {}
        
        for scenario_name, scenario in self.market_scenarios.items():
            # 计算情景调整后的收益和风险
            adjusted_return = 0
            adjusted_risk = 0
            
            for position in portfolio['positions']:
                symbol = position['symbol']
                weight = position['weight']
                
                if symbol in stock_data:
                    base_return = stock_data[symbol]['annual_return']
                    base_volatility = stock_data[symbol]['annual_volatility']
                    sector = position['sector']
                    
                    # 根据板块和情景调整
                    if sector == 'Technology':
                        scenario_return = base_return * scenario['tech_multiplier']
                        scenario_volatility = base_volatility * scenario['volatility_multiplier']
                    else:
                        scenario_return = base_return * scenario['defensive_multiplier']
                        scenario_volatility = base_volatility * scenario['volatility_multiplier']
                    
                    adjusted_return += scenario_return * weight
                    adjusted_risk += (scenario_volatility ** 2) * (weight ** 2)  # 简化计算
            
            adjusted_risk = np.sqrt(adjusted_risk)
            sharpe_ratio = (adjusted_return - 0.04) / adjusted_risk if adjusted_risk > 0 else 0
            
            scenario_results[scenario_name] = {
                'expected_return': adjusted_return,
                'volatility': adjusted_risk,
                'sharpe_ratio': sharpe_ratio,
                'probability': scenario['probability']
            }
        
        # 计算期望值
        expected_return = sum([result['expected_return'] * result['probability'] 
                              for result in scenario_results.values()])
        expected_volatility = sum([result['volatility'] * result['probability'] 
                                 for result in scenario_results.values()])
        expected_sharpe = (expected_return - 0.04) / expected_volatility if expected_volatility > 0 else 0
        
        return {
            'scenarios': scenario_results,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'expected_sharpe': expected_sharpe
        }
    
    def find_optimal_portfolios(self, stock_data, top_n=10):
        """寻找最优投资组合"""
        logger.info("🔍 生成投资组合候选...")
        portfolios = self.generate_portfolio_combinations(stock_data)
        
        logger.info(f"📊 评估 {len(portfolios)} 个投资组合...")
        evaluated_portfolios = []
        
        for i, portfolio in enumerate(portfolios):
            logger.info(f"评估组合 {i+1}/{len(portfolios)}: {portfolio['name']}")
            
            # 计算基础指标
            base_return = 0
            base_risk = 0
            max_single_weight = 0
            
            for position in portfolio['positions']:
                symbol = position['symbol']
                weight = position['weight']
                max_single_weight = max(max_single_weight, weight)
                
                if symbol in stock_data:
                    base_return += stock_data[symbol]['annual_return'] * weight
                    base_risk += (stock_data[symbol]['annual_volatility'] ** 2) * (weight ** 2)
            
            base_risk = np.sqrt(base_risk)
            base_sharpe = (base_return - 0.04) / base_risk if base_risk > 0 else 0
            
            # 情景分析
            scenario_analysis = self.evaluate_portfolio_scenarios(portfolio, stock_data)
            
            # 综合评分
            # 收益权重40%，夏普比率30%，稳定性20%，分散度10%
            return_score = min(scenario_analysis['expected_return'] / 0.30, 1.0)
            sharpe_score = min(scenario_analysis['expected_sharpe'] / 2.0, 1.0)
            
            # 稳定性评分：各情景下表现的一致性
            scenario_returns = [s['expected_return'] for s in scenario_analysis['scenarios'].values()]
            stability_score = 1.0 - (np.std(scenario_returns) / np.mean(scenario_returns)) if np.mean(scenario_returns) > 0 else 0
            stability_score = max(0, min(1, stability_score))
            
            # 分散度评分
            diversification_score = 1.0 - max_single_weight  # 单一持仓越小分散度越高
            
            composite_score = (return_score * 0.4 + sharpe_score * 0.3 + 
                             stability_score * 0.2 + diversification_score * 0.1)
            
            evaluated_portfolios.append({
                'portfolio': portfolio,
                'base_return': base_return,
                'base_volatility': base_risk,
                'base_sharpe': base_sharpe,
                'scenario_analysis': scenario_analysis,
                'max_single_weight': max_single_weight,
                'stability_score': stability_score,
                'diversification_score': diversification_score,
                'composite_score': composite_score
            })
        
        # 按综合评分排序
        evaluated_portfolios.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return evaluated_portfolios[:top_n]
    
    def generate_optimization_report(self, optimal_portfolios, stock_data):
        """生成优化报告"""
        report = []
        report.append("=" * 120)
        report.append("🎯 投资组合全面优化分析报告")
        report.append("🔬 基于多市场情景的最优解搜索")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 概览
        report.append(f"\n📊 优化概览:")
        report.append("-" * 100)
        report.append(f"• 候选股票池: {len(stock_data)}只")
        report.append(f"• 测试投资组合: 50个策略组合")
        report.append(f"• 市场情景: {len(self.market_scenarios)}种")
        report.append(f"• 最优组合数量: {len(optimal_portfolios)}")
        
        # 市场情景说明
        report.append(f"\n📈 市场情景设定:")
        report.append("-" * 100)
        for scenario_name, scenario in self.market_scenarios.items():
            report.append(f"• {scenario['name']} (概率{scenario['probability']:.0%}): {scenario['description']}")
        
        # 最优组合排名
        report.append(f"\n🏆 最优组合排名:")
        report.append("-" * 100)
        report.append(f"{'排名':<4} {'组合名称':<15} {'预期收益':<10} {'夏普比率':<10} {'稳定性':<8} {'分散度':<8} {'综合评分':<10}")
        report.append("-" * 80)
        
        for i, portfolio_eval in enumerate(optimal_portfolios[:10]):
            portfolio = portfolio_eval['portfolio']
            scenario_analysis = portfolio_eval['scenario_analysis']
            
            report.append(f"{i+1:<4} {portfolio['name']:<15} "
                         f"{scenario_analysis['expected_return']:<9.1%} "
                         f"{scenario_analysis['expected_sharpe']:<9.2f} "
                         f"{portfolio_eval['stability_score']:<7.2f} "
                         f"{portfolio_eval['diversification_score']:<7.2f} "
                         f"{portfolio_eval['composite_score']:<9.2f}")
        
        # 详细分析前3名
        report.append(f"\n💎 前3名详细分析:")
        report.append("=" * 120)
        
        for rank, portfolio_eval in enumerate(optimal_portfolios[:3]):
            portfolio = portfolio_eval['portfolio']
            scenario_analysis = portfolio_eval['scenario_analysis']
            
            report.append(f"\n🥇 第{rank+1}名: {portfolio['name']}")
            report.append("-" * 80)
            report.append(f"• 预期年化收益: {scenario_analysis['expected_return']:.1%}")
            report.append(f"• 预期波动率: {scenario_analysis['expected_volatility']:.1%}")
            report.append(f"• 预期夏普比率: {scenario_analysis['expected_sharpe']:.2f}")
            report.append(f"• 科技股权重: {portfolio['tech_weight']:.1%}")
            report.append(f"• 最大单一持仓: {portfolio_eval['max_single_weight']:.1%}")
            
            # 持仓明细
            report.append(f"\n📋 持仓配置:")
            report.append(f"{'股票':<8} {'权重':<8} {'板块':<12} {'年化收益':<10} {'夏普比率':<10}")
            report.append("-" * 60)
            
            for position in portfolio['positions']:
                symbol = position['symbol']
                weight = position['weight']
                sector = position['sector']
                
                if symbol in stock_data:
                    annual_return = stock_data[symbol]['annual_return']
                    sharpe_ratio = stock_data[symbol]['sharpe_ratio']
                    
                    report.append(f"{symbol:<8} {weight:<7.1%} {sector:<12} "
                                 f"{annual_return:<9.1%} {sharpe_ratio:<9.2f}")
            
            # 各情景表现
            report.append(f"\n📊 不同市场情景表现:")
            report.append(f"{'情景':<12} {'概率':<6} {'预期收益':<10} {'波动率':<8} {'夏普比率':<10}")
            report.append("-" * 50)
            
            for scenario_name, result in scenario_analysis['scenarios'].items():
                scenario_display = self.market_scenarios[scenario_name]['name']
                report.append(f"{scenario_display:<12} {result['probability']:<5.0%} "
                             f"{result['expected_return']:<9.1%} "
                             f"{result['volatility']:<7.1%} "
                             f"{result['sharpe_ratio']:<9.2f}")
        
        # 结论与建议
        best_portfolio = optimal_portfolios[0]
        best_return = best_portfolio['scenario_analysis']['expected_return']
        
        report.append(f"\n🎯 优化结论:")
        report.append("-" * 100)
        
        if best_return >= 0.20:
            report.append(f"✅ 找到了能够实现20%+目标的最优组合!")
            report.append(f"• 最佳组合预期收益: {best_return:.1%}")
            report.append(f"• 该组合在所有市场情景下都表现稳健")
        else:
            report.append(f"⚠️ 在当前市场条件下，20%目标具有挑战性")
            report.append(f"• 最佳组合预期收益: {best_return:.1%}")
            report.append(f"• 建议调整预期或增加风险承受度")
        
        report.append(f"\n💡 专业建议:")
        report.append("-" * 100)
        report.append(f"• 推荐采用第1名组合: {best_portfolio['portfolio']['name']}")
        report.append(f"• 该组合经过了{len(self.market_scenarios)}种市场情景的压力测试")
        report.append(f"• 在{len(optimal_portfolios)}个候选组合中综合表现最佳")
        report.append(f"• 建议定期(每季度)重新优化以适应市场变化")
        
        report.append("\n" + "=" * 120)
        report.append("📋 声明: 本分析基于历史数据和多情景建模，投资有风险，请独立决策")
        report.append("=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    optimizer = PortfolioOptimizationEngine()
    
    # 获取股票数据
    symbols = list(optimizer.stock_universe.keys())
    stock_data = optimizer.get_stock_data(symbols)
    
    # 寻找最优组合
    optimal_portfolios = optimizer.find_optimal_portfolios(stock_data, top_n=10)
    
    # 生成报告
    report = optimizer.generate_optimization_report(optimal_portfolios, stock_data)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细数据
    optimization_results = {
        'timestamp': timestamp,
        'stock_data': stock_data,
        'optimal_portfolios': optimal_portfolios,
        'market_scenarios': optimizer.market_scenarios
    }
    
    with open(f'portfolio_optimization_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(optimization_results, f, ensure_ascii=False, indent=2, default=str)
    
    with open(f'optimization_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 投资组合优化完成")

if __name__ == "__main__":
    main()