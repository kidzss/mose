#!/usr/bin/env python3
"""
市场时机策略分析器
分析在不同市场情景下的最优减仓和买入时机
考虑SPY从当前水平到5800(下跌)或6250(上涨)的情景
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

class MarketTimingAnalyzer:
    """市场时机策略分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 从配置文件读取当前持仓
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 当前总资产
        self.total_assets = self.config['portfolio']['total_value']
        
        # AMD减仓释放的资金 (从18.83%降至10%)
        self.amd_reduction_cash = 2409  # 从之前分析得出
        
        # 新增投资目标股票及其当前价格
        self.target_stocks = {
            'META': {'sector': 'Technology', 'target_weight': 0.12},
            'ABT': {'sector': 'Healthcare', 'target_weight': 0.06},
            'JNJ': {'sector': 'Healthcare', 'target_weight': 0.06},
            'WFC': {'sector': 'Financial', 'target_weight': 0.05},
            'BAC': {'sector': 'Financial', 'target_weight': 0.05},
            'WMT': {'sector': 'Consumer', 'target_weight': 0.045},
            'KO': {'sector': 'Consumer', 'target_weight': 0.045},
            'COST': {'sector': 'Consumer', 'target_weight': 0.045},
            'CAT': {'sector': 'Industrial', 'target_weight': 0.025},
            'BA': {'sector': 'Industrial', 'target_weight': 0.02},
            'XOM': {'sector': 'Energy', 'target_weight': 0.015},
            'CVX': {'sector': 'Energy', 'target_weight': 0.015}
        }
        
        # 市场情景设置
        self.market_scenarios = {
            'current': {'spy_level': None, 'probability': 0.3, 'description': '维持当前水平'},
            'bearish': {'spy_level': 5800, 'probability': 0.35, 'description': '下跌至5800'},
            'bullish': {'spy_level': 6250, 'probability': 0.35, 'description': '上涨至6250'}
        }
        
        # 策略选项
        self.strategies = {
            'immediate_buy': {'description': '立即减仓并买入', 'cash_holding_period': 0},
            'wait_for_dip': {'description': '减仓持有现金等回调', 'cash_holding_period': 90},
            'gradual_entry': {'description': '分批建仓策略', 'cash_holding_period': 45},
            'market_timing': {'description': '技术指标择时', 'cash_holding_period': 60}
        }
        
        logger.info("📊 市场时机策略分析器初始化完成")
    
    def get_current_market_data(self):
        """获取当前市场数据和技术指标"""
        try:
            # 获取SPY当前数据
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            current_spy = spy_hist['Close'].iloc[-1]
            
            # 计算技术指标
            ma_20 = spy_hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = spy_hist['Close'].rolling(50).mean().iloc[-1]
            ma_200 = spy_hist['Close'].rolling(200).mean().iloc[-1]
            
            # 计算VIX
            vix = yf.Ticker("^VIX")
            vix_current = vix.history(period="5d")['Close'].iloc[-1]
            
            # 更新市场情景的当前水平
            self.market_scenarios['current']['spy_level'] = current_spy
            
            # 计算预期变化幅度
            bear_change = (5800 - current_spy) / current_spy
            bull_change = (6250 - current_spy) / current_spy
            
            market_data = {
                'current_spy': current_spy,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'vix': vix_current,
                'bear_scenario_change': bear_change,
                'bull_scenario_change': bull_change,
                'trend_signal': 'bullish' if current_spy > ma_20 > ma_50 else 'bearish' if current_spy < ma_20 < ma_50 else 'neutral'
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None
    
    def get_target_stocks_data(self):
        """获取目标股票的当前数据和技术指标"""
        stocks_data = {}
        
        for symbol in self.target_stocks.keys():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    # 技术指标
                    ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                    
                    # 计算RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]
                    
                    # 计算相对SPY的Beta
                    spy_hist = yf.Ticker("SPY").history(period="1y")['Close']
                    stock_returns = hist['Close'].pct_change().dropna()
                    spy_returns = spy_hist.pct_change().dropna()
                    
                    # 对齐时间序列
                    common_dates = stock_returns.index.intersection(spy_returns.index)
                    if len(common_dates) > 50:
                        aligned_stock = stock_returns.loc[common_dates]
                        aligned_spy = spy_returns.loc[common_dates]
                        beta = np.cov(aligned_stock, aligned_spy)[0, 1] / np.var(aligned_spy)
                    else:
                        beta = 1.0
                    
                    stocks_data[symbol] = {
                        'current_price': current_price,
                        'ma_20': ma_20,
                        'ma_50': ma_50,
                        'rsi': rsi,
                        'beta': beta,
                        'sector': self.target_stocks[symbol]['sector'],
                        'target_weight': self.target_stocks[symbol]['target_weight'],
                        'distance_from_ma20': (current_price - ma_20) / ma_20,
                        'technical_signal': 'bullish' if current_price > ma_20 > ma_50 else 'bearish' if current_price < ma_20 < ma_50 else 'neutral'
                    }
                    
            except Exception as e:
                logger.warning(f"获取{symbol}数据失败: {e}")
        
        return stocks_data
    
    def calculate_scenario_outcomes(self, market_data, stocks_data):
        """计算不同市场情景下的股票预期价格"""
        scenario_outcomes = {}
        
        for scenario_name, scenario_info in self.market_scenarios.items():
            scenario_outcomes[scenario_name] = {
                'info': scenario_info,
                'stock_prices': {}
            }
            
            if scenario_name == 'current':
                # 当前价格
                for symbol, data in stocks_data.items():
                    scenario_outcomes[scenario_name]['stock_prices'][symbol] = data['current_price']
            else:
                # 根据Beta和市场变化计算预期价格
                spy_change = (scenario_info['spy_level'] - market_data['current_spy']) / market_data['current_spy']
                
                for symbol, data in stocks_data.items():
                    # 考虑Beta和个股特性的价格变化
                    beta = data['beta']
                    
                    # 行业调整因子
                    sector_adjustments = {
                        'Technology': 1.2,      # 科技股对市场变化更敏感
                        'Healthcare': 0.8,      # 医疗股相对防御
                        'Financial': 1.1,       # 金融股对经济敏感
                        'Consumer': 0.9,        # 消费股相对稳定
                        'Industrial': 1.0,      # 工业股跟随市场
                        'Energy': 1.3          # 能源股波动较大
                    }
                    
                    sector_factor = sector_adjustments.get(data['sector'], 1.0)
                    expected_change = spy_change * beta * sector_factor
                    
                    # 加入一些随机性(正态分布)
                    volatility_adjustment = np.random.normal(0, 0.02)  # 2%的随机波动
                    final_change = expected_change + volatility_adjustment
                    
                    expected_price = data['current_price'] * (1 + final_change)
                    scenario_outcomes[scenario_name]['stock_prices'][symbol] = expected_price
        
        return scenario_outcomes
    
    def analyze_strategy_performance(self, market_data, stocks_data, scenario_outcomes):
        """分析不同策略在各种市场情景下的表现"""
        strategy_analysis = {}
        
        # 计算总投资金额
        total_investment_needed = sum([
            self.total_assets * stock_info['target_weight'] 
            for stock_info in self.target_stocks.values()
        ])
        
        for strategy_name, strategy_info in self.strategies.items():
            strategy_analysis[strategy_name] = {
                'description': strategy_info['description'],
                'scenario_outcomes': {},
                'weighted_return': 0,
                'risk_score': 0
            }
            
            for scenario_name, scenario_data in scenario_outcomes.items():
                outcome = self.calculate_strategy_scenario_outcome(
                    strategy_name, scenario_name, scenario_data, stocks_data, 
                    market_data, total_investment_needed
                )
                
                strategy_analysis[strategy_name]['scenario_outcomes'][scenario_name] = outcome
                
                # 计算加权收益
                probability = scenario_data['info']['probability']
                strategy_analysis[strategy_name]['weighted_return'] += outcome['total_return'] * probability
        
        return strategy_analysis
    
    def calculate_strategy_scenario_outcome(self, strategy_name, scenario_name, scenario_data, 
                                          stocks_data, market_data, total_investment):
        """计算特定策略在特定情景下的结果"""
        
        # 基础参数
        cash_available = self.amd_reduction_cash
        cash_holding_period = self.strategies[strategy_name]['cash_holding_period']
        
        # 现金机会成本 (假设4%年化收益的货币基金)
        cash_opportunity_cost = 0 if cash_holding_period == 0 else (cash_available * 0.04 * cash_holding_period / 365)
        
        total_portfolio_value = 0
        total_cost = 0
        detailed_positions = {}
        
        for symbol, target_data in self.target_stocks.items():
            target_value = self.total_assets * target_data['target_weight']
            current_price = stocks_data[symbol]['current_price']
            scenario_price = scenario_data['stock_prices'][symbol]
            
            # 根据策略确定买入价格
            if strategy_name == 'immediate_buy':
                buy_price = current_price
                shares = int(target_value / buy_price)
                cost = shares * buy_price
                position_value = shares * scenario_price
                
            elif strategy_name == 'wait_for_dip':
                # 等待回调策略：如果市场下跌，在低点买入；如果上涨，错失机会
                if scenario_name == 'bearish':
                    # 在低点买入，获得更好价格
                    discount_factor = 0.95  # 假设能在低点前5%买入
                    buy_price = scenario_price * discount_factor
                    shares = int(target_value / buy_price)
                    cost = shares * buy_price
                    position_value = shares * scenario_price
                elif scenario_name == 'bullish':
                    # 错失上涨，在高点买入
                    premium_factor = 1.02  # 追高买入溢价2%
                    buy_price = scenario_price * premium_factor
                    shares = int(target_value / buy_price)
                    cost = shares * buy_price
                    position_value = shares * scenario_price
                else:  # current
                    # 当前价格买入
                    buy_price = current_price
                    shares = int(target_value / buy_price)
                    cost = shares * buy_price
                    position_value = shares * scenario_price
                    
            elif strategy_name == 'gradual_entry':
                # 分批建仓：部分立即买入，部分等待
                immediate_ratio = 0.6  # 60%立即买入
                delayed_ratio = 0.4    # 40%等待买入
                
                # 立即买入部分
                immediate_value = target_value * immediate_ratio
                immediate_shares = int(immediate_value / current_price)
                immediate_cost = immediate_shares * current_price
                
                # 延迟买入部分
                delayed_value = target_value * delayed_ratio
                if scenario_name == 'bearish':
                    delayed_buy_price = scenario_price * 0.97  # 在下跌中分批买入
                elif scenario_name == 'bullish':
                    delayed_buy_price = scenario_price * 1.01  # 在上涨中追买
                else:
                    delayed_buy_price = current_price * 0.995  # 轻微折扣
                
                delayed_shares = int(delayed_value / delayed_buy_price)
                delayed_cost = delayed_shares * delayed_buy_price
                
                shares = immediate_shares + delayed_shares
                cost = immediate_cost + delayed_cost
                position_value = shares * scenario_price
                
            else:  # market_timing
                # 技术择时策略
                rsi = stocks_data[symbol]['rsi']
                technical_signal = stocks_data[symbol]['technical_signal']
                
                if rsi < 40 and technical_signal != 'bullish':  # 超卖时买入
                    buy_price = current_price * 0.98
                elif rsi > 60 and technical_signal == 'bullish':  # 超买时谨慎
                    buy_price = current_price * 1.01
                else:
                    buy_price = current_price
                
                shares = int(target_value / buy_price)
                cost = shares * buy_price
                position_value = shares * scenario_price
            
            detailed_positions[symbol] = {
                'shares': shares,
                'cost': cost,
                'position_value': position_value,
                'return': (position_value - cost) / cost if cost > 0 else 0
            }
            
            total_portfolio_value += position_value
            total_cost += cost
        
        # 计算总体收益
        investment_return = (total_portfolio_value - total_cost) / total_cost if total_cost > 0 else 0
        
        # 扣除现金机会成本
        net_return = investment_return - (cash_opportunity_cost / total_cost if total_cost > 0 else 0)
        
        return {
            'total_cost': total_cost,
            'total_value': total_portfolio_value,
            'investment_return': investment_return,
            'cash_opportunity_cost': cash_opportunity_cost,
            'net_return': net_return,
            'total_return': net_return,  # 简化，实际应考虑更多因素
            'detailed_positions': detailed_positions
        }
    
    def monte_carlo_simulation(self, market_data, stocks_data, iterations=1000):
        """蒙特卡洛模拟分析策略稳健性"""
        simulation_results = {}
        
        for strategy_name in self.strategies.keys():
            strategy_returns = []
            
            for _ in range(iterations):
                # 随机选择市场情景
                scenario_choice = np.random.choice(
                    list(self.market_scenarios.keys()),
                    p=[scenario['probability'] for scenario in self.market_scenarios.values()]
                )
                
                # 生成随机的股票价格变化
                random_scenario_outcomes = {}
                for scenario_name, scenario_info in self.market_scenarios.items():
                    random_scenario_outcomes[scenario_name] = {
                        'info': scenario_info,
                        'stock_prices': {}
                    }
                    
                    if scenario_name == 'current':
                        spy_change = np.random.normal(0, 0.02)  # 当前水平附近小幅波动
                    elif scenario_name == 'bearish':
                        spy_change = (5800 - market_data['current_spy']) / market_data['current_spy']
                        spy_change += np.random.normal(0, 0.03)  # 加入不确定性
                    else:  # bullish
                        spy_change = (6250 - market_data['current_spy']) / market_data['current_spy']
                        spy_change += np.random.normal(0, 0.03)
                    
                    for symbol, data in stocks_data.items():
                        beta = data['beta']
                        sector_factor = {'Technology': 1.2, 'Healthcare': 0.8, 'Financial': 1.1, 
                                       'Consumer': 0.9, 'Industrial': 1.0, 'Energy': 1.3}.get(data['sector'], 1.0)
                        
                        expected_change = spy_change * beta * sector_factor
                        volatility = np.random.normal(0, 0.04)  # 个股特有风险
                        final_change = expected_change + volatility
                        
                        random_scenario_outcomes[scenario_name]['stock_prices'][symbol] = \
                            data['current_price'] * (1 + final_change)
                
                # 计算该次模拟的策略收益
                total_investment = sum([self.total_assets * info['target_weight'] for info in self.target_stocks.values()])
                outcome = self.calculate_strategy_scenario_outcome(
                    strategy_name, scenario_choice, random_scenario_outcomes[scenario_choice], 
                    stocks_data, market_data, total_investment
                )
                
                strategy_returns.append(outcome['total_return'])
            
            # 统计结果
            strategy_returns = np.array(strategy_returns)
            simulation_results[strategy_name] = {
                'mean_return': np.mean(strategy_returns),
                'std_return': np.std(strategy_returns),
                'percentile_5': np.percentile(strategy_returns, 5),
                'percentile_25': np.percentile(strategy_returns, 25),
                'percentile_75': np.percentile(strategy_returns, 75),
                'percentile_95': np.percentile(strategy_returns, 95),
                'probability_positive': np.mean(strategy_returns > 0),
                'max_return': np.max(strategy_returns),
                'min_return': np.min(strategy_returns)
            }
        
        return simulation_results
    
    def generate_timing_strategy_report(self, market_data, stocks_data, scenario_outcomes, 
                                      strategy_analysis, simulation_results):
        """生成市场时机策略报告"""
        report = []
        report.append("=" * 120)
        report.append("📊 市场时机策略分析报告")
        report.append("🎯 AMD减仓后的最优买入时机分析")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 当前市场状况
        if market_data:
            report.append(f"\n📈 当前市场状况:")
            report.append("-" * 100)
            report.append(f"• SPY当前价格: ${market_data['current_spy']:.2f}")
            report.append(f"• 技术趋势: {market_data['trend_signal'].upper()}")
            report.append(f"• VIX恐慌指数: {market_data['vix']:.1f}")
            report.append(f"• 相对20日均线: {((market_data['current_spy'] - market_data['ma_20']) / market_data['ma_20']) * 100:+.1f}%")
            report.append(f"• 熊市情景变化: {market_data['bear_scenario_change']:+.1%} (跌至5800)")
            report.append(f"• 牛市情景变化: {market_data['bull_scenario_change']:+.1%} (涨至6250)")
        
        # 待买入股票当前状况
        report.append(f"\n🎯 待买入股票当前状况:")
        report.append("-" * 100)
        
        if stocks_data:
            report.append(f"{'股票':<6} {'当前价格':<10} {'目标权重':<8} {'RSI':<6} {'技术信号':<8} {'相对MA20':<10}")
            report.append("-" * 70)
            
            for symbol, data in sorted(stocks_data.items(), key=lambda x: x[1]['target_weight'], reverse=True):
                report.append(f"{symbol:<6} ${data['current_price']:<9.2f} {data['target_weight']:<7.1%} "
                             f"{data['rsi']:<5.0f} {data['technical_signal']:<8} {data['distance_from_ma20']:>+8.1%}")
        
        # 市场情景分析
        report.append(f"\n📊 市场情景概率分析:")
        report.append("-" * 100)
        
        for scenario_name, scenario_info in self.market_scenarios.items():
            report.append(f"• {scenario_info['description']}: {scenario_info['probability']:.1%}概率")
            if scenario_name != 'current':
                spy_level = scenario_info['spy_level']
                change = (spy_level - market_data['current_spy']) / market_data['current_spy']
                report.append(f"  SPY目标: ${spy_level} ({change:+.1%})")
        
        # 策略表现对比
        report.append(f"\n🎯 策略表现对比分析:")
        report.append("-" * 100)
        
        if strategy_analysis:
            # 表头
            report.append(f"{'策略':<15} {'加权收益':<10} {'熊市5800':<10} {'当前水平':<10} {'牛市6250':<10}")
            report.append("-" * 70)
            
            for strategy_name, analysis in strategy_analysis.items():
                weighted_return = analysis['weighted_return']
                bear_return = analysis['scenario_outcomes'].get('bearish', {}).get('total_return', 0)
                current_return = analysis['scenario_outcomes'].get('current', {}).get('total_return', 0)
                bull_return = analysis['scenario_outcomes'].get('bullish', {}).get('total_return', 0)
                
                report.append(f"{self.strategies[strategy_name]['description']:<15} "
                             f"{weighted_return:>8.1%} {bear_return:>9.1%} {current_return:>9.1%} {bull_return:>9.1%}")
        
        # 蒙特卡洛模拟结果
        if simulation_results:
            report.append(f"\n🎲 蒙特卡洛模拟结果 (1000次):")
            report.append("-" * 100)
            
            # 表头
            report.append(f"{'策略':<15} {'平均收益':<10} {'标准差':<8} {'胜率':<8} {'5%分位':<8} {'95%分位':<8}")
            report.append("-" * 70)
            
            for strategy_name, results in simulation_results.items():
                report.append(f"{self.strategies[strategy_name]['description']:<15} "
                             f"{results['mean_return']:>8.1%} {results['std_return']:>7.1%} "
                             f"{results['probability_positive']:>7.1%} {results['percentile_5']:>7.1%} "
                             f"{results['percentile_95']:>7.1%}")
        
        # 策略详细分析
        report.append(f"\n📋 各策略详细分析:")
        report.append("-" * 100)
        
        strategy_rankings = sorted(strategy_analysis.items(), 
                                 key=lambda x: x[1]['weighted_return'], reverse=True)
        
        for rank, (strategy_name, analysis) in enumerate(strategy_rankings, 1):
            strategy_info = self.strategies[strategy_name]
            report.append(f"\n【排名第{rank}: {strategy_info['description']}】")
            
            # 优势分析
            advantages = []
            disadvantages = []
            
            if strategy_name == 'immediate_buy':
                advantages.extend([
                    "✅ 立即参与市场，不错失任何上涨机会",
                    "✅ 操作简单，无需择时判断",
                    "✅ 避免现金拖累收益"
                ])
                disadvantages.extend([
                    "❌ 如果市场下跌，会在高点买入",
                    "❌ 无法利用市场波动获得更好价格"
                ])
                
            elif strategy_name == 'wait_for_dip':
                advantages.extend([
                    "✅ 如果市场下跌，可在低点获得更好价格",
                    "✅ 风险控制较好",
                    "✅ 符合逆向投资思维"
                ])
                disadvantages.extend([
                    "❌ 如果市场上涨，会错失收益机会",
                    "❌ 现金拖累期间收益",
                    "❌ 需要准确判断市场底部"
                ])
                
            elif strategy_name == 'gradual_entry':
                advantages.extend([
                    "✅ 平衡了时机风险",
                    "✅ 可以捕捉部分上涨，也能在下跌中分批买入",
                    "✅ 心理压力较小"
                ])
                disadvantages.extend([
                    "❌ 可能两头不讨好",
                    "❌ 执行复杂度较高"
                ])
                
            else:  # market_timing
                advantages.extend([
                    "✅ 基于技术指标，相对客观",
                    "✅ 可以识别超买超卖机会",
                    "✅ 适合有技术分析基础的投资者"
                ])
                disadvantages.extend([
                    "❌ 技术分析可能失效",
                    "❌ 需要持续监控市场",
                    "❌ 交易频率可能较高"
                ])
            
            for advantage in advantages:
                report.append(f"  {advantage}")
            for disadvantage in disadvantages:
                report.append(f"  {disadvantage}")
            
            # 适合条件
            report.append(f"  💡 适合条件:")
            if strategy_name == 'immediate_buy':
                report.append(f"     - 看好后市，认为当前是相对低点")
                report.append(f"     - 不想承担择时风险")
                report.append(f"     - 有长期投资视角")
                
            elif strategy_name == 'wait_for_dip':
                report.append(f"     - 认为市场短期可能回调")
                report.append(f"     - 有耐心等待更好买点")
                report.append(f"     - 现金管理能力强")
                
            elif strategy_name == 'gradual_entry':
                report.append(f"     - 对市场方向不确定")
                report.append(f"     - 希望平衡风险收益")
                report.append(f"     - 有足够时间执行分批策略")
                
            else:
                report.append(f"     - 有技术分析经验")
                report.append(f"     - 能够客观执行信号")
                report.append(f"     - 市场波动较大时期")
        
        # 最优策略推荐
        best_strategy = strategy_rankings[0]
        report.append(f"\n🏆 最优策略推荐:")
        report.append("-" * 100)
        
        report.append(f"• 综合分析最优策略: {self.strategies[best_strategy[0]]['description']}")
        report.append(f"• 预期加权收益: {best_strategy[1]['weighted_return']:+.1%}")
        
        if simulation_results and best_strategy[0] in simulation_results:
            sim_result = simulation_results[best_strategy[0]]
            report.append(f"• 模拟胜率: {sim_result['probability_positive']:.1%}")
            report.append(f"• 收益区间: [{sim_result['percentile_5']:+.1%}, {sim_result['percentile_95']:+.1%}] (90%置信度)")
        
        # 执行建议
        report.append(f"\n💡 具体执行建议:")
        report.append("-" * 100)
        
        if best_strategy[0] == 'immediate_buy':
            report.append(f"• 🚀 建议立即开始减持AMD并买入目标股票")
            report.append(f"• 📊 优先买入技术面较强的股票: META, WFC, COST")
            report.append(f"• ⏰ 在1-2周内完成建仓")
            
        elif best_strategy[0] == 'wait_for_dip':
            report.append(f"• ⏳ 建议先减持AMD，持有现金等待回调")
            report.append(f"• 🎯 关键支撑位: SPY 5850-5900区间")
            report.append(f"• ⚠️ 设置止损位: 如果SPY突破6100，开始分批买入")
            
        elif best_strategy[0] == 'gradual_entry':
            report.append(f"• 📈 建议60%资金立即买入，40%分批买入")
            report.append(f"• 🎯 立即买入防御性股票: ABT, JNJ, KO")
            report.append(f"• ⏰ 剩余资金在1-2个月内分批建仓")
            
        else:  # market_timing
            report.append(f"• 📊 基于技术指标择时买入")
            report.append(f"• 🔍 关注RSI<40的超卖机会")
            report.append(f"• 📈 优先买入技术信号转好的股票")
        
        # 风险提示
        report.append(f"\n⚠️ 风险提示:")
        report.append("-" * 100)
        report.append(f"• 市场预测存在不确定性，任何策略都有失败可能")
        report.append(f"• 建议根据个人风险承受能力调整策略")
        report.append(f"• 保持一定现金储备，应对意外情况")
        report.append(f"• 定期回顾和调整策略执行")
        
        # 监控指标
        report.append(f"\n📊 关键监控指标:")
        report.append("-" * 100)
        report.append(f"• SPY价格水平: 关注5800-6250区间")
        report.append(f"• VIX恐慌指数: >25时考虑加大买入力度")
        report.append(f"• 美债收益率: 10年期国债收益率变化")
        report.append(f"• 个股技术指标: RSI、均线系统、成交量")
        
        report.append("\n" + "=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    analyzer = MarketTimingAnalyzer()
    
    # 获取当前市场数据
    market_data = analyzer.get_current_market_data()
    
    # 获取目标股票数据
    stocks_data = analyzer.get_target_stocks_data()
    
    if market_data and stocks_data:
        # 计算各种市场情景下的预期结果
        scenario_outcomes = analyzer.calculate_scenario_outcomes(market_data, stocks_data)
        
        # 分析不同策略的表现
        strategy_analysis = analyzer.analyze_strategy_performance(market_data, stocks_data, scenario_outcomes)
        
        # 蒙特卡洛模拟
        simulation_results = analyzer.monte_carlo_simulation(market_data, stocks_data, 1000)
        
        # 生成报告
        report = analyzer.generate_timing_strategy_report(
            market_data, stocks_data, scenario_outcomes, 
            strategy_analysis, simulation_results)
        
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细数据
        output_data = {
            'timestamp': timestamp,
            'market_data': market_data,
            'stocks_data': stocks_data,
            'scenario_outcomes': scenario_outcomes,
            'strategy_analysis': strategy_analysis,
            'simulation_results': simulation_results
        }
        
        with open(f'market_timing_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        
        with open(f'market_timing_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("✅ 市场时机策略分析完成，报告已保存")
        
    else:
        logger.error("❌ 无法获取必要的市场数据")

if __name__ == "__main__":
    main() 