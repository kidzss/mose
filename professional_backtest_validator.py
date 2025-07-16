#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业回测验证系统
模拟真实投资场景，验证推荐系统的实际表现
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personal_investor_automation import PersonalInvestorAutomation
from data.data_interface import DataInterface
from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
from monitor.phase2_professional_screener import Phase2ProfessionalScreener

class ProfessionalBacktestValidator:
    """专业回测验证系统"""
    
    def __init__(self):
        self.automation = PersonalInvestorAutomation()
        self.data_interface = DataInterface()
        self.enhanced_analyzer = EnhancedStockAnalyzer()
        self.professional_screener = Phase2ProfessionalScreener()
        
        # 测试股票池
        self.test_stocks = [
            'NVDA', 'AMD', 'GOOGL', 'MSFT', 'AAPL',
            'TSLA', 'NEM', 'ASML', 'PDD', 'CF',
            'BRK-B', 'JPM', 'JNJ', 'PFE', 'XOM'
        ]
        
        # 回测参数
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2025, 7, 13)
        self.freq = 'M'  # 每月回测
        self.holding_periods = [1, 3, 6]  # 持有期：1个月、3个月、6个月
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def get_real_factors(self, symbol: str, date: datetime) -> Dict:
        """获取真实分析因子"""
        try:
            # 获取历史数据
            start = date - timedelta(days=30)
            end = date
            hist = self.data_interface.get_historical_data(symbol, start, end, 'daily')
            if hist.empty:
                return None
            
            close = hist['close'].iloc[-1]
            
            # 增强分析
            enhanced = self.enhanced_analyzer.analyze_stock_enhanced(symbol, close)
            enhanced_score = enhanced.get('overall_score', 0)
            
            # 专业分析
            prof = self.professional_screener.analyze_stock_professional(symbol)
            quality = prof.get('quality_factor', 0.5) if prof else 0.5
            multifactor_score = prof.get('multifactor_score', 50) if prof else 50
            
            return {
                'quality_factor': quality,
                'enhanced_score': enhanced_score,
                'multifactor_score': multifactor_score,
                'current_price': close,
                'enhanced_analysis': enhanced
            }
        except Exception as e:
            print(f"  {symbol} 因子分析失败: {e}")
            return None

    def get_market_condition(self, date: datetime) -> str:
        """获取市场环境"""
        try:
            spy = self.data_interface.get_historical_data('SPY', date - timedelta(days=60), date, 'daily')
            if spy.empty:
                return '未知'
            
            spy['sma_20'] = spy['close'].rolling(20).mean()
            spy['sma_50'] = spy['close'].rolling(50).mean()
            latest = spy.iloc[-1]
            
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                return '牛市'
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                return '熊市'
            else:
                return '震荡市'
        except:
            return '未知'

    def get_adaptive_thresholds(self, market_condition: str) -> Dict:
        """获取自适应阈值"""
        if market_condition == '牛市':
            return {
                'strong_buy': {'enhanced_score': 0.65, 'quality': 0.6, 'score': 60},
                'buy': {'enhanced_score': 0.6, 'quality': 0.55, 'score': 50},
                'hold': {'enhanced_score': 0.5, 'quality': 0.5, 'score': 45},
                'watch': {'enhanced_score': 0.3}
            }
        elif market_condition == '熊市':
            return {
                'strong_buy': {'enhanced_score': 0.75, 'quality': 0.75, 'score': 70},
                'buy': {'enhanced_score': 0.7, 'quality': 0.7, 'score': 65},
                'hold': {'enhanced_score': 0.6, 'quality': 0.6, 'score': 55},
                'watch': {'enhanced_score': 0.3}
            }
        else:
            return {
                'strong_buy': {'enhanced_score': 0.7, 'quality': 0.65, 'score': 65},
                'buy': {'enhanced_score': 0.65, 'quality': 0.6, 'score': 55},
                'hold': {'enhanced_score': 0.55, 'quality': 0.55, 'score': 50},
                'watch': {'enhanced_score': 0.3}
            }

    def calculate_returns(self, symbol: str, entry_date: datetime, holding_months: int) -> Dict:
        """计算持有期收益率"""
        try:
            # 获取入场价格
            entry_start = entry_date - timedelta(days=5)
            entry_end = entry_date + timedelta(days=5)
            entry_data = self.data_interface.get_historical_data(symbol, entry_start, entry_end, 'daily')
            if entry_data.empty:
                return None
            
            entry_price = entry_data['close'].iloc[-1]
            
            # 获取退出价格
            exit_date = entry_date + timedelta(days=30 * holding_months)
            exit_start = exit_date - timedelta(days=5)
            exit_end = exit_date + timedelta(days=5)
            exit_data = self.data_interface.get_historical_data(symbol, exit_start, exit_end, 'daily')
            if exit_data.empty:
                return None
            
            exit_price = exit_data['close'].iloc[-1]
            
            # 计算收益率
            total_return = (exit_price - entry_price) / entry_price
            annualized_return = ((1 + total_return) ** (12 / holding_months)) - 1
            
            # 计算最大回撤
            period_data = self.data_interface.get_historical_data(symbol, entry_date, exit_date, 'daily')
            if not period_data.empty:
                period_data['cummax'] = period_data['close'].cummax()
                period_data['drawdown'] = (period_data['close'] - period_data['cummax']) / period_data['cummax']
                max_drawdown = period_data['drawdown'].min()
            else:
                max_drawdown = 0
            
            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'holding_months': holding_months
            }
        except Exception as e:
            print(f"  {symbol} 收益率计算失败: {e}")
            return None

    def run_professional_backtest(self) -> Dict:
        """运行专业回测验证"""
        print("🚀 专业回测验证系统启动")
        print("=" * 80)
        
        dates = pd.date_range(self.start_date, self.end_date, freq=self.freq)
        all_recommendations = []
        all_returns = []
        
        for date in dates:
            print(f"\n📅 回测日期: {date.strftime('%Y-%m-%d')}")
            market_condition = self.get_market_condition(date)
            thresholds = self.get_adaptive_thresholds(market_condition)
            print(f"  市场环境: {market_condition}")
            
            month_recommendations = []
            
            for symbol in self.test_stocks:
                # 获取分析因子
                factors = self.get_real_factors(symbol, date)
                if not factors:
                    continue
                
                # 获取策略信号
                try:
                    strategy_analysis = self.automation._analyze_strategy_signals(symbol)
                except:
                    strategy_analysis = {}
                
                # 组合数据
                stock_data = {
                    'symbol': symbol,
                    'quality_factor': factors['quality_factor'],
                    'multifactor_score': factors['multifactor_score'],
                    'enhanced_score': factors['enhanced_score'],
                    'current_price': factors['current_price'],
                    'strategy_analysis': strategy_analysis
                }
                
                # 获取投资建议
                advice = self.automation._get_enhanced_investment_advice(stock_data, thresholds=thresholds)
                
                recommendation = {
                    'date': date,
                    'symbol': symbol,
                    'advice': advice,
                    'quality_factor': factors['quality_factor'],
                    'enhanced_score': factors['enhanced_score'],
                    'multifactor_score': factors['multifactor_score'],
                    'current_price': factors['current_price'],
                    'market_condition': market_condition
                }
                
                month_recommendations.append(recommendation)
                print(f"  {symbol}: {advice} (价格: ${factors['current_price']:.2f})")
                
                # 计算后续收益率
                for holding_period in self.holding_periods:
                    returns = self.calculate_returns(symbol, date, holding_period)
                    if returns:
                        return_data = {
                            'date': date,
                            'symbol': symbol,
                            'advice': advice,
                            'holding_period': holding_period,
                            **returns
                        }
                        all_returns.append(return_data)
            
            all_recommendations.extend(month_recommendations)
        
        return {
            'recommendations': all_recommendations,
            'returns': all_returns
        }

    def analyze_performance(self, results: Dict) -> Dict:
        """分析回测表现"""
        print("\n📊 回测表现分析")
        print("=" * 80)
        
        recommendations_df = pd.DataFrame(results['recommendations'])
        returns_df = pd.DataFrame(results['returns'])
        
        # 按建议类型分组分析
        performance_analysis = {}
        
        for advice_type in ['🟢 强烈推荐', '🔵 推荐买入', '🟡 小仓位试仓', '🟠 观望为主']:
            advice_returns = returns_df[returns_df['advice'] == advice_type]
            
            if not advice_returns.empty:
                analysis = {
                    'count': len(advice_returns),
                    'avg_return_1m': advice_returns[advice_returns['holding_period'] == 1]['total_return'].mean(),
                    'avg_return_3m': advice_returns[advice_returns['holding_period'] == 3]['total_return'].mean(),
                    'avg_return_6m': advice_returns[advice_returns['holding_period'] == 6]['total_return'].mean(),
                    'win_rate_1m': (advice_returns[advice_returns['holding_period'] == 1]['total_return'] > 0).mean(),
                    'win_rate_3m': (advice_returns[advice_returns['holding_period'] == 3]['total_return'] > 0).mean(),
                    'win_rate_6m': (advice_returns[advice_returns['holding_period'] == 6]['total_return'] > 0).mean(),
                    'max_drawdown_avg': advice_returns['max_drawdown'].mean()
                }
                performance_analysis[advice_type] = analysis
                
                print(f"\n{advice_type}:")
                print(f"  推荐次数: {analysis['count']}")
                print(f"  1个月平均收益: {analysis['avg_return_1m']:.2%}")
                print(f"  3个月平均收益: {analysis['avg_return_3m']:.2%}")
                print(f"  6个月平均收益: {analysis['avg_return_6m']:.2%}")
                print(f"  1个月胜率: {analysis['win_rate_1m']:.2%}")
                print(f"  3个月胜率: {analysis['win_rate_3m']:.2%}")
                print(f"  6个月胜率: {analysis['win_rate_6m']:.2%}")
                print(f"  平均最大回撤: {analysis['max_drawdown_avg']:.2%}")
        
        return performance_analysis

    def generate_performance_report(self, results: Dict, performance: Dict):
        """生成专业表现报告"""
        print("\n📈 专业表现报告")
        print("=" * 80)
        
        recommendations_df = pd.DataFrame(results['recommendations'])
        returns_df = pd.DataFrame(results['returns'])
        
        # 总体统计
        total_recommendations = len(recommendations_df)
        total_returns = len(returns_df)
        
        print(f"总推荐次数: {total_recommendations}")
        print(f"总收益率计算次数: {total_returns}")
        
        # 按市场环境分析
        print("\n📊 按市场环境分析:")
        for market_condition in ['牛市', '熊市', '震荡市']:
            market_returns = returns_df[returns_df['market_condition'] == market_condition]
            if not market_returns.empty:
                avg_return = market_returns['total_return'].mean()
                win_rate = (market_returns['total_return'] > 0).mean()
                print(f"  {market_condition}: 平均收益 {avg_return:.2%}, 胜率 {win_rate:.2%}")
        
        # 最佳表现股票
        print("\n🏆 最佳表现股票 (6个月持有期):")
        six_month_returns = returns_df[returns_df['holding_period'] == 6]
        if not six_month_returns.empty:
            best_performers = six_month_returns.nlargest(5, 'total_return')
            for _, row in best_performers.iterrows():
                print(f"  {row['symbol']}: {row['total_return']:.2%} ({row['advice']})")
        
        # 最差表现股票
        print("\n📉 最差表现股票 (6个月持有期):")
        worst_performers = six_month_returns.nsmallest(5, 'total_return')
        for _, row in worst_performers.iterrows():
            print(f"  {row['symbol']}: {row['total_return']:.2%} ({row['advice']})")

def main():
    """主函数"""
    validator = ProfessionalBacktestValidator()
    
    # 运行专业回测
    results = validator.run_professional_backtest()
    
    # 分析表现
    performance = validator.analyze_performance(results)
    
    # 生成报告
    validator.generate_performance_report(results, performance)
    
    print("\n🎉 专业回测验证完成！")
    print("=" * 80)
    print("💡 这个系统模拟了真实投资场景:")
    print("  1. 在推荐日期买入股票")
    print("  2. 持有1个月、3个月、6个月后卖出")
    print("  3. 计算实际收益率和风险指标")
    print("  4. 验证推荐系统的有效性")

if __name__ == "__main__":
    main() 