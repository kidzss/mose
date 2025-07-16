#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统回归测试
验证不同质量因子阈值下的推荐效果
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personal_investor_automation import PersonalInvestorAutomation
from data.data_interface import DataInterface
from monitor.enhanced_stock_analyzer import EnhancedStockAnalyzer
from monitor.phase2_professional_screener import Phase2ProfessionalScreener

class RecommendationSystemBacktest:
    """推荐系统真实数据回测"""
    def __init__(self):
        self.automation = PersonalInvestorAutomation()
        self.data_interface = DataInterface()
        self.enhanced_analyzer = EnhancedStockAnalyzer()
        self.professional_screener = Phase2ProfessionalScreener()
        self.test_stocks = [
            'NVDA', 'AMD', 'GOOGL', 'MSFT', 'AAPL',
            'TSLA', 'NEM', 'ASML', 'PDD', 'CF',
            'BRK-B', 'JPM', 'JNJ', 'PFE', 'XOM'
        ]
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2025, 7, 13)
        self.freq = 'M'  # 每月回测

    def get_real_factors(self, symbol, date):
        """获取真实quality_factor/enhanced_score等"""
        # 获取该日期前30天的历史数据
        start = date - timedelta(days=30)
        end = date
        hist = self.data_interface.get_historical_data(symbol, start, end, 'daily')
        if hist.empty:
            return None
        close = hist['close'].iloc[-1]
        # 用增强分析器获取 enhanced_score
        try:
            enhanced = self.enhanced_analyzer.analyze_stock_enhanced(symbol, close)
            enhanced_score = enhanced.get('overall_score', 0)
        except Exception as e:
            print(f"  {symbol} 增强分析失败: {e}")
            enhanced_score = 0
        # 用专业分析器获取 quality_factor 和 multifactor_score
        try:
            prof = self.professional_screener.analyze_stock_professional(symbol)
            quality = prof.get('quality_factor', 0.5) if prof else 0.5
            multifactor_score = prof.get('multifactor_score', 50) if prof else 50
        except Exception as e:
            print(f"  {symbol} 专业分析失败: {e}")
            quality = 0.5
            multifactor_score = 50
        return {
            'quality_factor': quality,
            'enhanced_score': enhanced_score,
            'multifactor_score': multifactor_score,
            'enhanced_analysis': enhanced if 'enhanced' in locals() else {}
        }

    def get_market_condition(self, date):
        """获取该日期的市场环境（牛/熊/震荡）"""
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

    def get_adaptive_thresholds(self, market_condition):
        """根据市场环境返回自适应阈值"""
        if market_condition == '牛市':
            return {'strong_buy': {'enhanced_score': 0.65, 'quality': 0.6, 'score': 60},
                    'buy': {'enhanced_score': 0.6, 'quality': 0.55, 'score': 50},
                    'hold': {'enhanced_score': 0.5, 'quality': 0.5, 'score': 45},
                    'watch': {'enhanced_score': 0.3}}
        elif market_condition == '熊市':
            return {'strong_buy': {'enhanced_score': 0.75, 'quality': 0.75, 'score': 70},
                    'buy': {'enhanced_score': 0.7, 'quality': 0.7, 'score': 65},
                    'hold': {'enhanced_score': 0.6, 'quality': 0.6, 'score': 55},
                    'watch': {'enhanced_score': 0.3}}
        else:
            return {'strong_buy': {'enhanced_score': 0.7, 'quality': 0.65, 'score': 65},
                    'buy': {'enhanced_score': 0.65, 'quality': 0.6, 'score': 55},
                    'hold': {'enhanced_score': 0.55, 'quality': 0.55, 'score': 50},
                    'watch': {'enhanced_score': 0.3}}

    def run_real_backtest(self):
        print("🚀 推荐系统真实数据回测（每月）")
        print("=" * 80)
        dates = pd.date_range(self.start_date, self.end_date, freq=self.freq)
        all_results = []
        for date in dates:
            print(f"\n📅 回测日期: {date.strftime('%Y-%m-%d')}")
            market_condition = self.get_market_condition(date)
            thresholds = self.get_adaptive_thresholds(market_condition)
            print(f"  市场环境: {market_condition}，阈值: {thresholds}")
            month_results = []
            for symbol in self.test_stocks:
                factors = self.get_real_factors(symbol, date)
                if not factors:
                    continue
                # 获取策略信号
                try:
                    strategy_analysis = self.automation._analyze_strategy_signals(symbol)
                except Exception as e:
                    print(f"  {symbol} 策略信号失败: {e}")
                    strategy_analysis = {}
                # 组合数据
                stock = {
                    'symbol': symbol,
                    'quality_factor': factors['quality_factor'],
                    'multifactor_score': factors['multifactor_score'],
                    'enhanced_score': factors['enhanced_score'],
                    'enhanced_analysis': factors['enhanced_analysis'],
                    'strategy_analysis': strategy_analysis
                }
                advice = self.automation._get_enhanced_investment_advice(stock, thresholds=thresholds)
                month_results.append({'symbol': symbol, 'advice': advice, 'quality': factors['quality_factor'],
                                      'enhanced_score': factors['enhanced_score'],
                                      'score': factors['multifactor_score']})
                print(f"  {symbol}: {advice} (质量: {factors['quality_factor']:.2f}, 增强: {factors['enhanced_score']:.2f}, 综合: {factors['multifactor_score']})")
            all_results.append({'date': date, 'results': month_results})
        return all_results

def main():
    backtest = RecommendationSystemBacktest()
    results = backtest.run_real_backtest()
    print("\n🎉 真实数据回测完成！")

if __name__ == "__main__":
    main() 