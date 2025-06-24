#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年最优投资组合买入策略分析
基于当前持仓、推荐方案和外部环境预测
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class OptimalEntryStrategy2025:
    """2025年最优买入策略分析器"""
    
    def __init__(self):
        # 当前持仓 (2025年6月20日)
        self.current_holdings = {
            'NVDA': {'shares': 35, 'cost_basis': 137.942, 'weight': 18.29},
            'GOOG': {'shares': 30, 'cost_basis': 170.00, 'weight': 18.28},
            'AMD': {'shares': 30, 'cost_basis': 125.212, 'weight': 13.97},
            'PFE': {'shares': 80, 'cost_basis': 25.899, 'weight': 6.96},
            'MRK': {'shares': 8, 'cost_basis': 79.363, 'weight': 2.30},
            'BRK-B': {'shares': 2, 'cost_basis': 485.36, 'weight': 3.52}
        }
        
        # 最优推荐配置
        self.optimal_allocation = {
            # 成长股 50%
            'NVDA': {'target_weight': 8.0, 'category': '成长股'},
            'GOOG': {'target_weight': 12.0, 'category': '成长股'},
            'AMD': {'target_weight': 5.0, 'category': '成长股'},
            'META': {'target_weight': 12.0, 'category': '成长股'},
            'AMZN': {'target_weight': 8.0, 'category': '成长股'},
            'PLTR': {'target_weight': 5.0, 'category': '成长股'},
            
            # 价值成长 25%
            'JPM': {'target_weight': 8.0, 'category': '价值成长'},
            'BRK-B': {'target_weight': 8.0, 'category': '价值成长'},
            'ORCL': {'target_weight': 5.0, 'category': '价值成长'},
            'IBM': {'target_weight': 4.0, 'category': '价值成长'},
            
            # 防御股 25%
            'MRK': {'target_weight': 8.0, 'category': '防御股'},
            'JNJ': {'target_weight': 7.0, 'category': '防御股'},
            'VZ': {'target_weight': 5.0, 'category': '防御股'},
            'CVX': {'target_weight': 5.0, 'category': '防御股'}
        }
        
        # 2025年外部环境因素
        self.macro_factors_2025 = {
            'trade_war_impact': 0.15,  # 关税战影响
            'geopolitical_risk': 0.12,  # 地区冲突风险
            'dollar_weakness': 0.10,   # 美元疲软
            'economic_uncertainty': 0.18,  # 经济不稳定
            'volatility_increase': 0.25    # 波动性增加
        }
        
        self.total_assets = 27533.17
        self.available_cash = 4014.34 + 5988.46  # 现金 + 货币基金
        
    def get_current_market_data(self):
        """获取当前市场数据"""
        symbols = list(self.optimal_allocation.keys()) + ['VIX']
        
        try:
            data = {}
            for symbol in symbols:
                if symbol == 'VIX':
                    # VIX数据
                    vix = yf.Ticker('^VIX')
                    vix_data = vix.history(period='5d')
                    if not vix_data.empty:
                        data[symbol] = {
                            'current_price': vix_data['Close'].iloc[-1],
                            'volatility_level': 'High' if vix_data['Close'].iloc[-1] > 25 else 'Medium' if vix_data['Close'].iloc[-1] > 20 else 'Low'
                        }
                else:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='30d')
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                        
                        # 计算技术指标
                        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else sma_20
                        
                        # 52周高低点
                        high_52w = hist['High'].max()
                        low_52w = hist['Low'].min()
                        
                        data[symbol] = {
                            'current_price': current_price,
                            'volatility': volatility,
                            'sma_20': sma_20,
                            'sma_50': sma_50,
                            'high_52w': high_52w,
                            'low_52w': low_52w,
                            'distance_from_high': (high_52w - current_price) / high_52w,
                            'distance_from_low': (current_price - low_52w) / (high_52w - low_52w),
                            'pe_ratio': info.get('trailingPE', 0),
                            'market_cap': info.get('marketCap', 0),
                            'dividend_yield': info.get('dividendYield', 0) or 0
                        }
            
            return data
        except Exception as e:
            print(f"获取市场数据时出错: {e}")
            return {}
    
    def calculate_volatility_adjusted_entries(self, market_data):
        """基于2025年高波动环境计算调整后的买入点位"""
        
        vix_level = market_data.get('VIX', {}).get('current_price', 20)
        volatility_multiplier = 1.0
        
        # 根据VIX水平调整策略
        if vix_level > 30:
            volatility_multiplier = 1.3  # 高波动，更保守
        elif vix_level > 25:
            volatility_multiplier = 1.15
        elif vix_level < 15:
            volatility_multiplier = 0.9   # 低波动，可以更积极
        
        entry_strategies = {}
        
        for symbol, allocation in self.optimal_allocation.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                
                # 基础技术分析买入点
                technical_entry = self._calculate_technical_entry(data)
                
                # 2025年环境调整
                macro_adjustment = self._calculate_macro_adjustment(allocation['category'])
                
                # 波动性调整
                volatility_adjustment = volatility_multiplier
                
                # 综合买入策略
                conservative_entry = technical_entry * (1 + macro_adjustment) * volatility_adjustment
                aggressive_entry = technical_entry * (1 + macro_adjustment * 0.5) * volatility_adjustment
                
                # 分批买入点位
                entry_points = {
                    'immediate_entry': current_price * 0.98,  # 轻微回调即买入
                    'conservative_entry': conservative_entry,
                    'aggressive_entry': aggressive_entry,
                    'deep_dip_entry': current_price * 0.85,   # 深度回调买入
                }
                
                # 计算建议仓位
                target_value = self.total_assets * allocation['target_weight'] / 100
                current_value = self.current_holdings.get(symbol, {}).get('shares', 0) * current_price
                needed_investment = max(0, target_value - current_value)
                
                entry_strategies[symbol] = {
                    'current_price': current_price,
                    'entry_points': entry_points,
                    'target_value': target_value,
                    'current_value': current_value,
                    'needed_investment': needed_investment,
                    'suggested_shares': int(needed_investment / current_price) if needed_investment > 0 else 0,
                    'category': allocation['category'],
                    'target_weight': allocation['target_weight'],
                    'risk_level': self._assess_risk_level(data, allocation['category']),
                    'entry_timing': self._suggest_entry_timing(data, allocation['category'])
                }
        
        return entry_strategies
    
    def _calculate_technical_entry(self, data):
        """计算技术分析买入点位"""
        current_price = data['current_price']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        
        # 基于移动平均线的买入点
        if current_price > sma_20 > sma_50:  # 上升趋势
            return current_price * 0.95  # 5%回调买入
        elif current_price < sma_20:  # 下降趋势
            return sma_20 * 0.98  # 接近20日均线买入
        else:  # 震荡趋势
            return current_price * 0.97  # 3%回调买入
    
    def _calculate_macro_adjustment(self, category):
        """根据2025年宏观环境计算调整系数"""
        base_adjustment = sum(self.macro_factors_2025.values()) / len(self.macro_factors_2025)
        
        if category == '成长股':
            # 成长股在不确定环境下风险更高
            return base_adjustment * 1.2
        elif category == '价值成长':
            # 价值成长股相对稳定
            return base_adjustment * 0.8
        elif category == '防御股':
            # 防御股在不确定环境下更受青睐
            return base_adjustment * 0.5
        
        return base_adjustment
    
    def _assess_risk_level(self, data, category):
        """评估股票风险水平"""
        volatility = data['volatility']
        distance_from_high = data['distance_from_high']
        
        risk_score = 0
        
        # 波动性评分
        if volatility > 0.4:
            risk_score += 3
        elif volatility > 0.25:
            risk_score += 2
        else:
            risk_score += 1
        
        # 位置评分
        if distance_from_high > 0.2:  # 距离高点超过20%
            risk_score -= 1
        elif distance_from_high < 0.05:  # 接近历史高点
            risk_score += 2
        
        # 类别调整
        if category == '成长股':
            risk_score += 1
        elif category == '防御股':
            risk_score -= 1
        
        if risk_score <= 2:
            return '低风险'
        elif risk_score <= 4:
            return '中等风险'
        else:
            return '高风险'
    
    def _suggest_entry_timing(self, data, category):
        """建议买入时机"""
        current_price = data['current_price']
        sma_20 = data['sma_20']
        distance_from_high = data['distance_from_high']
        
        if category == '防御股':
            if distance_from_high > 0.1:
                return '立即买入'
            else:
                return '等待回调'
        elif category == '成长股':
            if current_price < sma_20 and distance_from_high > 0.15:
                return '逢低买入'
            elif distance_from_high < 0.05:
                return '等待深度回调'
            else:
                return '分批买入'
        else:  # 价值成长
            if distance_from_high > 0.08:
                return '分批买入'
            else:
                return '等待回调'
    
    def create_2025_volatility_scenario_analysis(self):
        """创建2025年波动性情景分析"""
        scenarios = {
            '基准情景': {
                'vix_range': (18, 25),
                'market_return': 0.08,
                'volatility_increase': 1.0,
                'probability': 0.4
            },
            '高波动情景': {
                'vix_range': (25, 35),
                'market_return': 0.03,
                'volatility_increase': 1.5,
                'probability': 0.35
            },
            '极端波动情景': {
                'vix_range': (35, 50),
                'market_return': -0.05,
                'volatility_increase': 2.0,
                'probability': 0.15
            },
            '低波动情景': {
                'vix_range': (12, 18),
                'market_return': 0.15,
                'volatility_increase': 0.7,
                'probability': 0.1
            }
        }
        
        return scenarios
    
    def generate_entry_execution_plan(self, entry_strategies):
        """生成买入执行计划"""
        
        print("=" * 80)
        print("🎯 2025年最优投资组合买入策略")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总资产: ${self.total_assets:,.2f}")
        print(f"可用资金: ${self.available_cash:,.2f}")
        print()
        
        # 2025年环境分析
        print("🌍 2025年外部环境评估:")
        print("-" * 50)
        for factor, impact in self.macro_factors_2025.items():
            factor_name = {
                'trade_war_impact': '关税贸易战',
                'geopolitical_risk': '地区冲突风险', 
                'dollar_weakness': '美元疲软',
                'economic_uncertainty': '经济不稳定',
                'volatility_increase': '市场波动加剧'
            }[factor]
            print(f"  {factor_name}: {impact:.1%} 影响")
        print()
        
        # 按优先级排序
        priority_order = []
        for symbol, strategy in entry_strategies.items():
            if strategy['needed_investment'] > 0:
                priority_score = self._calculate_priority_score(strategy)
                priority_order.append((symbol, strategy, priority_score))
        
        priority_order.sort(key=lambda x: x[2], reverse=True)
        
        print("📋 买入执行计划 (按优先级排序):")
        print("=" * 80)
        
        total_needed = 0
        execution_plan = []
        
        for i, (symbol, strategy, score) in enumerate(priority_order, 1):
            needed = strategy['needed_investment']
            total_needed += needed
            
            print(f"\n{i}. {symbol} - {strategy['category']}")
            print(f"   目标权重: {strategy['target_weight']:.1f}%")
            print(f"   当前价格: ${strategy['current_price']:.2f}")
            print(f"   需要投资: ${needed:,.0f}")
            print(f"   建议股数: {strategy['suggested_shares']}股")
            print(f"   风险等级: {strategy['risk_level']}")
            print(f"   买入时机: {strategy['entry_timing']}")
            print(f"   优先级评分: {score:.2f}")
            
            print(f"   📍 买入点位策略:")
            for entry_type, price in strategy['entry_points'].items():
                entry_name = {
                    'immediate_entry': '立即买入点',
                    'conservative_entry': '保守买入点',
                    'aggressive_entry': '积极买入点',
                    'deep_dip_entry': '深度回调点'
                }[entry_type]
                discount = (strategy['current_price'] - price) / strategy['current_price']
                print(f"     • {entry_name}: ${price:.2f} (折扣 {discount:.1%})")
            
            execution_plan.append({
                'symbol': symbol,
                'priority': i,
                'needed_investment': needed,
                'strategy': strategy
            })
        
        print(f"\n💰 资金需求分析:")
        print(f"   总需求资金: ${total_needed:,.0f}")
        print(f"   可用资金: ${self.available_cash:,.0f}")
        
        if total_needed > self.available_cash:
            print(f"   ⚠️  资金缺口: ${total_needed - self.available_cash:,.0f}")
            print(f"   建议: 分批投资或考虑减仓现有过重股票")
        else:
            print(f"   ✅ 资金充足，剩余: ${self.available_cash - total_needed:,.0f}")
        
        return execution_plan
    
    def _calculate_priority_score(self, strategy):
        """计算买入优先级评分"""
        score = 0
        
        # 基于类别的基础分数
        if strategy['category'] == '防御股':
            score += 3  # 2025年不确定环境下防御股优先
        elif strategy['category'] == '价值成长':
            score += 2
        else:  # 成长股
            score += 1
        
        # 基于风险等级调整
        if strategy['risk_level'] == '低风险':
            score += 2
        elif strategy['risk_level'] == '中等风险':
            score += 1
        
        # 基于买入时机调整
        if strategy['entry_timing'] == '立即买入':
            score += 2
        elif strategy['entry_timing'] == '分批买入':
            score += 1.5
        elif strategy['entry_timing'] == '逢低买入':
            score += 1
        
        # 基于需要投资金额调整（金额越大，优先级相对降低）
        if strategy['needed_investment'] > 5000:
            score -= 0.5
        
        return score
    
    def create_market_timing_dashboard(self, market_data):
        """创建市场择时仪表板"""
        
        print("\n" + "=" * 80)
        print("📊 2025年市场择时仪表板")
        print("=" * 80)
        
        # VIX分析
        if 'VIX' in market_data:
            vix_level = market_data['VIX']['current_price']
            vix_status = market_data['VIX']['volatility_level']
            
            print(f"📈 VIX恐慌指数: {vix_level:.2f} ({vix_status})")
            
            if vix_level > 30:
                print("   🚨 极度恐慌，优质股票可能出现超卖，考虑逢低买入")
            elif vix_level > 25:
                print("   ⚠️  市场恐慌，谨慎操作，分批建仓")
            elif vix_level > 20:
                print("   📊 正常波动，可以按计划执行买入")
            else:
                print("   😌 市场平静，可能过于乐观，注意风险")
        
        print()
        
        # 市场环境评估
        print("🌡️  2025年市场环境温度计:")
        print("-" * 50)
        
        risk_factors = [
            ("关税贸易战", "高风险", "🔴"),
            ("地区冲突", "中高风险", "🟠"),
            ("美元疲软", "中等风险", "🟡"),
            ("经济不稳定", "高风险", "🔴"),
            ("通胀压力", "中等风险", "🟡"),
            ("利率政策", "中等风险", "🟡")
        ]
        
        for factor, level, emoji in risk_factors:
            print(f"   {emoji} {factor}: {level}")
        
        print()
        
        # 投资建议
        print("💡 2025年投资策略建议:")
        print("-" * 50)
        print("   1. 🛡️  优先配置防御性资产 (MRK, JNJ, VZ)")
        print("   2. 📊 分批建仓，避免一次性大额投入")
        print("   3. 🎯 关注技术面支撑位，逢低买入优质股票")
        print("   4. ⚖️  保持适当现金比例，应对突发事件")
        print("   5. 🔄 定期重新平衡，控制单一股票权重")
        print("   6. 📈 利用波动性，在VIX>25时积极买入")
        
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("正在获取市场数据...")
        market_data = self.get_current_market_data()
        
        if not market_data:
            print("无法获取市场数据，使用模拟数据进行分析...")
            return
        
        print("正在计算买入策略...")
        entry_strategies = self.calculate_volatility_adjusted_entries(market_data)
        
        print("正在生成执行计划...")
        execution_plan = self.generate_entry_execution_plan(entry_strategies)
        
        print("正在创建市场择时分析...")
        self.create_market_timing_dashboard(market_data)
        
        # 情景分析
        scenarios = self.create_2025_volatility_scenario_analysis()
        
        print("\n" + "=" * 80)
        print("🎭 2025年市场情景分析")
        print("=" * 80)
        
        for scenario_name, scenario_data in scenarios.items():
            vix_range = scenario_data['vix_range']
            market_return = scenario_data['market_return']
            probability = scenario_data['probability']
            
            print(f"\n📋 {scenario_name}:")
            print(f"   VIX范围: {vix_range[0]}-{vix_range[1]}")
            print(f"   预期收益: {market_return:+.1%}")
            print(f"   发生概率: {probability:.1%}")
            
            if scenario_name == '高波动情景':
                print(f"   💡 策略: 重点买入防御股，减少成长股仓位")
            elif scenario_name == '极端波动情景':
                print(f"   💡 策略: 保持高现金比例，等待极度超卖机会")
            elif scenario_name == '低波动情景':
                print(f"   💡 策略: 积极买入成长股，提高风险资产比例")
            else:
                print(f"   💡 策略: 按既定计划分批建仓")
        
        print("\n" + "=" * 80)
        print("✅ 分析完成！请根据市场实际情况调整执行计划。")
        print("⚠️  投资有风险，决策需谨慎！")
        print("=" * 80)
        
        return execution_plan

if __name__ == "__main__":
    analyzer = OptimalEntryStrategy2025()
    execution_plan = analyzer.run_comprehensive_analysis() 