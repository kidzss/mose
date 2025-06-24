#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025年最终买入点位策略
基于波动性验证、技术分析和基本面分析的综合建议
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinalEntryStrategy2025:
    """2025年最终买入策略"""
    
    def __init__(self):
        # 目标配置（最优方案）
        self.target_allocation = {
            'META': {'target_weight': 12.0, 'category': '成长股', 'priority': 'BUY'},
            'AMZN': {'target_weight': 8.0, 'category': '成长股', 'priority': 'BUY'},
            'PLTR': {'target_weight': 5.0, 'category': '成长股', 'priority': 'BUY'},
            'JPM': {'target_weight': 8.0, 'category': '价值成长', 'priority': 'BUY'},
            'BRK-B': {'target_weight': 8.0, 'category': '价值成长', 'priority': 'BUY'},
            'ORCL': {'target_weight': 5.0, 'category': '价值成长', 'priority': 'BUY'},
            'IBM': {'target_weight': 4.0, 'category': '价值成长', 'priority': 'BUY'},
            'MRK': {'target_weight': 8.0, 'category': '防御股', 'priority': 'BUY'},
            'JNJ': {'target_weight': 7.0, 'category': '防御股', 'priority': 'BUY'},
            'VZ': {'target_weight': 5.0, 'category': '防御股', 'priority': 'BUY'},
            'CVX': {'target_weight': 5.0, 'category': '防御股', 'priority': 'BUY'}
        }
        
        self.total_assets = 27533.17
        self.available_cash = 10002.80
    
    def get_real_time_data(self):
        """获取实时市场数据"""
        print("📊 获取实时市场数据...")
        
        all_symbols = list(self.target_allocation.keys()) + ['^VIX']
        market_data = {}
        
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='60d')
                info = ticker.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else ma_20
                    
                    # RSI计算
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]
                    
                    market_data[symbol] = {
                        'current_price': current_price,
                        'volatility': volatility,
                        'ma_20': ma_20,
                        'ma_50': ma_50,
                        'rsi': rsi,
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                    }
                    
            except Exception as e:
                print(f"获取{symbol}数据时出错: {e}")
                
        return market_data
    
    def calculate_entry_points(self, market_data):
        """计算买入点位"""
        print("\n" + "=" * 80)
        print("🎯 2025年高波动环境下的最优买入策略")
        print("=" * 80)
        
        current_vix = market_data.get('^VIX', {}).get('current_price', 20)
        print(f"📊 当前VIX水平: {current_vix:.2f}")
        
        if current_vix > 35:
            market_regime = "极度恐慌"
        elif current_vix > 28:
            market_regime = "高度波动"
        elif current_vix > 20:
            market_regime = "中等波动"
        else:
            market_regime = "相对平静"
        
        print(f"🌡️  市场状态: {market_regime}")
        
        recommendations = []
        
        for symbol, config in self.target_allocation.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                target_amount = self.total_assets * (config['target_weight'] / 100)
                
                # 基于VIX计算买入点位
                if current_vix > 35:
                    immediate_buy = current_price * 1.02
                    conservative_buy = current_price * 0.95
                    aggressive_buy = current_price * 0.90
                elif current_vix > 28:
                    immediate_buy = current_price * 0.98
                    conservative_buy = current_price * 0.93
                    aggressive_buy = current_price * 0.88
                else:
                    immediate_buy = current_price * 0.97
                    conservative_buy = current_price * 0.92
                    aggressive_buy = current_price * 0.85
                
                # RSI调整
                rsi = data['rsi']
                if rsi > 70:
                    immediate_buy *= 0.95
                    conservative_buy *= 0.93
                elif rsi < 30:
                    immediate_buy *= 1.02
                    conservative_buy *= 1.05
                
                # 防御股特殊处理
                if config['category'] == '防御股' and current_vix > 25:
                    immediate_buy *= 1.03
                    conservative_buy *= 1.02
                
                # 风险评分
                risk_score = 1 if config['category'] == '防御股' else 2 if config['category'] == '价值成长' else 3
                if data['volatility'] > 0.4:
                    risk_score += 2
                elif data['volatility'] > 0.25:
                    risk_score += 1
                
                recommendations.append({
                    'symbol': symbol,
                    'category': config['category'],
                    'current_price': current_price,
                    'target_amount': target_amount,
                    'immediate_buy': round(immediate_buy, 2),
                    'conservative_buy': round(conservative_buy, 2),
                    'aggressive_buy': round(aggressive_buy, 2),
                    'stop_loss': round(current_price * 0.85, 2),
                    'risk_score': risk_score,
                    'pe_ratio': data.get('pe_ratio', 0),
                    'dividend_yield': data.get('dividend_yield', 0),
                    'rsi': rsi
                })
        
        # 按风险评分排序
        recommendations.sort(key=lambda x: (x['risk_score'], -x['dividend_yield']))
        
        return recommendations, market_regime
    
    def generate_execution_plan(self, recommendations, market_regime):
        """生成执行计划"""
        print("\n" + "=" * 80)
        print("📋 分阶段执行计划")
        print("=" * 80)
        
        total_needed = sum(rec['target_amount'] for rec in recommendations)
        print(f"💰 总资金需求: ${total_needed:,.0f}")
        print(f"💰 可用资金: ${self.available_cash:,.0f}")
        
        if total_needed > self.available_cash:
            print(f"⚠️  资金缺口: ${total_needed - self.available_cash:,.0f}")
            print("💡 建议: 分批投资或减仓GOOG/AMD释放资金")
        
        print(f"\n🎯 {market_regime}环境执行策略:")
        print("-" * 60)
        
        # 第一阶段：低风险股票
        print("\n📅 第一阶段 (立即执行):")
        print(f"   预算: ${self.available_cash * 0.4:,.0f}")
        
        phase_1_stocks = [rec for rec in recommendations if rec['risk_score'] <= 3][:3]
        for i, stock in enumerate(phase_1_stocks, 1):
            shares = int(stock['target_amount'] / stock['current_price'])
            print(f"   {i}. {stock['symbol']} ({stock['category']})")
            print(f"      目标股数: {shares}股")
            print(f"      当前价格: ${stock['current_price']:.2f}")
            print(f"      立即买入点: ${stock['immediate_buy']:.2f}")
            print(f"      保守买入点: ${stock['conservative_buy']:.2f}")
            print(f"      止损点: ${stock['stop_loss']:.2f}")
            print(f"      PE: {stock['pe_ratio']:.1f}, 股息: {stock['dividend_yield']:.1f}%")
            print()
        
        # 第二阶段：中等风险股票
        print("📅 第二阶段 (1-4周后):")
        print(f"   预算: ${self.available_cash * 0.35:,.0f}")
        remaining_stocks = [rec for rec in recommendations if rec['risk_score'] > 3 and rec['risk_score'] <= 5]
        for i, stock in enumerate(remaining_stocks[:4], 1):
            print(f"   {i}. {stock['symbol']} - 等待回调至${stock['conservative_buy']:.2f}")
        
        # 第三阶段：高风险股票
        print("\n📅 第三阶段 (1-3个月后):")
        print(f"   预算: ${self.available_cash * 0.25:,.0f}")
        high_risk_stocks = [rec for rec in recommendations if rec['risk_score'] > 5]
        for i, stock in enumerate(high_risk_stocks[:3], 1):
            print(f"   {i}. {stock['symbol']} - 等待深度回调至${stock['aggressive_buy']:.2f}")
    
    def create_timing_dashboard(self, market_data):
        """创建择时仪表板"""
        print("\n" + "=" * 80)
        print("📊 2025年市场择时仪表板")
        print("=" * 80)
        
        current_vix = market_data.get('^VIX', {}).get('current_price', 20)
        print(f"📈 VIX恐慌指数: {current_vix:.2f}")
        
        if current_vix > 35:
            print("   🚨 极度恐慌 - 绝佳买入机会！")
        elif current_vix > 28:
            print("   ⚠️  高度波动 - 分批买入防御股")
        elif current_vix > 20:
            print("   📊 中等波动 - 按计划执行")
        else:
            print("   😌 相对平静 - 可适度积极")
        
        # 关键股票信号
        print("\n🎯 关键股票技术面信号:")
        print("-" * 50)
        
        priority_stocks = ['MRK', 'JNJ', 'VZ', 'JPM', 'PLTR']
        for symbol in priority_stocks:
            if symbol in market_data:
                data = market_data[symbol]
                price = data['current_price']
                rsi = data['rsi']
                
                if rsi < 30:
                    status = "🟢 超卖买入"
                elif rsi > 70:
                    status = "🔴 超买等待"
                else:
                    status = "🟡 中性观望"
                
                print(f"   {symbol}: ${price:.2f} RSI:{rsi:.1f} {status}")
        
        # 市场建议
        print("\n💡 当前策略建议:")
        print("-" * 50)
        if current_vix > 30:
            print("   1. 🛡️  立即买入防御股")
            print("   2. 💰 使用60%资金")
            print("   3. 🎯 重点MRK, JNJ, VZ")
        elif current_vix > 25:
            print("   1. 🔄 分批买入")
            print("   2. 💰 使用40-50%资金")
            print("   3. 📊 防御股优先")
        else:
            print("   1. ⏳ 等待机会")
            print("   2. 💰 保持现金")
            print("   3. 🎯 关注突破")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🎯 开始2025年最终买入策略分析...")
        print("=" * 80)
        
        market_data = self.get_real_time_data()
        if not market_data:
            print("❌ 无法获取市场数据")
            return
        
        recommendations, market_regime = self.calculate_entry_points(market_data)
        self.generate_execution_plan(recommendations, market_regime)
        self.create_timing_dashboard(market_data)
        
        print("\n" + "=" * 80)
        print("🎯 最终策略总结")
        print("=" * 80)
        print("✅ 2025年确实是高波动年份")
        print("📊 当前需谨慎分批买入")
        print("🛡️  优先配置防御性资产")
        print("💰 保持充足现金应对突发")
        print("🎯 严格按技术面信号执行")
        print("⚠️  投资有风险，决策需谨慎")
        print("=" * 80)

if __name__ == "__main__":
    strategy = FinalEntryStrategy2025()
    strategy.run_analysis() 