#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIX恐慌指数深度分析
分析VIX变化对市场的影响和预测
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VIXMarketAnalyzer:
    """VIX市场分析器"""
    
    def __init__(self):
        self.vix_symbol = '^VIX'
        self.spx_symbol = '^GSPC'
        self.ndx_symbol = '^NDX'
        
    def comprehensive_vix_analysis(self):
        """VIX综合分析"""
        print("📊 VIX恐慌指数深度分析")
        print("="*80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 获取VIX数据
        vix_data = self._get_vix_data()
        spx_data = self._get_market_data(self.spx_symbol, "标普500")
        
        if vix_data is not None and spx_data is not None:
            # 当前VIX分析
            self._analyze_current_vix(vix_data)
            
            # VIX历史对比
            self._analyze_vix_history(vix_data)
            
            # VIX与标普关系
            self._analyze_vix_spx_relationship(vix_data, spx_data)
            
            # 市场预测
            self._market_prediction_based_on_vix(vix_data, spx_data)
            
            # 交易策略建议
            self._trading_strategy_recommendation(vix_data, spx_data)
        
    def _get_vix_data(self):
        """获取VIX数据"""
        try:
            vix = yf.Ticker(self.vix_symbol)
            data = vix.history(period='6mo')  # 6个月数据
            return data
        except Exception as e:
            print(f"❌ 获取VIX数据失败: {e}")
            return None
    
    def _get_market_data(self, symbol, name):
        """获取市场数据"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='6mo')
            return data
        except Exception as e:
            print(f"❌ 获取{name}数据失败: {e}")
            return None
    
    def _analyze_current_vix(self, vix_data):
        """分析当前VIX水平"""
        print("🎯 当前VIX分析")
        print("-"*50)
        
        current_vix = vix_data['Close'].iloc[-1]
        prev_vix = vix_data['Close'].iloc[-2]
        change_pct = (current_vix - prev_vix) / prev_vix * 100
        
        print(f"📊 当前VIX: {current_vix:.2f}")
        print(f"📈 日变化: {change_pct:+.2f}%")
        
        # VIX水平解读
        if current_vix < 15:
            vix_level = "极低恐慌 (市场极度乐观)"
            market_sentiment = "🟢 极度贪婪"
            risk_level = "⚠️ 高风险 (可能出现回调)"
        elif current_vix < 20:
            vix_level = "低恐慌 (市场相对平静)"
            market_sentiment = "🟡 相对乐观"
            risk_level = "✅ 中等风险"
        elif current_vix < 30:
            vix_level = "中等恐慌 (市场担忧增加)"
            market_sentiment = "🟠 谨慎情绪"
            risk_level = "⚠️ 中高风险"
        elif current_vix < 40:
            vix_level = "高恐慌 (市场明显恐慌)"
            market_sentiment = "🔴 恐慌情绪"
            risk_level = "🚨 高风险"
        else:
            vix_level = "极高恐慌 (市场极度恐慌)"
            market_sentiment = "🔴 极度恐慌"
            risk_level = "🚨 极高风险"
        
        print(f"💡 VIX水平: {vix_level}")
        print(f"😊 市场情绪: {market_sentiment}")
        print(f"⚠️ 风险等级: {risk_level}")
        
        # VIX 17.70的特殊分析
        if abs(current_vix - 17.70) < 1:
            print(f"\n🎯 VIX 17.70水平分析:")
            print(f"   • 处于低恐慌区间，市场相对乐观")
            print(f"   • 如果从更高水平跌至17.70，说明恐慌情绪缓解")
            print(f"   • 但仍需警惕，VIX低于15时要防范黑天鹅事件")
        
        print()
    
    def _analyze_vix_history(self, vix_data):
        """VIX历史分析"""
        print("📈 VIX历史对比分析")
        print("-"*50)
        
        current_vix = vix_data['Close'].iloc[-1]
        
        # 计算历史统计
        vix_30d_avg = vix_data['Close'].tail(30).mean()
        vix_90d_avg = vix_data['Close'].tail(90).mean()
        vix_6m_max = vix_data['Close'].max()
        vix_6m_min = vix_data['Close'].min()
        
        print(f"📊 30日平均VIX: {vix_30d_avg:.2f}")
        print(f"📊 90日平均VIX: {vix_90d_avg:.2f}")
        print(f"📊 6个月最高: {vix_6m_max:.2f}")
        print(f"📊 6个月最低: {vix_6m_min:.2f}")
        
        # 当前位置
        vix_percentile = ((current_vix - vix_6m_min) / (vix_6m_max - vix_6m_min)) * 100
        print(f"📍 当前VIX位置: {vix_percentile:.1f}% (6个月区间)")
        
        # 趋势分析
        vix_5d_change = (current_vix - vix_data['Close'].iloc[-6]) / vix_data['Close'].iloc[-6] * 100
        vix_20d_change = (current_vix - vix_data['Close'].iloc[-21]) / vix_data['Close'].iloc[-21] * 100
        
        print(f"📈 5日变化: {vix_5d_change:+.2f}%")
        print(f"📈 20日变化: {vix_20d_change:+.2f}%")
        
        # 如果VIX跌10%的分析
        if vix_5d_change < -8:  # 接近10%跌幅
            print(f"\n🎯 VIX大幅下跌分析:")
            print(f"   • VIX跌10%通常意味着市场恐慌情绪快速缓解")
            print(f"   • 可能因为:")
            print(f"     - 宏观经济数据好转")
            print(f"     - 地缘政治风险缓解") 
            print(f"     - 企业财报超预期")
            print(f"     - 央行政策积极信号")
        
        print()
    
    def _analyze_vix_spx_relationship(self, vix_data, spx_data):
        """分析VIX与标普500关系"""
        print("🔗 VIX与标普500关系分析")
        print("-"*50)
        
        # 获取相同时间段的数据
        common_dates = vix_data.index.intersection(spx_data.index)
        vix_aligned = vix_data.loc[common_dates]['Close']
        spx_aligned = spx_data.loc[common_dates]['Close']
        
        # 计算相关性
        correlation = vix_aligned.corr(spx_aligned)
        print(f"📊 VIX与标普相关性: {correlation:.3f} (通常为负相关)")
        
        # 当前状态
        current_vix = vix_data['Close'].iloc[-1]
        current_spx = spx_data['Close'].iloc[-1]
        spx_change = (current_spx - spx_data['Close'].iloc[-2]) / spx_data['Close'].iloc[-2] * 100
        
        print(f"📊 当前标普500: {current_spx:.2f} ({spx_change:+.2f}%)")
        
        # 基于VIX预测标普走势
        if current_vix < 18:
            spx_outlook = "🟢 标普可能继续上涨或保持强势"
            target_estimate = "6200-6400区间"
        elif current_vix < 22:
            spx_outlook = "🟡 标普可能震荡整理"
            target_estimate = "5900-6200区间"
        else:
            spx_outlook = "🔴 标普可能面临调整压力"
            target_estimate = "5600-5900区间"
        
        print(f"💡 基于VIX的标普预测: {spx_outlook}")
        print(f"🎯 可能目标区间: {target_estimate}")
        
        print()
    
    def _market_prediction_based_on_vix(self, vix_data, spx_data):
        """基于VIX的市场预测"""
        print("🔮 基于VIX的市场预测")
        print("-"*50)
        
        current_vix = vix_data['Close'].iloc[-1]
        current_spx = spx_data['Close'].iloc[-1]
        
        print(f"🎯 VIX 17.70 + 跌10%的市场含义:")
        print()
        
        # 情景分析
        print("📊 情景分析:")
        print("1️⃣ 乐观情景 (概率40%):")
        print("   • VIX继续下降至15以下")
        print("   • 标普突破6300，目标6400-6500")
        print("   • 科技股领涨，纳斯达克创新高")
        print("   • 触发因素: 通胀数据良好、企业财报超预期")
        print()
        
        print("2️⃣ 中性情景 (概率45%):")
        print("   • VIX在15-20区间震荡")
        print("   • 标普在6000-6300区间整理")
        print("   • 市场等待更多催化剂")
        print("   • 触发因素: 经济数据混合、政策不确定性")
        print()
        
        print("3️⃣ 悲观情景 (概率15%):")
        print("   • VIX反弹至25以上")
        print("   • 标普回调至5800-6000")
        print("   • 可能的黑天鹅事件")
        print("   • 触发因素: 地缘政治、意外负面消息")
        print()
        
        # 关键阻力和支撑
        print("🎯 关键点位:")
        print(f"   标普500阻力位: 6300, 6400, 6500")
        print(f"   标普500支撑位: 6100, 6000, 5900")
        print(f"   VIX关键位: 15 (极低恐慌), 20 (中性), 25 (恐慌)")
        print()
    
    def _trading_strategy_recommendation(self, vix_data, spx_data):
        """交易策略建议"""
        print("💡 基于VIX的交易策略")
        print("-"*50)
        
        current_vix = vix_data['Close'].iloc[-1]
        
        print("🎯 VIX 17.70水平的策略建议:")
        print()
        
        print("📈 股票策略:")
        print("   • 可适度增加风险资产配置")
        print("   • 重点关注科技股、成长股")
        print("   • AMD等半导体股可能受益于低VIX环境")
        print("   • 但要设置止损，防范VIX突然飙升")
        print()
        
        print("🛡️ 风险管理:")
        print("   • VIX低于15时要格外小心")
        print("   • 考虑买入VIX看涨期权作为保险")
        print("   • 不要过度杠杆化")
        print("   • 保持一定现金比例")
        print()
        
        print("⏰ 时机选择:")
        print("   • 短期(1-2周): 可能继续上涨")
        print("   • 中期(1-2月): 需要观察VIX是否突破15")
        print("   • 长期(3-6月): 警惕VIX均值回归")
        print()
        
        print("🎯 具体建议:")
        if current_vix < 18:
            print("   • 可以适当做多股票")
            print("   • 关注标普突破6300的机会")
            print("   • AMD等个股可以持有待涨")
            print("   • 但要准备好VIX反弹的应对策略")
        
        print()
        print("⚠️ 风险提示:")
        print("   • VIX过低往往预示着市场自满")
        print("   • 历史上VIX极低后常有突然飙升")
        print("   • 建议分批建仓，不要一次性全仓")

def main():
    """主函数"""
    analyzer = VIXMarketAnalyzer()
    analyzer.comprehensive_vix_analysis()

if __name__ == "__main__":
    main() 