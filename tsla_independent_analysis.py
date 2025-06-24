#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特斯拉(TSLA)独立投资分析
与AMD对比，避免被割韭菜
"""

import asyncio
import sys
import os
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_interface import DataInterface

class TSLAAnalyzer:
    """特斯拉独立分析器"""
    
    def __init__(self):
        self.data_interface = DataInterface()
        self.yahoo_source = self.data_interface.get_data_source('yahoo')
        
        # 基本面数据 (来自缓存)
        self.fundamentals = {
            'pe_ratio': 189.16,
            'eps': 1.74,
            'earnings_growth': -0.707,  # -70.7%
            'revenue_growth': -0.092,   # -9.2%
            'analyst_target': 301.56,
            'analyst_rating': 'Hold',
            'market_cap': '1.06T',
            'beta': 2.461,  # 高波动性
            'profit_margin': 0.0638,
            'debt_to_equity': 17.407
        }
    
    async def comprehensive_analysis(self):
        """综合分析"""
        print("🚗 TESLA (TSLA) 独立投资分析")
        print("="*60)
        print("🎯 目标：理性分析，避免情绪化投资")
        print()
        
        # 获取实时数据
        tech_data = await self._get_technical_data()
        
        if tech_data:
            # 技术面分析
            tech_score = self._analyze_technical(tech_data)
            
            # 基本面分析
            fundamental_score = self._analyze_fundamentals()
            
            # 市场情绪分析
            sentiment_score = self._analyze_market_sentiment(tech_data)
            
            # 综合评分 (波段股权重：技术50% + 基本面40% + 情绪10%)
            total_score = (tech_score * 0.5 + fundamental_score * 0.4 + sentiment_score * 0.1)
            
            # 投资建议
            self._generate_recommendation(total_score, tech_data)
            
            # 与AMD对比
            await self._compare_with_amd(tech_data)
            
        else:
            print("❌ 无法获取TSLA数据，使用历史分析")
            self._fallback_analysis()
    
    async def _get_technical_data(self):
        """获取技术数据"""
        try:
            realtime_data = await self.yahoo_source.get_realtime_data(['TSLA'], timeframe='1d')
            
            if 'TSLA' in realtime_data and not realtime_data['TSLA'].empty:
                df = realtime_data['TSLA']
                latest = df.iloc[-1]
                
                current_price = float(latest['close'])
                volume = int(latest['volume']) if latest['volume'] > 0 else 0
                
                # 计算技术指标
                closes = df['close'].values
                
                # RSI
                rsi = self._calculate_rsi(closes) if len(closes) >= 14 else 50
                
                # 移动平均线
                ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
                ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
                
                # 价格区间
                high_52w = np.max(df['high'].values)
                low_52w = np.min(df['low'].values)
                price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
                
                return {
                    'current_price': current_price,
                    'volume': volume,
                    'rsi': rsi,
                    'ma20': ma20,
                    'ma50': ma50,
                    'high_52w': high_52w,
                    'low_52w': low_52w,
                    'price_position': price_position
                }
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            
        return None
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _analyze_technical(self, data):
        """技术面分析"""
        print("🔧 技术面分析:")
        
        current_price = data['current_price']
        rsi = data['rsi']
        ma20 = data['ma20']
        ma50 = data['ma50']
        price_position = data['price_position']
        volume = data['volume']
        
        print(f"   💰 当前价格: ${current_price:.2f}")
        print(f"   📊 RSI: {rsi:.1f} ({'超买' if rsi > 70 else '超卖' if rsi < 30 else '中性'})")
        print(f"   📈 MA20: ${ma20:.2f} ({'上方' if current_price > ma20 else '下方'})")
        print(f"   📈 MA50: ${ma50:.2f} ({'上方' if current_price > ma50 else '下方'})")
        print(f"   📍 价格位置: {price_position:.1f}% (52周区间)")
        print(f"   📦 成交量: {volume:,}")
        
        # 技术面评分
        score = 5  # 基础分
        
        # RSI评分
        if 30 <= rsi <= 70:
            score += 2
            print(f"   ✓ RSI在合理区间 (+2分)")
        elif rsi < 30:
            score += 3
            print(f"   ✓ RSI超卖，可能反弹 (+3分)")
        else:
            score -= 1
            print(f"   ⚠ RSI超买，注意风险 (-1分)")
        
        # 均线评分
        if current_price > ma20 > ma50:
            score += 2
            print(f"   ✓ 价格在均线上方，趋势向上 (+2分)")
        elif current_price > ma20:
            score += 1
            print(f"   ✓ 价格在20日均线上方 (+1分)")
        else:
            score -= 1
            print(f"   ⚠ 价格在均线下方 (-1分)")
        
        # 价格位置评分
        if price_position < 30:
            score += 2
            print(f"   ✓ 价格在低位，上涨空间大 (+2分)")
        elif price_position > 80:
            score -= 2
            print(f"   ⚠ 价格在高位，风险较大 (-2分)")
        
        score = max(0, min(10, score))  # 限制在0-10分
        print(f"   🎯 技术面评分: {score}/10")
        
        return score
    
    def _analyze_fundamentals(self):
        """基本面分析"""
        print(f"\n📊 基本面分析:")
        
        pe = self.fundamentals['pe_ratio']
        eps = self.fundamentals['eps']
        earnings_growth = self.fundamentals['earnings_growth']
        revenue_growth = self.fundamentals['revenue_growth']
        
        print(f"   💼 PE比率: {pe:.1f} (极高估值)")
        print(f"   💰 EPS: ${eps:.2f}")
        print(f"   📈 收益增长: {earnings_growth:.1%} (大幅下滑)")
        print(f"   📊 营收增长: {revenue_growth:.1%} (负增长)")
        print(f"   🎯 分析师目标价: ${self.fundamentals['analyst_target']:.2f}")
        print(f"   📝 分析师评级: {self.fundamentals['analyst_rating']}")
        
        # 基本面评分
        score = 2  # 基础分很低
        
        # PE评分 (对于成长股，PE<50合理)
        if pe > 200:
            score += 0
            print(f"   ❌ PE超过200，估值泡沫严重 (+0分)")
        elif pe > 100:
            score += 1
            print(f"   ⚠ PE超过100，估值偏高 (+1分)")
        else:
            score += 3
            print(f"   ✓ PE相对合理 (+3分)")
        
        # 增长率评分
        if earnings_growth < -0.5:
            score += 0
            print(f"   ❌ 收益大幅下滑，基本面恶化 (+0分)")
        elif earnings_growth < 0:
            score += 1
            print(f"   ⚠ 收益负增长 (+1分)")
        else:
            score += 3
            print(f"   ✓ 收益正增长 (+3分)")
        
        # 行业地位
        score += 2
        print(f"   ✓ 电动车龙头地位 (+2分)")
        
        score = max(0, min(10, score))
        print(f"   🎯 基本面评分: {score}/10")
        
        return score
    
    def _analyze_market_sentiment(self, data):
        """市场情绪分析"""
        print(f"\n😊 市场情绪分析:")
        
        volume = data['volume']
        beta = self.fundamentals['beta']
        
        print(f"   📦 成交量: {volume:,}")
        print(f"   📊 Beta系数: {beta} (高波动性)")
        
        # 情绪评分
        score = 5  # 基础分
        
        # 成交量分析
        if volume > 100000000:  # 1亿股以上
            score += 3
            print(f"   ✓ 成交量活跃，市场关注度高 (+3分)")
        elif volume > 50000000:
            score += 2
            print(f"   ✓ 成交量较活跃 (+2分)")
        else:
            score += 1
            print(f"   ⚠ 成交量一般 (+1分)")
        
        # Beta分析
        if beta > 2:
            score += 1  # 高波动既是机会也是风险
            print(f"   ⚠ 高波动性，适合波段交易 (+1分)")
        
        score = max(0, min(10, score))
        print(f"   🎯 市场情绪评分: {score}/10")
        
        return score
    
    def _generate_recommendation(self, total_score, data):
        """生成投资建议"""
        print(f"\n💡 综合投资分析:")
        print(f"   🎯 综合评分: {total_score:.1f}/10")
        
        current_price = data['current_price']
        
        if total_score >= 7:
            recommendation = "🟢 买入"
            action = "可以考虑建仓"
        elif total_score >= 5:
            recommendation = "🟡 持有/观望"
            action = "等待更好时机"
        else:
            recommendation = "🔴 谨慎/减持"
            action = "不建议新增投资"
        
        print(f"   📝 投资建议: {recommendation}")
        print(f"   🎬 操作建议: {action}")
        
        # 具体操作建议
        print(f"\n🎯 具体操作建议:")
        if total_score >= 6:
            buy_price = current_price * 0.95
            stop_loss = current_price * 0.85
            target_price = current_price * 1.20
            
            print(f"   💰 建议买入价: ${buy_price:.2f} (当前价-5%)")
            print(f"   🛡️ 止损价: ${stop_loss:.2f} (当前价-15%)")
            print(f"   🎯 目标价: ${target_price:.2f} (当前价+20%)")
        else:
            print(f"   ⚠️ 当前不建议买入，等待更好时机")
        
        # 风险提示
        print(f"\n⚠️ 风险提示:")
        print(f"   • 估值极高，PE比率{self.fundamentals['pe_ratio']:.0f}倍")
        print(f"   • 收益大幅下滑{self.fundamentals['earnings_growth']:.1%}")
        print(f"   • 高波动性股票，适合波段交易")
        print(f"   • 不适合保守投资者")
        print(f"   • 需要严格止损纪律")
    
    async def _compare_with_amd(self, tsla_data):
        """与AMD对比分析"""
        print(f"\n🔄 TSLA vs AMD 对比分析:")
        
        try:
            # 获取AMD数据
            amd_data = await self.yahoo_source.get_realtime_data(['AMD'], timeframe='1d')
            
            if 'AMD' in amd_data and not amd_data['AMD'].empty:
                amd_df = amd_data['AMD']
                amd_price = float(amd_df.iloc[-1]['close'])
                amd_volume = int(amd_df.iloc[-1]['volume'])
            else:
                amd_price = 137.0  # 估算价格
                amd_volume = 50000000
            
            print(f"   📊 价格对比:")
            print(f"      TSLA: ${tsla_data['current_price']:.2f}")
            print(f"      AMD:  ${amd_price:.2f}")
            
            print(f"   📦 成交量对比:")
            print(f"      TSLA: {tsla_data['volume']:,}")
            print(f"      AMD:  {amd_volume:,}")
            
            print(f"   🏢 基本面对比:")
            print(f"      TSLA: PE 189倍, 收益-70.7%")
            print(f"      AMD:  PE ~50倍, AI芯片龙头")
            
            print(f"   💡 投资建议:")
            print(f"      • AMD: 相对稳健，基本面较好")
            print(f"      • TSLA: 高风险高收益，需要时机")
            print(f"      • 如果只能选一个：建议AMD")
            print(f"      • 如果都想要：AMD 70% + TSLA 30%")
            
        except Exception as e:
            print(f"   ❌ AMD数据获取失败: {e}")
    
    def _fallback_analysis(self):
        """备用分析"""
        print("📊 基于历史数据的分析:")
        print("   • TSLA估值过高，PE比率189倍")
        print("   • 收益大幅下滑70.7%")
        print("   • 高波动性，Beta系数2.46")
        print("   • 适合波段交易，不适合长期持有")
        print("   • 建议等待更好的买入时机")

async def main():
    """主函数"""
    analyzer = TSLAAnalyzer()
    await analyzer.comprehensive_analysis()
    
    print(f"\n🎯 最终结论:")
    print(f"   如果你已经有AMD持仓且表现良好")
    print(f"   建议专注AMD，不要分散注意力")
    print(f"   TSLA风险太高，容易被割韭菜")
    print(f"   投资要有纪律，不要追热点")

if __name__ == "__main__":
    asyncio.run(main()) 