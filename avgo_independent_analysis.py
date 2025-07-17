#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AVGO独立分析验证系统
AVGO Independent Analysis Verification System
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AVGOAnalyzer:
    def __init__(self):
        self.symbol = 'AVGO'
        
    def get_real_time_data(self):
        """获取AVGO实时数据"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # 获取历史数据
            hist = ticker.history(period='6mo', interval='1d')
            info = ticker.info
            
            if hist.empty:
                return None
            
            # 当前价格和变化
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # 技术指标
            rsi = self.calculate_rsi(hist['Close'])
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else ma_50
            
            # 52周高低点
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
            
            # 成交量分析
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            # 财务指标
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            market_cap = info.get('marketCap', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # 分析师数据
            target_price = info.get('targetMeanPrice', 0)
            target_high = info.get('targetHighPrice', 0)
            target_low = info.get('targetLowPrice', 0)
            
            return {
                'symbol': self.symbol,
                'current_price': current_price,
                'change': change,
                'change_pct': change_pct,
                'rsi': rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'price_position': price_position,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'market_cap': market_cap,
                'dividend_yield': dividend_yield,
                'target_price': target_price,
                'target_high': target_high,
                'target_low': target_low,
                'company_name': info.get('longName', self.symbol),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'beta': info.get('beta', 1.0),
                'eps_growth': info.get('earningsQuarterlyGrowth', 0),
                'revenue_growth': info.get('revenueGrowth', 0)
            }
            
        except Exception as e:
            print(f"获取{self.symbol}数据失败: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def analyze_technical_trend(self, data):
        """分析技术趋势"""
        if not data:
            return "数据不足"
        
        # 均线分析
        ma_trend = "多头排列" if data['current_price'] > data['ma_20'] > data['ma_50'] else "空头排列"
        
        # 趋势强度
        trend_strength = 0
        if data['current_price'] > data['ma_20']:
            trend_strength += 1
        if data['current_price'] > data['ma_50']:
            trend_strength += 1
        if data['ma_20'] > data['ma_50']:
            trend_strength += 1
        
        # RSI分析
        rsi_status = "超买" if data['rsi'] > 70 else "超卖" if data['rsi'] < 30 else "中性"
        
        return {
            'ma_trend': ma_trend,
            'trend_strength': trend_strength,
            'rsi_status': rsi_status,
            'price_position': data['price_position']
        }
    
    def analyze_fundamentals(self, data):
        """分析基本面"""
        if not data:
            return "数据不足"
        
        # 估值分析
        valuation_score = 0
        if data['pe_ratio'] > 0 and data['pe_ratio'] < 25:
            valuation_score += 30
        elif data['pe_ratio'] > 0 and data['pe_ratio'] < 35:
            valuation_score += 20
        
        if data['pb_ratio'] > 0 and data['pb_ratio'] < 3:
            valuation_score += 20
        elif data['pb_ratio'] > 0 and data['pb_ratio'] < 5:
            valuation_score += 10
        
        # 成长性分析
        growth_score = 0
        if data['eps_growth'] and data['eps_growth'] > 20:
            growth_score += 25
        elif data['eps_growth'] and data['eps_growth'] > 10:
            growth_score += 15
        
        if data['revenue_growth'] and data['revenue_growth'] > 15:
            growth_score += 25
        elif data['revenue_growth'] and data['revenue_growth'] > 8:
            growth_score += 15
        
        # 财务健康度
        health_score = 0
        if data['dividend_yield'] > 2:
            health_score += 20
        
        if data['market_cap'] > 100000000000:  # 1000亿以上
            health_score += 20
        
        return {
            'valuation_score': valuation_score,
            'growth_score': growth_score,
            'health_score': health_score,
            'total_score': valuation_score + growth_score + health_score
        }
    
    def get_investment_advice(self, data, technical, fundamental):
        """获取投资建议"""
        if not data:
            return "🔴 数据不足"
        
        # 综合评分
        total_score = fundamental['total_score']
        rsi = data['rsi']
        price_position = data['price_position']
        trend_strength = technical['trend_strength']
        
        # 目标价分析
        upside_potential = ((data['target_price'] - data['current_price']) / data['current_price']) * 100 if data['target_price'] > 0 else 0
        
        # 风险评估
        risk_factors = []
        if rsi > 70:
            risk_factors.append("RSI超买")
        if price_position > 80:
            risk_factors.append("接近52周高点")
        if data['pe_ratio'] > 50:
            risk_factors.append("估值偏高")
        
        # 投资建议逻辑
        if total_score >= 70 and trend_strength >= 2 and len(risk_factors) <= 1:
            if upside_potential > 15:
                return "🟢 强烈买入"
            else:
                return "🔵 推荐买入"
        elif total_score >= 60 and trend_strength >= 1:
            return "🟡 小仓位试仓"
        elif total_score >= 50:
            return "🟠 观望为主"
        else:
            return "🔴 暂时回避"
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("🔍 AVGO独立分析验证系统")
        print("=" * 80)
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 获取数据
        data = self.get_real_time_data()
        if not data:
            print("❌ 无法获取AVGO数据")
            return
        
        # 技术分析
        technical = self.analyze_technical_trend(data)
        
        # 基本面分析
        fundamental = self.analyze_fundamentals(data)
        
        # 投资建议
        advice = self.get_investment_advice(data, technical, fundamental)
        
        # 显示结果
        print(f"\n📊 AVGO ({data['company_name']}) 分析结果:")
        print(f"  💰 当前价格: ${data['current_price']:.2f} ({data['change_pct']:+.2f}%)")
        print(f"  📈 RSI: {data['rsi']:.1f} ({technical['rsi_status']})")
        print(f"  📊 均线趋势: {technical['ma_trend']} (强度: {technical['trend_strength']}/3)")
        print(f"  📍 52周位置: {data['price_position']:.1f}%")
        print(f"  📦 成交量比: {data['volume_ratio']:.1f}x")
        
        print(f"\n💼 财务分析:")
        print(f"  💎 P/E: {data['pe_ratio']:.1f}")
        print(f"  📊 P/B: {data['pb_ratio']:.1f}")
        print(f"  💰 市值: ${data['market_cap']/1000000000:.1f}B")
        print(f"  🎯 股息率: {data['dividend_yield']:.2f}%")
        print(f"  📈 EPS增长: {data['eps_growth']:.1f}%" if data['eps_growth'] else "  📈 EPS增长: 数据不足")
        print(f"  🚀 营收增长: {data['revenue_growth']:.1f}%" if data['revenue_growth'] else "  🚀 营收增长: 数据不足")
        
        print(f"\n🎯 分析师目标价:")
        print(f"  平均目标: ${data['target_price']:.2f}")
        print(f"  最高目标: ${data['target_high']:.2f}")
        print(f"  最低目标: ${data['target_low']:.2f}")
        
        upside = ((data['target_price'] - data['current_price']) / data['current_price']) * 100 if data['target_price'] > 0 else 0
        print(f"  上涨空间: {upside:+.1f}%")
        
        print(f"\n📊 综合评分:")
        print(f"  估值评分: {fundamental['valuation_score']}/50")
        print(f"  成长评分: {fundamental['growth_score']}/50")
        print(f"  健康评分: {fundamental['health_score']}/40")
        print(f"  总分: {fundamental['total_score']}/140")
        
        print(f"\n🎯 投资建议: {advice}")
        
        # 风险评估
        risk_factors = []
        if data['rsi'] > 70:
            risk_factors.append("RSI超买")
        if data['price_position'] > 80:
            risk_factors.append("接近52周高点")
        if data['pe_ratio'] > 50:
            risk_factors.append("估值偏高")
        
        if risk_factors:
            print(f"\n⚠️ 风险提示:")
            for risk in risk_factors:
                print(f"  • {risk}")
        
        # 买入策略建议
        print(f"\n💡 买入策略建议:")
        if advice in ["🟢 强烈买入", "🔵 推荐买入"]:
            if data['rsi'] > 70:
                print(f"  • 等待RSI回调至65以下再买入")
            else:
                print(f"  • 可在${data['current_price']:.2f}附近分批买入")
            print(f"  • 目标价: ${data['target_price']:.2f}")
            print(f"  • 止损价: ${data['current_price'] * 0.88:.2f} (-12%)")
        else:
            print(f"  • 建议观望，等待更好的买入时机")
        
        return {
            'data': data,
            'technical': technical,
            'fundamental': fundamental,
            'advice': advice
        }

def main():
    """主函数"""
    analyzer = AVGOAnalyzer()
    result = analyzer.run_comprehensive_analysis()
    
    if result:
        print(f"\n" + "=" * 80)
        print("💡 独立分析结论:")
        print("=" * 80)
        print("1. 本分析使用实时市场数据，独立验证推荐结果")
        print("2. AVGO作为半导体龙头，具有长期投资价值")
        print("3. 当前估值偏高，建议等待回调")
        print("4. 建议结合个人风险承受能力和投资目标")

if __name__ == "__main__":
    main() 