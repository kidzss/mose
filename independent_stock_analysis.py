#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立股票分析验证系统
Independent Stock Analysis Verification System
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IndependentStockAnalyzer:
    def __init__(self):
        self.recommended_stocks = [
            'NEM', 'PDD', 'GOOG', 'CF', 'TER', 'ASML', 'MPWR', 'GOOGL', 'LRCX', 'REGN'
        ]
        
    def get_real_time_data(self, symbol: str) -> dict:
        """获取实时股票数据"""
        try:
            ticker = yf.Ticker(symbol)
            
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
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'change': change,
                'change_pct': change_pct,
                'rsi': rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'price_position': price_position,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'market_cap': market_cap,
                'dividend_yield': dividend_yield,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', '')
            }
            
        except Exception as e:
            print(f"获取{symbol}数据失败: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def calculate_quality_score(self, data):
        """计算质量因子"""
        if not data:
            return 0
        
        score = 0
        
        # 财务健康度 (40%)
        if data['pe_ratio'] > 0 and data['pe_ratio'] < 25:
            score += 20
        elif data['pe_ratio'] > 0 and data['pe_ratio'] < 35:
            score += 10
        
        if data['pb_ratio'] > 0 and data['pb_ratio'] < 3:
            score += 10
        elif data['pb_ratio'] > 0 and data['pb_ratio'] < 5:
            score += 5
        
        if data['dividend_yield'] > 2:
            score += 10
        
        # 技术面 (30%)
        if 30 <= data['rsi'] <= 70:
            score += 15
        elif 20 <= data['rsi'] <= 80:
            score += 10
        
        if data['current_price'] > data['ma_20']:
            score += 10
        elif data['current_price'] > data['ma_50']:
            score += 5
        
        if data['price_position'] > 20 and data['price_position'] < 80:
            score += 5
        
        # 市场表现 (30%)
        if data['change_pct'] > -5:
            score += 10
        
        if data['volume_ratio'] > 0.8:
            score += 10
        
        if data['market_cap'] > 10000000000:  # 100亿以上
            score += 10
        
        return min(score, 100)
    
    def calculate_enhanced_score(self, data):
        """计算增强评分"""
        if not data:
            return 0
        
        base_score = self.calculate_quality_score(data)
        
        # 额外加分项
        bonus = 0
        
        # 强势股票加分
        if data['current_price'] > data['ma_20'] > data['ma_50']:
            bonus += 10
        
        # 低估值加分
        if data['pe_ratio'] > 0 and data['pe_ratio'] < 15:
            bonus += 5
        
        # 高股息加分
        if data['dividend_yield'] > 3:
            bonus += 5
        
        # 成交量活跃加分
        if data['volume_ratio'] > 1.2:
            bonus += 5
        
        return min(base_score + bonus, 100)
    
    def get_investment_advice(self, data):
        """获取投资建议"""
        if not data:
            return "🔴 数据不足"
        
        enhanced_score = self.calculate_enhanced_score(data)
        rsi = data['rsi']
        price_position = data['price_position']
        
        # 基于综合评分和建议
        if enhanced_score >= 80:
            if rsi < 70 and price_position < 80:
                return "🟢 强烈推荐"
            else:
                return "🔵 推荐买入"
        elif enhanced_score >= 70:
            return "🟡 小仓位试仓"
        elif enhanced_score >= 60:
            return "🟠 观望为主"
        else:
            return "🔴 暂时回避"
    
    def analyze_all_stocks(self):
        """分析所有推荐股票"""
        print("🔍 独立股票分析验证系统")
        print("=" * 80)
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        results = []
        
        for symbol in self.recommended_stocks:
            print(f"\n📊 分析 {symbol}...")
            
            data = self.get_real_time_data(symbol)
            if not data:
                print(f"  ❌ 无法获取{symbol}数据")
                continue
            
            # 计算评分
            quality_score = self.calculate_quality_score(data)
            enhanced_score = self.calculate_enhanced_score(data)
            
            # 获取投资建议
            advice = self.get_investment_advice(data)
            
            # 计算夏普比率（简化版）
            returns = pd.Series([data['change_pct']])
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            result = {
                'symbol': symbol,
                'company_name': data['company_name'],
                'current_price': data['current_price'],
                'change_pct': data['change_pct'],
                'quality_score': quality_score,
                'enhanced_score': enhanced_score,
                'rsi': data['rsi'],
                'sharpe_ratio': sharpe_ratio,
                'pe_ratio': data['pe_ratio'],
                'pb_ratio': data['pb_ratio'],
                'dividend_yield': data['dividend_yield'],
                'volume_ratio': data['volume_ratio'],
                'price_position': data['price_position'],
                'advice': advice,
                'sector': data['sector']
            }
            
            results.append(result)
            
            # 显示结果
            print(f"  💰 当前价格: ${data['current_price']:.2f} ({data['change_pct']:+.2f}%)")
            print(f"  📊 质量评分: {quality_score:.1f}")
            print(f"  🚀 增强评分: {enhanced_score:.1f}")
            print(f"  📈 RSI: {data['rsi']:.1f}")
            print(f"  💎 P/E: {data['pe_ratio']:.1f}")
            print(f"  📦 成交量比: {data['volume_ratio']:.1f}x")
            print(f"  🎯 投资建议: {advice}")
        
        return results
    
    def generate_comparison_report(self, results):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("📋 独立分析结果对比")
        print("=" * 80)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 按增强评分排序
        df_sorted = df.sort_values('enhanced_score', ascending=False)
        
        print("\n🏆 按增强评分排序:")
        print("-" * 80)
        for _, row in df_sorted.iterrows():
            print(f"{row['symbol']:6} | 评分: {row['enhanced_score']:5.1f} | "
                  f"价格: ${row['current_price']:6.2f} | "
                  f"RSI: {row['rsi']:4.1f} | "
                  f"P/E: {row['pe_ratio']:5.1f} | "
                  f"{row['advice']}")
        
        # 统计信息
        print("\n📊 统计信息:")
        print("-" * 40)
        print(f"平均增强评分: {df['enhanced_score'].mean():.1f}")
        print(f"平均RSI: {df['rsi'].mean():.1f}")
        print(f"平均P/E: {df['pe_ratio'].mean():.1f}")
        
        # 建议分布
        advice_counts = df['advice'].value_counts()
        print(f"\n🎯 建议分布:")
        for advice, count in advice_counts.items():
            print(f"  {advice}: {count}只")
        
        # 行业分布
        sector_counts = df['sector'].value_counts()
        print(f"\n🏭 行业分布:")
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count}只")
        
        return df_sorted

def main():
    """主函数"""
    analyzer = IndependentStockAnalyzer()
    
    # 分析所有股票
    results = analyzer.analyze_all_stocks()
    
    # 生成对比报告
    if results:
        df_sorted = analyzer.generate_comparison_report(results)
        
        print("\n" + "=" * 80)
        print("💡 独立分析结论:")
        print("=" * 80)
        print("1. 本分析使用实时市场数据，独立验证推荐结果")
        print("2. 评分基于财务健康度、技术面和市场表现")
        print("3. 建议仅供参考，投资需谨慎")
        print("4. 建议结合个人风险承受能力和投资目标")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"independent_analysis_{timestamp}.csv"
        df_sorted.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 分析结果已保存至: {filename}")

if __name__ == "__main__":
    main() 