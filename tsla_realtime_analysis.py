#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特斯拉实时分析系统
TSLA Real-time Analysis System
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TSLAAnalyzer:
    def __init__(self):
        self.symbol = 'TSLA'
        
    def get_realtime_data(self):
        """获取特斯拉实时数据"""
        print("🔍 获取特斯拉最新数据...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            
            # 获取历史数据
            hist = ticker.history(period='6mo', interval='1d')
            info = ticker.info
            
            if hist.empty:
                print("❌ 无法获取TSLA数据")
                return None
            
            # 最新价格数据
            latest_data = hist.iloc[-1]
            prev_data = hist.iloc[-2] if len(hist) > 1 else latest_data
            
            current_price = latest_data['Close']
            prev_close = prev_data['Close']
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # 技术指标
            rsi = self.calculate_rsi(hist['Close'])
            
            # 移动平均线
            ma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma_10 = hist['Close'].rolling(10).mean().iloc[-1]
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            # 成交量分析
            volume = latest_data['Volume']
            avg_volume_20 = hist['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # 52周高低点
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            position_52w = (current_price - low_52w) / (high_52w - low_52w) * 100
            
            # 近期表现
            perf_1w = self.calculate_performance(hist, 5)
            perf_1m = self.calculate_performance(hist, 20)
            perf_3m = self.calculate_performance(hist, 60)
            
            return {
                'price': current_price,
                'change': change,
                'change_pct': change_pct,
                'prev_close': prev_close,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'ma_5': ma_5,
                'ma_10': ma_10,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'position_52w': position_52w,
                'perf_1w': perf_1w,
                'perf_1m': perf_1m,
                'perf_3m': perf_3m,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'hist_data': hist
            }
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        try:
            if len(prices) < period + 1:
                return 50
            
            delta = prices.diff().dropna()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.rolling(window=period, min_periods=period).mean()
            avg_loss = losses.rolling(window=period, min_periods=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def calculate_performance(self, hist, days):
        """计算指定天数的表现"""
        try:
            if len(hist) < days + 1:
                return 0
            
            current = hist['Close'].iloc[-1]
            past = hist['Close'].iloc[-(days+1)]
            return ((current - past) / past) * 100
        except:
            return 0
    
    def analyze_technical_signals(self, data):
        """技术面分析"""
        print("\n📈 === 技术面分析 ===")
        
        price = data['price']
        rsi = data['rsi']
        ma_5 = data['ma_5']
        ma_10 = data['ma_10']
        ma_20 = data['ma_20']
        ma_50 = data['ma_50']
        
        signals = []
        score = 0
        
        # RSI分析
        if rsi < 30:
            signals.append("✅ RSI超卖，存在反弹机会")
            score += 2
        elif rsi < 40:
            signals.append("⚠️ RSI偏低，技术面偏弱")
            score += 1
        elif rsi > 70:
            signals.append("❌ RSI超买，回调风险")
            score -= 2
        elif rsi > 60:
            signals.append("⚠️ RSI偏高，注意风险")
            score -= 1
        else:
            signals.append("🔄 RSI中性区间")
        
        # 均线分析
        if price > ma_5 > ma_10 > ma_20:
            signals.append("✅ 短期多头排列")
            score += 2
        elif price < ma_5 < ma_10 < ma_20:
            signals.append("❌ 短期空头排列")
            score -= 2
        
        if price > ma_50:
            signals.append("✅ 站稳中长期均线")
            score += 1
        else:
            signals.append("❌ 跌破中长期均线")
            score -= 1
        
        # 成交量分析
        volume_ratio = data['volume_ratio']
        if volume_ratio > 1.5:
            signals.append("✅ 成交量放大")
            score += 1
        elif volume_ratio < 0.8:
            signals.append("⚠️ 成交量萎缩")
            score -= 1
        
        print(f"💰 当前价格: ${price:.2f}")
        print(f"📊 RSI指标: {rsi:.1f}")
        print(f"📈 均线状态:")
        print(f"   MA5:  ${ma_5:.2f}")
        print(f"   MA10: ${ma_10:.2f}")
        print(f"   MA20: ${ma_20:.2f}")
        print(f"   MA50: ${ma_50:.2f}")
        print(f"📊 成交量比率: {volume_ratio:.2f}x")
        
        print(f"\n🎯 技术信号:")
        for signal in signals:
            print(f"   {signal}")
        
        # 综合评分
        if score >= 4:
            rating = "强烈买入"
        elif score >= 2:
            rating = "买入"
        elif score <= -4:
            rating = "强烈卖出"
        elif score <= -2:
            rating = "卖出"
        else:
            rating = "中性观望"
        
        print(f"\n⭐ 技术面综合评级: {rating} (评分: {score:+d})")
        
        return score, rating
    
    def analyze_price_action(self, data):
        """价格行为分析"""
        print("\n💹 === 价格行为分析 ===")
        
        current_price = data['price']
        change_pct = data['change_pct']
        position_52w = data['position_52w']
        high_52w = data['high_52w']
        low_52w = data['low_52w']
        
        print(f"📊 当前价格: ${current_price:.2f} ({change_pct:+.2f}%)")
        print(f"📊 52周区间: ${low_52w:.2f} - ${high_52w:.2f}")
        print(f"📊 52周位置: {position_52w:.1f}%")
        
        # 关键价位分析
        key_levels = []
        
        # 支撑位分析
        if position_52w < 30:
            key_levels.append("🟢 接近52周低点，强支撑区域")
        elif position_52w < 50:
            key_levels.append("🟡 处于52周中下位置")
        elif position_52w > 80:
            key_levels.append("🔴 接近52周高点，阻力强劲")
        else:
            key_levels.append("🟡 处于52周中等位置")
        
        # 整数关口分析
        if current_price < 300:
            key_levels.append("⚠️ 跌破300重要心理关口")
        elif current_price < 350:
            key_levels.append("⚠️ 在300-350区间震荡")
        elif current_price < 400:
            key_levels.append("🔄 在350-400区间运行")
        
        print(f"\n🎯 关键价位分析:")
        for level in key_levels:
            print(f"   {level}")
        
        # 近期表现
        perf_1w = data['perf_1w']
        perf_1m = data['perf_1m']
        perf_3m = data['perf_3m']
        
        print(f"\n📈 近期表现:")
        print(f"   1周: {perf_1w:+.2f}%")
        print(f"   1月: {perf_1m:+.2f}%")
        print(f"   3月: {perf_3m:+.2f}%")
    
    def generate_trading_strategy(self, data, tech_score):
        """生成交易策略"""
        print("\n🎯 === 交易策略建议 ===")
        
        price = data['price']
        rsi = data['rsi']
        position_52w = data['position_52w']
        change_pct = data['change_pct']
        
        strategies = []
        
        # 基于技术面评分的策略
        if tech_score >= 2 and rsi < 50:
            strategies.append("📈 技术面转好，可考虑分批建仓")
            strategies.append(f"   建议入场价位: ${price * 0.98:.2f} - ${price * 1.02:.2f}")
            strategies.append(f"   止损价位: ${price * 0.95:.2f}")
            strategies.append(f"   目标价位: ${price * 1.10:.2f} - ${price * 1.15:.2f}")
        
        elif tech_score <= -2:
            strategies.append("📉 技术面偏弱，建议观望或减仓")
            strategies.append(f"   反弹减仓价位: ${price * 1.03:.2f} - ${price * 1.05:.2f}")
            strategies.append(f"   关注支撑价位: ${price * 0.95:.2f}")
        
        else:
            strategies.append("🔄 技术面中性，建议区间操作")
            strategies.append(f"   低买价位: ${price * 0.97:.2f}")
            strategies.append(f"   高卖价位: ${price * 1.05:.2f}")
        
        # 基于RSI的具体建议
        if rsi < 30:
            strategies.append("💡 RSI超卖，适合逢低布局")
        elif rsi > 70:
            strategies.append("⚠️ RSI超买，注意减仓保护")
        
        # 基于位置的建议
        if position_52w < 25:
            strategies.append("🎯 接近年度低点，中长期价值显现")
        elif position_52w > 75:
            strategies.append("⚠️ 接近年度高点，谨慎追高")
        
        print(f"💼 策略建议:")
        for strategy in strategies:
            print(f"   {strategy}")
        
        # 风险提醒
        print(f"\n⚠️ 风险提醒:")
        print(f"   • 特斯拉波动性较大，注意仓位控制")
        print(f"   • 关注电动车行业政策变化")
        print(f"   • 留意马斯克相关新闻影响")
        print(f"   • 季度交付数据是重要催化剂")
    
    def analyze_market_context(self):
        """市场环境分析"""
        print("\n🌍 === 市场环境分析 ===")
        
        try:
            # 获取相关指数和股票
            symbols = ['^IXIC', '^GSPC', '^VIX', 'QQQ', 'NIO', 'RIVN', 'LCID']
            market_data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change_pct = ((current - prev) / prev) * 100
                    market_data[symbol] = {'price': current, 'change_pct': change_pct}
            
            # 市场指数
            if '^IXIC' in market_data:
                nasdaq = market_data['^IXIC']
                print(f"📊 纳斯达克: {nasdaq['price']:.2f} ({nasdaq['change_pct']:+.2f}%)")
            
            if '^VIX' in market_data:
                vix = market_data['^VIX']
                print(f"😰 VIX恐慌指数: {vix['price']:.2f} ({vix['change_pct']:+.2f}%)")
            
            # 同行业对比
            print(f"\n🚗 电动车板块对比:")
            ev_stocks = ['NIO', 'RIVN', 'LCID']
            for symbol in ev_stocks:
                if symbol in market_data:
                    stock = market_data[symbol]
                    print(f"   {symbol}: {stock['change_pct']:+.2f}%")
            
        except Exception as e:
            print(f"⚠️ 市场数据获取部分失败: {e}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚗 特斯拉实时分析启动...")
        print("=" * 50)
        
        # 获取数据
        data = self.get_realtime_data()
        if not data:
            return
        
        print(f"✅ 数据获取成功 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行各项分析
        self.analyze_market_context()
        tech_score, tech_rating = self.analyze_technical_signals(data)
        self.analyze_price_action(data)
        self.generate_trading_strategy(data, tech_score)
        
        print("\n" + "=" * 50)
        print("✅ 特斯拉分析完成!")

def main():
    analyzer = TSLAAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 