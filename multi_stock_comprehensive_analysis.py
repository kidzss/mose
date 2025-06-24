#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多股票综合分析系统
深度分析GS、V、MA、WMT、COST、JNJ、ABT、ABBV
技术面+基本面+估值+买入时机综合评估
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class MultiStockAnalyzer:
    """多股票综合分析器"""
    
    def __init__(self):
        # 分析标的
        self.target_stocks = {
            'GS': {'sector': '金融', 'category': '投资银行', 'weight_target': '5%'},
            'V': {'sector': '金融科技', 'category': '支付网络', 'weight_target': '4%'},
            'MA': {'sector': '金融科技', 'category': '支付网络', 'weight_target': '3%'},
            'WMT': {'sector': '消费', 'category': '零售巨头', 'weight_target': '3%'},
            'COST': {'sector': '消费', 'category': '会员制零售', 'weight_target': '3%'},
            'JNJ': {'sector': '医疗', 'category': '医疗器械+制药', 'weight_target': '6%'},
            'ABT': {'sector': '医疗', 'category': '医疗器械', 'weight_target': '4%'},
            'ABBV': {'sector': '医疗', 'category': '生物制药', 'weight_target': '4%'}
        }
        
        self.portfolio_value = 27533.17
        self.analysis_results = {}
        
    def get_comprehensive_data(self, symbol):
        """获取单只股票的全面数据"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 历史价格数据
            hist_5y = ticker.history(period='5y')
            hist_3y = ticker.history(period='3y')
            hist_1y = ticker.history(period='1y')
            hist_6m = ticker.history(period='6mo')
            hist_3m = ticker.history(period='3mo')
            
            # 基本面数据
            info = ticker.info
            
            if hist_1y.empty:
                return None
            
            return {
                'hist_5y': hist_5y,
                'hist_3y': hist_3y,
                'hist_1y': hist_1y,
                'hist_6m': hist_6m,
                'hist_3m': hist_3m,
                'info': info
            }
            
        except Exception as e:
            print(f"获取{symbol}数据失败: {e}")
            return None
    
    def calculate_technical_indicators(self, hist_data):
        """计算技术指标"""
        close_prices = hist_data['Close']
        
        # 移动平均线
        ma_5 = close_prices.rolling(5).mean().iloc[-1]
        ma_10 = close_prices.rolling(10).mean().iloc[-1]
        ma_20 = close_prices.rolling(20).mean().iloc[-1]
        ma_50 = close_prices.rolling(50).mean().iloc[-1]
        ma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else close_prices.iloc[-1]
        
        current_price = close_prices.iloc[-1]
        
        # RSI
        rsi = self.calculate_rsi(close_prices).iloc[-1]
        
        # MACD
        macd_line, macd_signal, macd_histogram = self.calculate_macd(close_prices)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100
        
        # 价格位置
        high_52w = hist_data['High'].max()
        low_52w = hist_data['Low'].min()
        price_position_52w = (current_price - low_52w) / (high_52w - low_52w) * 100
        
        # 成交量分析
        avg_volume_20 = hist_data['Volume'].rolling(20).mean().iloc[-1]
        recent_volume = hist_data['Volume'].iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume_20
        
        return {
            'current_price': float(current_price),
            'ma_5': float(ma_5),
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'ma_200': float(ma_200),
            'rsi': float(rsi),
            'macd_line': float(macd_line.iloc[-1]),
            'macd_signal': float(macd_signal.iloc[-1]),
            'macd_histogram': float(macd_histogram.iloc[-1]),
            'bb_position': float(bb_position),
            'bb_upper': float(bb_upper.iloc[-1]),
            'bb_lower': float(bb_lower.iloc[-1]),
            'high_52w': float(high_52w),
            'low_52w': float(low_52w),
            'price_position_52w': float(price_position_52w),
            'volume_ratio': float(volume_ratio)
        }
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band
    
    def analyze_fundamentals(self, info, symbol):
        """基本面分析"""
        try:
            # 基本财务指标
            market_cap = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0
            pe_ratio = info.get('trailingPE', 0) if info.get('trailingPE') else 0
            pb_ratio = info.get('priceToBook', 0) if info.get('priceToBook') else 0
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            debt_to_equity = info.get('debtToEquity', 0) if info.get('debtToEquity') else 0
            
            # 成长性指标
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            
            # 行业特定评分标准
            sector = self.target_stocks[symbol]['sector']
            
            # PE评分
            if sector == '金融':
                pe_score = 5 if 8 <= pe_ratio <= 15 else (4 if 15 < pe_ratio <= 18 else 3)
            elif sector == '金融科技':
                pe_score = 5 if 15 <= pe_ratio <= 25 else (4 if 25 < pe_ratio <= 30 else 3)
            elif sector == '消费':
                pe_score = 5 if 15 <= pe_ratio <= 25 else (4 if 25 < pe_ratio <= 30 else 3)
            elif sector == '医疗':
                pe_score = 5 if 12 <= pe_ratio <= 20 else (4 if 20 < pe_ratio <= 25 else 3)
            else:
                pe_score = 3
            
            # ROE评分
            roe_score = 5 if roe >= 15 else (4 if roe >= 12 else (3 if roe >= 8 else 2))
            
            # 股息率评分
            div_score = 5 if dividend_yield >= 3 else (4 if dividend_yield >= 2 else (3 if dividend_yield >= 1 else 2))
            
            # 成长性评分
            growth_score = 5 if revenue_growth >= 10 else (4 if revenue_growth >= 5 else (3 if revenue_growth >= 0 else 2))
            
            return {
                'market_cap': float(market_cap),
                'pe_ratio': float(pe_ratio),
                'pb_ratio': float(pb_ratio),
                'roe': float(roe),
                'dividend_yield': float(dividend_yield),
                'debt_to_equity': float(debt_to_equity),
                'revenue_growth': float(revenue_growth),
                'earnings_growth': float(earnings_growth),
                'pe_score': pe_score,
                'roe_score': roe_score,
                'div_score': div_score,
                'growth_score': growth_score,
                'fundamental_score': (pe_score + roe_score + div_score + growth_score) / 4
            }
            
        except Exception as e:
            print(f"分析{symbol}基本面时出错: {e}")
            return None
    
    def calculate_technical_score(self, tech_indicators):
        """计算技术面评分"""
        scores = []
        
        # 趋势评分
        current = tech_indicators['current_price']
        ma_20 = tech_indicators['ma_20']
        ma_50 = tech_indicators['ma_50']
        ma_200 = tech_indicators['ma_200']
        
        if current > ma_20 > ma_50 > ma_200:
            trend_score = 5
        elif current > ma_20 > ma_50:
            trend_score = 4
        elif current > ma_50:
            trend_score = 3
        else:
            trend_score = 2
        scores.append(trend_score)
        
        # RSI评分
        rsi = tech_indicators['rsi']
        if 40 <= rsi <= 60:
            rsi_score = 5
        elif 30 <= rsi <= 70:
            rsi_score = 4
        elif rsi < 30:
            rsi_score = 3  # 超卖
        else:
            rsi_score = 2  # 超买
        scores.append(rsi_score)
        
        # MACD评分
        macd_line = tech_indicators['macd_line']
        macd_signal = tech_indicators['macd_signal']
        macd_hist = tech_indicators['macd_histogram']
        
        if macd_line > macd_signal and macd_hist > 0:
            macd_score = 4
        elif macd_line > macd_signal:
            macd_score = 3
        else:
            macd_score = 2
        scores.append(macd_score)
        
        # 价格位置评分
        price_pos = tech_indicators['price_position_52w']
        if price_pos < 30:
            position_score = 5  # 低位
        elif price_pos < 50:
            position_score = 4
        elif price_pos < 70:
            position_score = 3
        else:
            position_score = 2  # 高位
        scores.append(position_score)
        
        return sum(scores) / len(scores)
    
    def determine_buy_timing(self, symbol, tech_indicators, fundamentals, tech_score):
        """确定买入时机"""
        current_price = tech_indicators['current_price']
        rsi = tech_indicators['rsi']
        price_position = tech_indicators['price_position_52w']
        bb_position = tech_indicators['bb_position']
        
        # 综合评分
        overall_score = (tech_score * 0.4 + fundamentals['fundamental_score'] * 0.6)
        
        # 买入时机判断
        buy_signals = []
        wait_signals = []
        
        # 价格位置
        if price_position < 40:
            buy_signals.append("价格相对低位")
        elif price_position > 80:
            wait_signals.append("价格相对高位")
        
        # RSI
        if rsi < 50:
            buy_signals.append("RSI健康")
        elif rsi > 70:
            wait_signals.append("RSI超买")
        
        # 布林带位置
        if bb_position < 30:
            buy_signals.append("接近布林带下轨")
        elif bb_position > 80:
            wait_signals.append("接近布林带上轨")
        
        # 基本面
        if fundamentals['fundamental_score'] >= 4:
            buy_signals.append("基本面优秀")
        elif fundamentals['fundamental_score'] >= 3.5:
            buy_signals.append("基本面良好")
        
        # 确定买入建议
        if overall_score >= 4 and len(buy_signals) >= 3:
            recommendation = "立即买入"
            urgency = "高"
        elif overall_score >= 3.5 and len(buy_signals) >= 2:
            recommendation = "分批买入"
            urgency = "中"
        elif overall_score >= 3 and len(wait_signals) <= 1:
            recommendation = "小仓位试探"
            urgency = "低"
        else:
            recommendation = "继续等待"
            urgency = "无"
        
        # 计算目标价位
        support_levels = [
            current_price * 0.97,  # 3%回调
            current_price * 0.95,  # 5%回调
            current_price * 0.92   # 8%回调
        ]
        
        target_levels = [
            current_price * 1.15,  # 15%目标
            current_price * 1.25   # 25%目标
        ]
        
        return {
            'overall_score': float(overall_score),
            'recommendation': recommendation,
            'urgency': urgency,
            'buy_signals': buy_signals,
            'wait_signals': wait_signals,
            'support_levels': [float(x) for x in support_levels],
            'target_levels': [float(x) for x in target_levels],
            'stop_loss': float(current_price * 0.88)  # 12%止损
        }
    
    def analyze_single_stock(self, symbol):
        """分析单只股票"""
        print(f"\n📊 {symbol} 综合分析")
        print("=" * 60)
        
        # 获取数据
        data = self.get_comprehensive_data(symbol)
        if not data:
            return None
        
        # 技术分析
        tech_indicators = self.calculate_technical_indicators(data['hist_1y'])
        
        # 基本面分析
        fundamentals = self.analyze_fundamentals(data['info'], symbol)
        if not fundamentals:
            return None
        
        # 技术面评分
        tech_score = self.calculate_technical_score(tech_indicators)
        
        # 买入时机分析
        buy_timing = self.determine_buy_timing(symbol, tech_indicators, fundamentals, tech_score)
        
        # 显示分析结果
        stock_info = self.target_stocks[symbol]
        current_price = tech_indicators['current_price']
        
        print(f"🏢 {stock_info['category']} ({stock_info['sector']})")
        print(f"💰 当前价格: ${current_price:.2f}")
        print(f"📈 52周位置: {tech_indicators['price_position_52w']:.1f}%")
        print(f"📊 RSI: {tech_indicators['rsi']:.1f}")
        
        print(f"\n📊 基本面指标:")
        print(f"   市值: ${fundamentals['market_cap']:.0f}B")
        print(f"   PE: {fundamentals['pe_ratio']:.1f}")
        print(f"   ROE: {fundamentals['roe']:.1f}%")
        print(f"   股息率: {fundamentals['dividend_yield']:.2f}%")
        print(f"   收入增长: {fundamentals['revenue_growth']:.1f}%")
        
        print(f"\n🎯 综合评估:")
        print(f"   技术面评分: {tech_score:.1f}/5")
        print(f"   基本面评分: {fundamentals['fundamental_score']:.1f}/5")
        print(f"   综合评分: {buy_timing['overall_score']:.1f}/5")
        
        print(f"\n💡 投资建议: {buy_timing['recommendation']} (紧急度: {buy_timing['urgency']})")
        
        if buy_timing['buy_signals']:
            print(f"✅ 买入信号: {', '.join(buy_timing['buy_signals'])}")
        
        if buy_timing['wait_signals']:
            print(f"⏳ 等待信号: {', '.join(buy_timing['wait_signals'])}")
        
        print(f"\n💰 价位建议:")
        print(f"   支撑位: ${buy_timing['support_levels'][0]:.2f} / ${buy_timing['support_levels'][1]:.2f} / ${buy_timing['support_levels'][2]:.2f}")
        print(f"   目标位: ${buy_timing['target_levels'][0]:.2f} / ${buy_timing['target_levels'][1]:.2f}")
        print(f"   止损位: ${buy_timing['stop_loss']:.2f}")
        
        # 计算建议仓位
        target_weight = float(stock_info['weight_target'].rstrip('%')) / 100
        target_amount = self.portfolio_value * target_weight
        
        print(f"\n💼 仓位建议:")
        print(f"   目标权重: {stock_info['weight_target']}")
        print(f"   目标金额: ${target_amount:.0f}")
        print(f"   建议股数: {target_amount / current_price:.0f}股")
        
        # 整合结果
        result = {
            'symbol': symbol,
            'sector': stock_info['sector'],
            'category': stock_info['category'],
            'technical_indicators': tech_indicators,
            'fundamentals': fundamentals,
            'tech_score': tech_score,
            'buy_timing': buy_timing,
            'target_weight': target_weight,
            'target_amount': target_amount,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return result
    
    def rank_stocks_by_attractiveness(self, results):
        """按投资吸引力排序"""
        print(f"\n🏆 投资吸引力排名")
        print("=" * 80)
        
        # 按综合评分排序
        sorted_stocks = sorted(results.items(), 
                             key=lambda x: x[1]['buy_timing']['overall_score'], 
                             reverse=True)
        
        print(f"{'排名':<4} {'股票':<6} {'综合评分':<8} {'投资建议':<12} {'紧急度':<6} {'52周位置':<8} {'RSI':<6}")
        print("-" * 80)
        
        for i, (symbol, data) in enumerate(sorted_stocks, 1):
            score = data['buy_timing']['overall_score']
            recommendation = data['buy_timing']['recommendation']
            urgency = data['buy_timing']['urgency']
            price_pos = data['technical_indicators']['price_position_52w']
            rsi = data['technical_indicators']['rsi']
            
            print(f"{i:<4} {symbol:<6} {score:<8.1f} {recommendation:<12} {urgency:<6} {price_pos:<8.1f}% {rsi:<6.1f}")
        
        return sorted_stocks
    
    def generate_buying_plan(self, sorted_stocks):
        """生成买入计划"""
        print(f"\n📋 综合买入计划")
        print("=" * 80)
        
        immediate_buys = []
        batch_buys = []
        wait_list = []
        
        for symbol, data in sorted_stocks:
            recommendation = data['buy_timing']['recommendation']
            
            if recommendation == "立即买入":
                immediate_buys.append((symbol, data))
            elif recommendation in ["分批买入", "小仓位试探"]:
                batch_buys.append((symbol, data))
            else:
                wait_list.append((symbol, data))
        
        # 立即买入
        if immediate_buys:
            print(f"🚀 立即买入 ({len(immediate_buys)}只):")
            total_immediate = 0
            for symbol, data in immediate_buys:
                amount = data['target_amount']
                total_immediate += amount
                print(f"   {symbol}: ${amount:.0f} ({data['target_weight']*100:.0f}%)")
            print(f"   小计: ${total_immediate:.0f}")
        
        # 分批买入
        if batch_buys:
            print(f"\n📊 分批买入 ({len(batch_buys)}只):")
            total_batch = 0
            for symbol, data in batch_buys:
                amount = data['target_amount']
                total_batch += amount
                current_price = data['technical_indicators']['current_price']
                support = data['buy_timing']['support_levels'][1]  # 5%回调位
                print(f"   {symbol}: ${amount:.0f} (当前${current_price:.2f}, 等待${support:.2f})")
            print(f"   小计: ${total_batch:.0f}")
        
        # 等待列表
        if wait_list:
            print(f"\n⏳ 继续等待 ({len(wait_list)}只):")
            for symbol, data in wait_list:
                reason = ', '.join(data['buy_timing']['wait_signals'][:2])
                print(f"   {symbol}: {reason}")
        
        # 总投资计划
        total_investment = sum(data['target_amount'] for _, data in sorted_stocks)
        print(f"\n💰 总投资计划:")
        print(f"   目标总投资: ${total_investment:.0f}")
        print(f"   占总资产比例: {total_investment/self.portfolio_value*100:.1f}%")
        
        return {
            'immediate_buys': immediate_buys,
            'batch_buys': batch_buys,
            'wait_list': wait_list,
            'total_investment': total_investment
        }
    
    def save_analysis_results(self, results, buying_plan):
        """保存分析结果"""
        analysis_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': list(results.keys()),
            'individual_analysis': results,
            'buying_plan': buying_plan,
            'portfolio_value': self.portfolio_value
        }
        
        filename = f"multi_stock_analysis_{datetime.now().strftime('%Y%m%d')}.json"
        
        def convert_numpy_types(obj):
            """转换numpy类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        converted_data = convert_numpy_types(analysis_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 分析结果已保存到: {filename}")
        
        return filename
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("📊 多股票综合投资分析")
        print("=" * 80)
        print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"分析标的: {', '.join(self.target_stocks.keys())}")
        print("=" * 80)
        
        results = {}
        
        # 分析每只股票
        for symbol in self.target_stocks.keys():
            result = self.analyze_single_stock(symbol)
            if result:
                results[symbol] = result
        
        if not results:
            print("❌ 没有获取到有效的分析结果")
            return
        
        # 排序和排名
        sorted_stocks = self.rank_stocks_by_attractiveness(results)
        
        # 生成买入计划
        buying_plan = self.generate_buying_plan(sorted_stocks)
        
        # 保存结果
        filename = self.save_analysis_results(results, buying_plan)
        
        print(f"\n🎊 综合分析完成！")
        print(f"已分析{len(results)}只股票，生成完整投资建议。")
        
        return results, buying_plan, filename

if __name__ == "__main__":
    analyzer = MultiStockAnalyzer()
    analyzer.run_comprehensive_analysis() 