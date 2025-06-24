import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class DefensiveStocksAnalysis:
    """防御性和价值股票分析"""
    
    def __init__(self):
        self.symbols = ['COST', 'ABT', 'MRK', 'GS']
        self.stock_info = {
            'COST': {'name': 'Costco', 'sector': '零售', 'type': '防御消费'},
            'ABT': {'name': 'Abbott Labs', 'sector': '医疗', 'type': '医疗器械'},
            'MRK': {'name': 'Merck', 'sector': '医药', 'type': '制药巨头'},
            'GS': {'name': 'Goldman Sachs', 'sector': '金融', 'type': '投资银行'}
        }
        
    def get_comprehensive_data(self, symbol):
        """获取全面的股票数据"""
        try:
            stock = yf.Ticker(symbol)
            
            # 获取不同时间周期的数据
            data_3y = stock.history(period="3y")
            data_1y = stock.history(period="1y")
            data_6m = stock.history(period="6mo")
            data_3m = stock.history(period="3mo")
            
            # 获取基本面信息
            info = stock.info
            
            return stock, data_3y, data_1y, data_6m, data_3m, info
            
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            return None, None, None, None, None, None
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        if data is None or len(data) == 0:
            return {}
        
        current_price = data['Close'].iloc[-1]
        
        # 移动平均线
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma50 = data['Close'].rolling(50).mean().iloc[-1]
        ma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else None
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 价格位置
        high_1y = data['High'].max()
        low_1y = data['Low'].min()
        price_position = (current_price - low_1y) / (high_1y - low_1y)
        
        # 从高点回调幅度
        drawdown = (current_price - high_1y) / high_1y
        
        # 成交量分析
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # 波动率
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        return {
            'current_price': current_price,
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'rsi': rsi,
            'price_position': price_position,
            'drawdown': drawdown,
            'high_1y': high_1y,
            'low_1y': low_1y,
            'volume_ratio': volume_ratio,
            'volatility': volatility
        }
    
    def analyze_fundamentals(self, info):
        """分析基本面"""
        if not info:
            return {}
        
        try:
            return {
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'pb_ratio': info.get('priceToBook', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 'N/A',
                'roe': info.get('returnOnEquity', 'N/A'),
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'profit_margin': info.get('profitMargins', 'N/A'),
                'revenue_growth': info.get('revenueGrowth', 'N/A'),
                'market_cap': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 'N/A'
            }
        except:
            return {}
    
    def calculate_entry_signals(self, tech_data, fund_data, symbol):
        """计算入场信号"""
        signals = []
        score = 0
        
        # 技术面评分
        if tech_data.get('rsi', 50) < 30:
            score += 3
            signals.append("RSI超卖(+3)")
        elif tech_data.get('rsi', 50) < 40:
            score += 2
            signals.append("RSI偏低(+2)")
        elif tech_data.get('rsi', 50) < 50:
            score += 1
            signals.append("RSI健康(+1)")
        
        # 价格位置评分
        pos = tech_data.get('price_position', 0.5)
        if pos < 0.3:
            score += 2
            signals.append("价格低位(+2)")
        elif pos < 0.5:
            score += 1
            signals.append("价格合理(+1)")
        
        # 回调幅度评分
        drawdown = tech_data.get('drawdown', 0)
        if drawdown < -0.15:
            score += 2
            signals.append("大幅回调(+2)")
        elif drawdown < -0.10:
            score += 1
            signals.append("适度回调(+1)")
        
        # 趋势评分
        current = tech_data.get('current_price', 0)
        ma50 = tech_data.get('ma50', 0)
        ma200 = tech_data.get('ma200', 0)
        
        if ma200 and current > ma200:
            score += 1
            signals.append("长期趋势向上(+1)")
        
        # 估值评分（针对不同类型股票）
        pe = fund_data.get('pe_ratio', 'N/A')
        if isinstance(pe, (int, float)):
            if symbol in ['COST', 'ABT']:  # 成长性防御股
                if pe < 25:
                    score += 1
                    signals.append("估值合理(+1)")
            elif symbol in ['MRK', 'GS']:  # 价值股
                if pe < 15:
                    score += 2
                    signals.append("估值偏低(+2)")
                elif pe < 20:
                    score += 1
                    signals.append("估值合理(+1)")
        
        # 股息评分
        div_yield = fund_data.get('dividend_yield', 0)
        if isinstance(div_yield, (int, float)) and div_yield > 2:
            score += 1
            signals.append("股息吸引力(+1)")
        
        return score, signals
    
    def determine_entry_strategy(self, score, tech_data, symbol):
        """确定入场策略"""
        current_price = tech_data.get('current_price', 0)
        
        # 计算建议买入价位
        support_levels = []
        
        # 基于技术支撑
        ma50 = tech_data.get('ma50', current_price)
        ma200 = tech_data.get('ma200', current_price)
        
        # 计算潜在支撑位
        support_5 = current_price * 0.95   # 5%回调
        support_10 = current_price * 0.90  # 10%回调
        support_15 = current_price * 0.85  # 15%回调
        
        # 基于移动平均线的支撑
        if ma50 < current_price:
            support_levels.append(('MA50支撑', ma50))
        if ma200 and ma200 < current_price:
            support_levels.append(('MA200支撑', ma200))
        
        support_levels.extend([
            ('5%回调', support_5),
            ('10%回调', support_10),
            ('15%回调', support_15)
        ])
        
        # 根据评分确定策略
        if score >= 8:
            strategy = "🟢 立即建仓"
            timing = "当前价位开始分批买入"
        elif score >= 6:
            strategy = "🟡 谨慎建仓"
            timing = "等待小幅回调后买入"
        elif score >= 4:
            strategy = "🟡 等待回调"
            timing = "等待5-10%回调后买入"
        else:
            strategy = "🔴 暂不建仓"
            timing = "等待更大幅度回调"
        
        return strategy, timing, support_levels
    
    def analyze_single_stock(self, symbol):
        """分析单只股票"""
        print(f"\n{'='*60}")
        print(f"📊 {symbol} ({self.stock_info[symbol]['name']}) 深度分析")
        print(f"{'='*60}")
        print(f"行业: {self.stock_info[symbol]['sector']} | 类型: {self.stock_info[symbol]['type']}")
        
        # 获取数据
        stock, data_3y, data_1y, data_6m, data_3m, info = self.get_comprehensive_data(symbol)
        
        if data_1y is None:
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        
        # 计算指标
        tech_data = self.calculate_technical_indicators(data_1y)
        fund_data = self.analyze_fundamentals(info)
        
        # 显示当前状况
        print(f"\n📈 当前市场表现:")
        print(f"  当前价格: ${tech_data['current_price']:.2f}")
        print(f"  1年高点: ${tech_data['high_1y']:.2f}")
        print(f"  1年低点: ${tech_data['low_1y']:.2f}")
        print(f"  价格位置: {tech_data['price_position']:.1%} (0%=最低, 100%=最高)")
        print(f"  从高点回调: {tech_data['drawdown']:.1%}")
        
        print(f"\n🔍 技术指标:")
        print(f"  RSI(14): {tech_data['rsi']:.1f}")
        print(f"  MA20: ${tech_data['ma20']:.2f} ({'支撑' if tech_data['current_price'] > tech_data['ma20'] else '阻力'})")
        print(f"  MA50: ${tech_data['ma50']:.2f} ({'支撑' if tech_data['current_price'] > tech_data['ma50'] else '阻力'})")
        if tech_data['ma200']:
            print(f"  MA200: ${tech_data['ma200']:.2f} ({'支撑' if tech_data['current_price'] > tech_data['ma200'] else '阻力'})")
        print(f"  年化波动率: {tech_data['volatility']:.1f}%")
        
        print(f"\n💰 基本面指标:")
        pe = fund_data.get('pe_ratio', 'N/A')
        print(f"  P/E比率: {pe:.1f}" if isinstance(pe, (int, float)) else f"  P/E比率: {pe}")
        
        pb = fund_data.get('pb_ratio', 'N/A')
        print(f"  P/B比率: {pb:.2f}" if isinstance(pb, (int, float)) else f"  P/B比率: {pb}")
        
        div_yield = fund_data.get('dividend_yield', 'N/A')
        print(f"  股息率: {div_yield:.2f}%" if isinstance(div_yield, (int, float)) else f"  股息率: {div_yield}")
        
        roe = fund_data.get('roe', 'N/A')
        print(f"  ROE: {roe*100:.1f}%" if isinstance(roe, (int, float)) else f"  ROE: {roe}")
        
        market_cap = fund_data.get('market_cap', 'N/A')
        print(f"  市值: ${market_cap:.0f}B" if isinstance(market_cap, (int, float)) else f"  市值: {market_cap}")
        
        # 计算入场信号
        score, signals = self.calculate_entry_signals(tech_data, fund_data, symbol)
        strategy, timing, support_levels = self.determine_entry_strategy(score, tech_data, symbol)
        
        print(f"\n🎯 入场信号分析:")
        print(f"  综合评分: {score}/10")
        print(f"  信号因子: {', '.join(signals) if signals else '无明显信号'}")
        print(f"  建仓策略: {strategy}")
        print(f"  入场时机: {timing}")
        
        print(f"\n📍 关键价位分析:")
        for level_name, price in support_levels:
            distance = ((tech_data['current_price'] - price) / price) * 100
            print(f"  {level_name}: ${price:.2f} (距离当前{distance:+.1f}%)")
        
        return {
            'symbol': symbol,
            'current_price': tech_data['current_price'],
            'score': score,
            'strategy': strategy,
            'timing': timing,
            'support_levels': support_levels,
            'tech_data': tech_data,
            'fund_data': fund_data
        }
    
    def comprehensive_analysis(self):
        """综合分析所有股票"""
        print("🎯 防御性和价值股票建仓时机分析")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = []
        
        # 分析每只股票
        for symbol in self.symbols:
            result = self.analyze_single_stock(symbol)
            if result:
                results.append(result)
        
        # 生成综合建议
        self.generate_portfolio_recommendations(results)
        
        return results
    
    def generate_portfolio_recommendations(self, results):
        """生成投资组合建议"""
        print(f"\n" + "="*80)
        print(f"📋 综合投资建议")
        print(f"="*80)
        
        # 按评分排序
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 建仓优先级排序:")
        print("-" * 50)
        for i, result in enumerate(sorted_results, 1):
            symbol = result['symbol']
            score = result['score']
            strategy = result['strategy']
            current_price = result['current_price']
            
            print(f"{i}. {symbol:<6} | 评分:{score}/10 | ${current_price:.2f} | {strategy}")
        
        # 分类建议
        immediate_buys = [r for r in results if r['score'] >= 8]
        cautious_buys = [r for r in results if 6 <= r['score'] < 8]
        wait_for_dip = [r for r in results if 4 <= r['score'] < 6]
        avoid_now = [r for r in results if r['score'] < 4]
        
        print(f"\n🎯 分类操作建议:")
        print("-" * 40)
        
        if immediate_buys:
            print(f"🟢 立即建仓 (评分≥8):")
            for r in immediate_buys:
                print(f"  • {r['symbol']}: 当前价位分批买入")
        
        if cautious_buys:
            print(f"\n🟡 谨慎建仓 (评分6-7):")
            for r in cautious_buys:
                best_support = min(r['support_levels'], key=lambda x: x[1])
                print(f"  • {r['symbol']}: 等待回调至${best_support[1]:.2f}附近")
        
        if wait_for_dip:
            print(f"\n🟡 等待回调 (评分4-5):")
            for r in wait_for_dip:
                support_10 = r['current_price'] * 0.90
                print(f"  • {r['symbol']}: 等待10%+回调至${support_10:.2f}以下")
        
        if avoid_now:
            print(f"\n🔴 暂不建仓 (评分<4):")
            for r in avoid_now:
                print(f"  • {r['symbol']}: 等待更大幅度调整")
        
        # 资金配置建议
        print(f"\n💰 资金配置建议:")
        print("-" * 30)
        print(f"• 防御性投资总比例: 20-30%")
        print(f"• 单只股票权重: 5-8%")
        print(f"• 分批建仓: 每次1-2%仓位")
        print(f"• 建仓周期: 2-8周完成")
        
        # 风险提醒
        print(f"\n⚠️ 风险提醒:")
        print("-" * 20)
        print(f"• 防御股在牛市中可能跑输大盘")
        print(f"• 关注利率变化对高股息股的影响")
        print(f"• 医药股注意政策和监管风险")
        print(f"• 金融股关注经济周期和信贷风险")

if __name__ == "__main__":
    analyzer = DefensiveStocksAnalysis()
    results = analyzer.comprehensive_analysis() 