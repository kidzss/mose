#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日持股分析系统
Daily Holdings Analysis System
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

class DailyHoldingsAnalyzer:
    def __init__(self):
        # 从配置文件读取持仓信息
        try:
            with open('portfolio_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.portfolio = {}
                
                # 转换持仓数据格式
                for symbol, position in config['positions'].items():
                    if position.get('shares', 0) > 0:  # 只包含有持仓的股票
                        self.portfolio[symbol] = {
                            'shares': position['shares'],
                            'cost': position['cost_basis'],
                            'technical_analysis': position.get('technical_analysis', {})
                        }
        except Exception as e:
            print(f"⚠️ 无法读取配置文件，使用默认配置: {e}")
            # 默认持仓配置
            self.portfolio = {
                'AMD': {'shares': 20, 'cost': 125.212},
            }
        
        # 监控的市场指标
        self.market_indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        
        # 对比股票池
        self.watchlist = ['AMD', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'ASML', 'MRVL']
    
    def get_today_data(self, symbols):
        """获取今日收盘数据"""
        print("📊 获取最新收盘数据...")
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # 获取最近数据
                hist = ticker.history(period='6mo', interval='1d')
                info = ticker.info
                
                if not hist.empty:
                    # 今日数据
                    today_data = hist.iloc[-1]
                    yesterday_data = hist.iloc[-2] if len(hist) > 1 else today_data
                    
                    current_price = today_data['Close']
                    prev_close = yesterday_data['Close']
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # 技术指标
                    rsi = self.calculate_rsi(hist['Close'])
                    ma_5 = hist['Close'].rolling(5).mean().iloc[-1]
                    ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                    
                    # 成交量分析
                    volume = today_data['Volume']
                    avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                    volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                    
                    # 52周数据
                    high_52w = hist['High'].max()
                    low_52w = hist['Low'].min()
                    position_52w = (current_price - low_52w) / (high_52w - low_52w) * 100
                    
                    data[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': volume,
                        'volume_ratio': volume_ratio,
                        'rsi': rsi,
                        'ma_5': ma_5,
                        'ma_20': ma_20,
                        'ma_50': ma_50,
                        'high_52w': high_52w,
                        'low_52w': low_52w,
                        'position_52w': position_52w,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0)
                    }
                    
                    print(f"✅ {symbol}: ${current_price:.2f} ({change_pct:+.2f}%)")
                
            except Exception as e:
                print(f"❌ {symbol} 数据获取失败: {e}")
        
        return data
    
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
    
    def analyze_portfolio_performance(self, data):
        """分析投资组合表现"""
        print("\n💼 === 投资组合表现分析 ===")
        
        total_value = 0
        total_cost = 0
        portfolio_analysis = []
        
        for symbol, position in self.portfolio.items():
            if symbol in data:
                stock_data = data[symbol]
                shares = position['shares']
                cost_price = position['cost']
                
                current_price = stock_data['price']
                current_value = current_price * shares
                cost_value = cost_price * shares
                
                unrealized_pnl = current_value - cost_value
                pnl_pct = (unrealized_pnl / cost_value) * 100
                
                total_value += current_value
                total_cost += cost_value
                
                # 风险评估
                rsi = stock_data['rsi']
                position_52w = stock_data['position_52w']
                
                # 技术状态
                if rsi < 30:
                    tech_status = "超卖-机会"
                elif rsi > 70:
                    tech_status = "超买-风险"
                elif 30 <= rsi <= 50:
                    tech_status = "偏弱-观察"
                elif 50 <= rsi <= 70:
                    tech_status = "健康-持有"
                else:
                    tech_status = "中性"
                
                # 操作建议
                if pnl_pct > 15 and rsi > 70:
                    suggestion = "考虑减仓锁利"
                elif pnl_pct > 8 and rsi > 65:
                    suggestion = "设置止损保护"
                elif pnl_pct < -8 and rsi < 35:
                    suggestion = "考虑加仓摊成本"
                elif rsi < 30:
                    suggestion = "技术面支持持有"
                else:
                    suggestion = "维持当前仓位"
                
                # 检查是否有技术分析记录
                tech_analysis = position.get('technical_analysis', {})
                
                portfolio_analysis.append({
                    'symbol': symbol,
                    'shares': shares,
                    'cost_price': cost_price,
                    'current_price': current_price,
                    'current_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_pct': pnl_pct,
                    'rsi': rsi,
                    'tech_status': tech_status,
                    'suggestion': suggestion,
                    'position_52w': position_52w,
                    'tech_analysis': tech_analysis
                })
        
        # 总体表现
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        print(f"\n📈 组合总览:")
        print(f"   总市值: ${total_value:,.2f}")
        print(f"   总成本: ${total_cost:,.2f}")
        print(f"   总盈亏: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        print(f"   持仓数量: {len(portfolio_analysis)}只")
        
        print(f"\n📊 个股详情:")
        for stock in portfolio_analysis:
            print(f"\n🎯 {stock['symbol']}:")
            print(f"   持仓: {stock['shares']}股 @ ${stock['cost_price']:.2f}")
            print(f"   现价: ${stock['current_price']:.2f}")
            print(f"   市值: ${stock['current_value']:.2f}")
            print(f"   盈亏: ${stock['unrealized_pnl']:+.2f} ({stock['pnl_pct']:+.2f}%)")
            print(f"   RSI: {stock['rsi']:.1f} ({stock['tech_status']})")
            print(f"   52周位置: {stock['position_52w']:.1f}%")
            
            # 显示技术分析记录
            if stock['tech_analysis']:
                tech = stock['tech_analysis']
                print(f"   📈 技术分析 ({tech.get('date', 'N/A')}):")
                print(f"      形态: {tech.get('pattern', 'N/A')}")
                print(f"      趋势: {tech.get('trend_direction', 'N/A')}")
                print(f"      目标: {tech.get('price_target', 'N/A')}")
                print(f"      策略: {tech.get('strategy', 'N/A')}")
                if tech.get('note'):
                    print(f"      备注: {tech['note']}")
            
            print(f"   💡 建议: {stock['suggestion']}")
        
        return portfolio_analysis
    
    def analyze_market_environment(self, data):
        """分析市场环境"""
        print(f"\n🌍 === 市场环境分析 ===")
        
        # 市场指数分析
        if '^GSPC' in data:
            spx = data['^GSPC']
            print(f"📊 标普500: {spx['price']:.2f} ({spx['change_pct']:+.2f}%)")
        
        if '^IXIC' in data:
            nasdaq = data['^IXIC']
            print(f"📊 纳斯达克: {nasdaq['price']:.2f} ({nasdaq['change_pct']:+.2f}%)")
        
        if '^VIX' in data:
            vix = data['^VIX']
            vix_level = vix['price']
            
            if vix_level < 15:
                vix_status = "极低恐慌 - 市场过度乐观"
            elif vix_level < 20:
                vix_status = "低恐慌 - 市场相对平静"
            elif vix_level < 25:
                vix_status = "中等恐慌 - 保持谨慎"
            else:
                vix_status = "高恐慌 - 市场担忧加剧"
            
            print(f"😰 VIX恐慌指数: {vix_level:.2f} ({vix_status})")
        
        # 行业表现对比
        tech_stocks = ['AMD', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META']
        tech_performance = []
        
        print(f"\n💻 科技股表现:")
        for symbol in tech_stocks:
            if symbol in data:
                stock_data = data[symbol]
                print(f"   {symbol}: {stock_data['change_pct']:+.2f}% (RSI: {stock_data['rsi']:.1f})")
                tech_performance.append(stock_data['change_pct'])
        
        if tech_performance:
            avg_tech_performance = np.mean(tech_performance)
            print(f"   💡 科技股平均涨幅: {avg_tech_performance:+.2f}%")
    
    def generate_trading_signals(self, data):
        """生成交易信号"""
        print(f"\n⚡ === 交易信号分析 ===")
        
        signals = []
        
        for symbol in self.watchlist:
            if symbol in data:
                stock_data = data[symbol]
                price = stock_data['price']
                rsi = stock_data['rsi']
                ma_20 = stock_data['ma_20']
                ma_50 = stock_data['ma_50']
                volume_ratio = stock_data['volume_ratio']
                position_52w = stock_data['position_52w']
                
                signal_score = 0
                signal_reasons = []
                
                # RSI信号
                if rsi < 25:
                    signal_score += 3
                    signal_reasons.append("RSI严重超卖")
                elif rsi < 30:
                    signal_score += 2
                    signal_reasons.append("RSI超卖")
                elif rsi > 75:
                    signal_score -= 3
                    signal_reasons.append("RSI严重超买")
                elif rsi > 70:
                    signal_score -= 2
                    signal_reasons.append("RSI超买")
                
                # 均线信号
                if price > ma_20 > ma_50:
                    signal_score += 2
                    signal_reasons.append("多头排列")
                elif price < ma_20 < ma_50:
                    signal_score -= 2
                    signal_reasons.append("空头排列")
                
                # 成交量信号
                if volume_ratio > 2:
                    signal_score += 1
                    signal_reasons.append("放量突破")
                
                # 52周位置
                if position_52w < 20:
                    signal_score += 1
                    signal_reasons.append("接近年低")
                elif position_52w > 80:
                    signal_score -= 1
                    signal_reasons.append("接近年高")
                
                # 综合评级
                if signal_score >= 4:
                    rating = "强烈买入"
                elif signal_score >= 2:
                    rating = "买入"
                elif signal_score <= -4:
                    rating = "强烈卖出"
                elif signal_score <= -2:
                    rating = "卖出"
                else:
                    rating = "中性"
                
                signals.append({
                    'symbol': symbol,
                    'rating': rating,
                    'score': signal_score,
                    'reasons': signal_reasons,
                    'rsi': rsi,
                    'price': price
                })
        
        # 按评分排序
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🎯 今日交易信号排行:")
        for i, signal in enumerate(signals[:10], 1):
            reasons_str = ", ".join(signal['reasons']) if signal['reasons'] else "技术面中性"
            print(f"{i:2d}. {signal['symbol']}: {signal['rating']} (评分: {signal['score']:+d})")
            print(f"     理由: {reasons_str}")
            print(f"     价格: ${signal['price']:.2f}, RSI: {signal['rsi']:.1f}")
    
    def create_daily_summary(self, data):
        """创建每日总结"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"\n📋 === 每日总结 ({current_date}) ===")
        
        # 市场表现总结
        market_summary = []
        if '^GSPC' in data:
            market_summary.append(f"标普500: {data['^GSPC']['change_pct']:+.2f}%")
        if '^IXIC' in data:
            market_summary.append(f"纳斯达克: {data['^IXIC']['change_pct']:+.2f}%")
        
        print(f"🏛️ 市场表现: {' | '.join(market_summary)}")
        
        # 持仓表现总结
        portfolio_summary = []
        for symbol in self.portfolio:
            if symbol in data:
                change_pct = data[symbol]['change_pct']
                portfolio_summary.append(f"{symbol}: {change_pct:+.2f}%")
        
        if portfolio_summary:
            print(f"💼 持仓表现: {' | '.join(portfolio_summary)}")
        
        # 关键提醒
        alerts = []
        for symbol in self.portfolio:
            if symbol in data:
                rsi = data[symbol]['rsi']
                if rsi > 75:
                    alerts.append(f"{symbol} RSI超买需关注")
                elif rsi < 25:
                    alerts.append(f"{symbol} RSI超卖现机会")
        
        if alerts:
            print(f"⚠️ 重要提醒: {' | '.join(alerts)}")
        else:
            print(f"✅ 当前持仓技术面健康")
        
        print(f"\n🎯 明日关注要点:")
        print(f"   1. 继续关注地缘政治发展")
        print(f"   2. 美联储政策预期变化")
        print(f"   3. 科技股财报季表现")
        print(f"   4. 市场成交量变化")
    
    def check_tsla_add_position_triggers(self, data):
        """检查TSLA加仓触发条件"""
        if 'TSLA' not in data:
            return
        
        tsla_data = data['TSLA']
        current_price = tsla_data['price']
        
        print("\n🚗 === TSLA倒金字塔加仓策略提醒 ===")
        print(f"当前价格: ${current_price:.2f}")
        print(f"关键支撑: $336 (不破看涨)")
        
        # 加仓策略配置
        strategy = {
            'batch_1': {'range': (296, 300), 'amount': 825, 'allocation': '30%', 'logic': '试探性建仓，验证支撑有效性'},
            'batch_2': {'range': (285, 290), 'amount': 1100, 'allocation': '40%', 'logic': '确认趋势后重仓买入，获取主要收益'},
            'batch_3': {'range': (273, 280), 'amount': 825, 'allocation': '30%', 'logic': '极值区域收割，风险最低时加码'}
        }
        
        # 检查触发条件
        triggered_batches = []
        
        for batch_name, batch_info in strategy.items():
            min_price, max_price = batch_info['range']
            if min_price <= current_price <= max_price:
                triggered_batches.append((batch_name, batch_info))
        
        if triggered_batches:
            print("\n🎯 触发加仓条件:")
            for batch_name, batch_info in triggered_batches:
                print(f"   📍 {batch_name.upper()}: ${batch_info['range'][0]}-${batch_info['range'][1]}")
                print(f"      💰 资金: ${batch_info['amount']} ({batch_info['allocation']})")
                print(f"      📝 逻辑: {batch_info['logic']}")
                print(f"      ⚠️  建议: 立即考虑执行加仓!")
        else:
            print("\n⏳ 等待加仓时机:")
            
            # 显示距离各批次的差距
            for batch_name, batch_info in strategy.items():
                min_price, max_price = batch_info['range']
                mid_price = (min_price + max_price) / 2
                distance = ((current_price - mid_price) / mid_price) * 100
                
                if current_price > max_price:
                    status = f"还需下跌 {distance:.1f}%"
                elif current_price < min_price:
                    status = f"已跌破 {abs(distance):.1f}%"
                else:
                    status = "在区间内"
                
                print(f"   📍 {batch_name.upper()}: ${min_price}-${max_price} ({status})")
                print(f"      💰 准备资金: ${batch_info['amount']} ({batch_info['allocation']})")
        
        # 风险提醒
        if current_price < 336:
            print(f"\n⚠️  风险提醒: 已跌破关键支撑$336，当前${current_price:.2f}")
            print("   需要重新评估支撑有效性")
        elif current_price > 350:
            print(f"\n📈 价格偏高: 当前${current_price:.2f}，建议等待回调")
        
        # 技术分析提醒
        rsi = tsla_data.get('rsi', 50)
        if rsi < 30:
            print(f"   📊 RSI: {rsi:.1f} (超卖，支持加仓)")
        elif rsi > 70:
            print(f"   📊 RSI: {rsi:.1f} (超买，谨慎加仓)")
        else:
            print(f"   📊 RSI: {rsi:.1f} (中性)")
        
        print(f"\n💡 总策略: 倒金字塔加仓，总资金$2,750 (约占组合10%)")
        print(f"   止损位: 硬止损$260, 软止损$270")
        print(f"   持有期: 最长6个月，每月底评估")

    def run_daily_analysis(self):
        """运行每日分析"""
        print("🚀 开始每日持股分析...")
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取所有需要的股票数据
        all_symbols = list(self.portfolio.keys()) + self.market_indices + self.watchlist
        all_symbols = list(set(all_symbols))  # 去重
        
        data = self.get_today_data(all_symbols)
        
        if not data:
            print("❌ 无法获取市场数据")
            return
        
        # 分析投资组合
        portfolio_analysis = self.analyze_portfolio_performance(data)
        
        # 分析市场环境
        self.analyze_market_environment(data)
        
        # TSLA加仓提醒检查
        self.check_tsla_add_position_triggers(data)
        
        # 生成交易信号
        self.generate_trading_signals(data)
        
        # 创建每日总结
        self.create_daily_summary(data)
        
        print("\n✅ 每日分析完成!")
        print("="*80)

def main():
    analyzer = DailyHoldingsAnalyzer()
    analyzer.run_daily_analysis()

if __name__ == "__main__":
    main() 