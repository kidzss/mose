import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

class PortfolioMonitorDashboard:
    """投资组合监控仪表板"""
    
    def __init__(self):
        self.symbols = ['AMD', 'GOOG', 'NVDA', 'PFE', 'MSFT', 'META', 'TSLA', 'PLTR', 'JPM', 'BRK-B', 'MRK']
        self.target_prices = {
            'MSFT': {'buy_below': 450, 'ideal': 420},
            'META': {'buy_below': 650, 'ideal': 600},
            'PLTR': {'buy_below': 130, 'ideal': 110},
            'JPM': {'buy_below': 270, 'ideal': 250},
            'MRK': {'buy_below': 75, 'ideal': 70},
            'TSLA': {'buy_now': 325, 'stop_loss': 300},
            'BRK-B': {'buy_now': 485, 'stop_loss': 450}
        }
        
        self.current_positions = {
            'AMD': 40, 'GOOG': 30, 'NVDA': 35, 'PFE': 80
        }
        
        self.selling_targets = {
            'AMD': {'target_shares': 22, 'sell_above': 135},
            'GOOG': {'target_shares': 20, 'sell_above': 175},
            'NVDA': {'target_shares': 29, 'sell_above': 150}
        }
    def get_real_time_data(self):
        """获取实时股票数据"""
        stock_data = {}
        
        print("📡 获取实时市场数据...")
        
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                
                # 获取历史数据用于计算技术指标
                hist = stock.history(period="3mo")
                
                if len(hist) > 0:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # 计算技术指标
                    ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                    ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                    
                    # RSI计算
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = (100 - (100 / (1 + rs))).iloc[-1]
                    
                    # 价格位置
                    high_52w = hist['High'].max()
                    low_52w = hist['Low'].min()
                    price_position = (current_price - low_52w) / (high_52w - low_52w)
                    
                    # 成交量分析
                    avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                    current_volume = hist['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    
                    stock_data[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'ma20': ma20,
                        'ma50': ma50,
                        'rsi': rsi,
                        'price_position': price_position,
                        'volume_ratio': volume_ratio,
                        'high_52w': high_52w,
                        'low_52w': low_52w
                    }
                    
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                
        return stock_data
    
    def analyze_selling_opportunities(self, stock_data):
        """分析卖出机会"""
        print(f"\n🔴 卖出机会分析")
        print("=" * 60)
        
        selling_signals = {}
        
        for symbol in ['AMD', 'GOOG', 'NVDA']:
            if symbol in stock_data and symbol in self.selling_targets:
                data = stock_data[symbol]
                target = self.selling_targets[symbol]
                
                current_shares = self.current_positions[symbol]
                target_shares = target['target_shares']
                shares_to_sell = current_shares - target_shares
                
                # 卖出信号评分
                sell_score = 0
                signals = []
                
                if data['rsi'] > 70:
                    sell_score += 2
                    signals.append("RSI超买")
                elif data['rsi'] > 60:
                    sell_score += 1
                    signals.append("RSI偏高")
                
                if data['price'] > data['ma20']:
                    sell_score += 1
                    signals.append("价格>MA20")
                
                if data['price'] >= target.get('sell_above', float('inf')):
                    sell_score += 2
                    signals.append("达到卖出目标价")
                
                if data['volume_ratio'] > 1.5:
                    sell_score += 1
                    signals.append("成交量放大")
                
                # 卖出建议
                if sell_score >= 4:
                    recommendation = "🔴 强烈建议卖出"
                elif sell_score >= 3:
                    recommendation = "🟡 可以考虑卖出"
                else:
                    recommendation = "🟢 暂时持有"
                
                selling_signals[symbol] = {
                    'score': sell_score,
                    'signals': signals,
                    'recommendation': recommendation,
                    'shares_to_sell': shares_to_sell
                }
                
                print(f"{symbol:6} | ${data['price']:7.2f} | RSI:{data['rsi']:5.1f} | {recommendation}")
                print(f"       | 需减仓: {shares_to_sell}股 | 信号: {', '.join(signals)}")
                
        return selling_signals
    
    def analyze_buying_opportunities(self, stock_data):
        """分析买入机会"""
        print(f"\n🟢 买入机会分析")
        print("=" * 60)
        
        buying_signals = {}
        
        # 立即买入目标
        immediate_targets = ['TSLA', 'BRK-B']
        
        print("📊 立即买入目标:")
        print("-" * 30)
        
        for symbol in immediate_targets:
            if symbol in stock_data and symbol in self.target_prices:
                data = stock_data[symbol]
                target = self.target_prices[symbol]
                
                buy_score = 0
                signals = []
                
                if data['rsi'] < 50:
                    buy_score += 2
                    signals.append("RSI良好")
                
                if data['price_position'] < 0.6:
                    buy_score += 1
                    signals.append("价格合理")
                
                if data['price'] > data['ma20']:
                    buy_score += 1
                    signals.append("趋势向上")
                
                if buy_score >= 3:
                    recommendation = "🟢 立即买入"
                elif buy_score >= 2:
                    recommendation = "🟡 谨慎买入"
                else:
                    recommendation = "🔴 等待更好时机"
                
                buying_signals[symbol] = {
                    'score': buy_score,
                    'signals': signals,
                    'recommendation': recommendation,
                    'category': 'immediate'
                }
                
                print(f"{symbol:6} | ${data['price']:7.2f} | RSI:{data['rsi']:5.1f} | {recommendation}")
                print(f"       | 目标价: ${target.get('buy_now', 'N/A')} | 止损: ${target.get('stop_loss', 'N/A')}")
        
        print(f"\n📊 等待回调目标:")
        print("-" * 30)
        
        # 等待回调目标
        wait_targets = ['MSFT', 'META', 'PLTR', 'JPM', 'MRK']
        
        for symbol in wait_targets:
            if symbol in stock_data and symbol in self.target_prices:
                data = stock_data[symbol]
                target = self.target_prices[symbol]
                
                current_price = data['price']
                buy_below = target['buy_below']
                ideal_price = target['ideal']
                
                distance_to_buy = ((current_price - buy_below) / buy_below) * 100
                distance_to_ideal = ((current_price - ideal_price) / ideal_price) * 100
                
                if current_price <= buy_below:
                    status = "🟢 已达买入价位"
                elif distance_to_buy <= 5:
                    status = "🟡 接近买入价位"
                else:
                    status = f"🔴 等待回调 ({distance_to_buy:+.1f}%)"
                
                buying_signals[symbol] = {
                    'current_price': current_price,
                    'buy_below': buy_below,
                    'ideal_price': ideal_price,
                    'distance_to_buy': distance_to_buy,
                    'status': status,
                    'category': 'wait'
                }
                
                print(f"{symbol:6} | ${current_price:7.2f} | 目标: ${buy_below} | 理想: ${ideal_price} | {status}")
        
        return buying_signals
    
    def generate_daily_alerts(self, stock_data, selling_signals, buying_signals):
        """生成每日提醒"""
        print(f"\n🚨 今日重点提醒")
        print("=" * 60)
        
        alerts = []
        
        # 卖出提醒
        for symbol, signal in selling_signals.items():
            if signal['score'] >= 3:
                alerts.append(f"🔴 {symbol}: {signal['recommendation']} (评分: {signal['score']}/5)")
        
        # 买入提醒
        for symbol, signal in buying_signals.items():
            if signal['category'] == 'immediate' and signal['score'] >= 3:
                alerts.append(f"🟢 {symbol}: {signal['recommendation']} (评分: {signal['score']}/4)")
            elif signal['category'] == 'wait' and signal['distance_to_buy'] <= 2:
                alerts.append(f"🟡 {symbol}: 接近买入价位，距离目标{signal['distance_to_buy']:+.1f}%")
        
        # 技术面警告
        for symbol, data in stock_data.items():
            if data['rsi'] > 80:
                alerts.append(f"⚠️ {symbol}: RSI严重超买 ({data['rsi']:.1f})")
            elif data['rsi'] < 20:
                alerts.append(f"⚠️ {symbol}: RSI严重超卖 ({data['rsi']:.1f})")
        
        if alerts:
            for alert in alerts:
                print(alert)
        else:
            print("📊 当前无重要提醒，继续观察市场")
        
        return alerts
    
    def create_execution_checklist(self):
        """创建执行检查清单"""
        print(f"\n✅ 执行检查清单")
        print("=" * 60)
        
        checklist = {
            "第一阶段 - 减仓操作": [
                "[ ] AMD减仓18股 (当前40股 → 目标22股)",
                "[ ] GOOG减仓10股 (当前30股 → 目标20股)", 
                "[ ] NVDA减仓6股 (当前35股 → 目标29股)"
            ],
            "第二阶段 - 立即建仓": [
                "[ ] TSLA买入4股 @ $325附近",
                "[ ] BRK-B买入4股 @ $485附近"
            ],
            "第三阶段 - 等待回调": [
                "[ ] MSFT等待$450以下买入5股",
                "[ ] META等待$650以下买入3股",
                "[ ] PLTR等待$120以下买入5股"
            ],
            "第四阶段 - 防御建仓": [
                "[ ] JPM等待$260以下买入8股",
                "[ ] MRK等待$75以下买入20股"
            ]
        }
        
        for phase, tasks in checklist.items():
            print(f"\n{phase}:")
            for task in tasks:
                print(f"  {task}")
        
        return checklist
    
    def run_dashboard(self):
        """运行监控仪表板"""
        print("🎯 投资组合再平衡监控仪表板")
        print("=" * 80)
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取实时数据
        stock_data = self.get_real_time_data()
        
        if not stock_data:
            print("❌ 无法获取市场数据，请检查网络连接")
            return
        
        # 分析卖出机会
        selling_signals = self.analyze_selling_opportunities(stock_data)
        
        # 分析买入机会
        buying_signals = self.analyze_buying_opportunities(stock_data)
        
        # 生成每日提醒
        alerts = self.generate_daily_alerts(stock_data, selling_signals, buying_signals)
        
        # 执行检查清单
        self.create_execution_checklist()
        
        # 保存监控结果
        monitor_result = {
            'timestamp': datetime.now().isoformat(),
            'stock_data': stock_data,
            'selling_signals': selling_signals,
            'buying_signals': buying_signals,
            'alerts': alerts
        }
        
        # 将结果保存到文件
        with open('portfolio_monitor_result.json', 'w', encoding='utf-8') as f:
            json.dump(monitor_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 监控结果已保存至 portfolio_monitor_result.json")
        
        return monitor_result

if __name__ == "__main__":
    dashboard = PortfolioMonitorDashboard()
    result = dashboard.run_dashboard() 