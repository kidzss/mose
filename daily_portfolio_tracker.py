#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日持股分析和买入计划追踪系统
专门追踪JPM等待回调买入方案及其他投资计划
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class DailyPortfolioTracker:
    """每日投资组合追踪器"""
    
    def __init__(self):
        self.portfolio_value = 27533.17
        self.tracking_file = "daily_portfolio_tracking.json"
        self.current_holdings = {
            'NVDA': {'shares': 35, 'avg_cost': 143.50, 'weight': 18.29},
            'GOOG': {'shares': 30, 'avg_cost': 167.80, 'weight': 18.28},
            'AMD': {'shares': 30, 'avg_cost': 128.20, 'weight': 13.97},
            'PFE': {'shares': 80, 'avg_cost': 24.10, 'weight': 6.96},
            'MRK': {'shares': 8, 'avg_cost': 79.50, 'weight': 2.30},
            'BRK-B': {'shares': 2, 'avg_cost': 485.36, 'weight': 3.52}
        }
        
        # 买入计划
        self.buying_plans = {
            'JPM': {
                'strategy': 'A-等待回调',
                'status': '等待中',
                'target_weight': '6-8%',
                'total_allocation': 2200,  # $2200预算
                'entry_conditions': {
                    'price_range': [260, 265],
                    'rsi_target': [50, 60],
                    'technical_signal': '超买缓解'
                },
                'execution_plan': {
                    'batch_1': {'price': 265, 'amount': 880, 'percentage': 40},
                    'batch_2': {'price': 260, 'amount': 880, 'percentage': 40}, 
                    'batch_3': {'price': 255, 'amount': 440, 'percentage': 20}
                },
                'risk_control': {
                    'stop_loss': 242,
                    'target_1': 315,
                    'target_2': 320
                },
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'expected_timeframe': '2-4周'
            }
        }
        
        self.load_tracking_data()
    
    def load_tracking_data(self):
        """加载历史追踪数据"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    self.tracking_history = json.load(f)
            except:
                self.tracking_history = {}
        else:
            self.tracking_history = {}
    
    def save_tracking_data(self):
        """保存追踪数据"""
        def convert_numpy_types(obj):
            """转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 转换数据类型
        converted_data = convert_numpy_types(self.tracking_history)
        
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    def get_current_market_data(self):
        """获取当前市场数据"""
        print("📊 获取当前市场数据...")
        
        market_data = {}
        
        # 获取持仓股票数据
        all_symbols = list(self.current_holdings.keys()) + list(self.buying_plans.keys())
        
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')  # 获取最近5天数据
                info = ticker.info
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    daily_change = (current_price / prev_close - 1) * 100
                    
                    # 计算技术指标
                    hist_1y = ticker.history(period='1y')
                    rsi = self.calculate_rsi(hist_1y['Close']).iloc[-1] if len(hist_1y) > 14 else 50
                    
                    # 52周位置
                    high_52w = hist_1y['High'].max()
                    low_52w = hist_1y['Low'].min()
                    price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
                    
                    market_data[symbol] = {
                        'current_price': current_price,
                        'daily_change': daily_change,
                        'rsi': rsi,
                        'price_position_52w': price_position,
                        'high_52w': high_52w,
                        'low_52w': low_52w,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    }
                    
            except Exception as e:
                print(f"获取{symbol}数据失败: {e}")
        
        return market_data
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_current_holdings(self, market_data):
        """分析当前持仓"""
        print(f"\n📈 当前持仓分析 ({datetime.now().strftime('%Y-%m-%d')})")
        print("=" * 70)
        
        total_value = 0
        holdings_analysis = {}
        
        print(f"{'股票':<6} {'股数':<6} {'成本':<8} {'现价':<8} {'日涨跌':<8} {'持仓价值':<10} {'盈亏':<10}")
        print("-" * 70)
        
        for symbol, holding in self.current_holdings.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                daily_change = data['daily_change']
                
                shares = holding['shares']
                avg_cost = holding['avg_cost']
                current_value = shares * current_price
                total_return = (current_price / avg_cost - 1) * 100
                
                total_value += current_value
                
                holdings_analysis[symbol] = {
                    'shares': shares,
                    'avg_cost': avg_cost,
                    'current_price': current_price,
                    'current_value': current_value,
                    'daily_change': daily_change,
                    'total_return': total_return,
                    'rsi': data['rsi'],
                    'price_position': data['price_position_52w']
                }
                
                print(f"{symbol:<6} {shares:<6} ${avg_cost:<7.2f} ${current_price:<7.2f} "
                      f"{daily_change:>+6.1f}% ${current_value:<9.0f} {total_return:>+7.1f}%")
        
        cash_position = self.portfolio_value - total_value
        cash_percentage = cash_position / self.portfolio_value * 100
        
        print("-" * 70)
        print(f"持仓总值: ${total_value:.0f}")
        print(f"现金仓位: ${cash_position:.0f} ({cash_percentage:.1f}%)")
        print(f"总资产: ${self.portfolio_value:.0f}")
        
        return holdings_analysis, total_value, cash_position
    
    def monitor_buying_plans(self, market_data):
        """监控买入计划"""
        print(f"\n📋 买入计划监控")
        print("=" * 70)
        
        for symbol, plan in self.buying_plans.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                rsi = data['rsi']
                daily_change = data['daily_change']
                
                print(f"\n🎯 {symbol} - {plan['strategy']}")
                print("-" * 50)
                print(f"当前状态: {plan['status']}")
                print(f"当前价格: ${current_price:.2f} ({daily_change:+.1f}%)")
                print(f"RSI: {rsi:.1f}")
                print(f"52周位置: {data['price_position_52w']:.1f}%")
                
                # 检查买入条件
                target_range = plan['entry_conditions']['price_range']
                rsi_range = plan['entry_conditions']['rsi_target']
                
                conditions_met = []
                conditions_pending = []
                
                # 价格条件检查
                if target_range[0] <= current_price <= target_range[1]:
                    conditions_met.append(f"✅ 价格在目标区间 ${target_range[0]}-${target_range[1]}")
                else:
                    conditions_pending.append(f"⏳ 等待价格回调至 ${target_range[0]}-${target_range[1]} (当前${current_price:.2f})")
                
                # RSI条件检查
                if rsi_range[0] <= rsi <= rsi_range[1]:
                    conditions_met.append(f"✅ RSI在目标区间 {rsi_range[0]}-{rsi_range[1]}")
                else:
                    conditions_pending.append(f"⏳ 等待RSI回落至 {rsi_range[0]}-{rsi_range[1]} (当前{rsi:.1f})")
                
                print(f"\n📊 买入条件检查:")
                for condition in conditions_met:
                    print(f"   {condition}")
                for condition in conditions_pending:
                    print(f"   {condition}")
                
                # 执行建议
                if len(conditions_met) >= 2:
                    print(f"\n🚀 执行建议: 可以开始分批买入！")
                    
                    # 显示执行计划
                    print(f"\n💰 分批买入计划:")
                    for batch, details in plan['execution_plan'].items():
                        status = "✅ 可执行" if current_price <= details['price'] else "⏳ 等待"
                        print(f"   {batch}: ${details['price']:.0f} - ${details['amount']:.0f} ({details['percentage']}%) {status}")
                    
                elif len(conditions_met) >= 1:
                    print(f"\n⚠️ 执行建议: 部分条件满足，可考虑小仓位试探")
                else:
                    print(f"\n⏳ 执行建议: 继续等待，条件尚未成熟")
                
                # 风险提醒
                stop_loss = plan['risk_control']['stop_loss']
                risk_distance = (current_price - stop_loss) / current_price * 100
                print(f"\n⚠️ 风险控制:")
                print(f"   止损价: ${stop_loss:.0f} (风险距离: {risk_distance:.1f}%)")
                print(f"   目标价: ${plan['risk_control']['target_1']:.0f} - ${plan['risk_control']['target_2']:.0f}")
    
    def generate_daily_report(self, market_data, holdings_analysis, total_value, cash_position):
        """生成每日报告"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        daily_report = {
            'date': today,
            'portfolio_summary': {
                'total_value': self.portfolio_value,
                'holdings_value': total_value,
                'cash_position': cash_position,
                'cash_percentage': cash_position / self.portfolio_value * 100
            },
            'holdings_performance': holdings_analysis,
            'market_data': market_data,
            'buying_plans_status': {}
        }
        
        # 记录买入计划状态
        for symbol, plan in self.buying_plans.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                target_range = plan['entry_conditions']['price_range']
                rsi = data['rsi']
                rsi_range = plan['entry_conditions']['rsi_target']
                
                # 评估条件满足情况
                price_condition = target_range[0] <= current_price <= target_range[1]
                rsi_condition = rsi_range[0] <= rsi <= rsi_range[1]
                
                daily_report['buying_plans_status'][symbol] = {
                    'strategy': plan['strategy'],
                    'current_price': current_price,
                    'target_price_range': target_range,
                    'price_condition_met': price_condition,
                    'current_rsi': rsi,
                    'target_rsi_range': rsi_range,
                    'rsi_condition_met': rsi_condition,
                    'overall_ready': price_condition and rsi_condition,
                    'days_waiting': (datetime.now() - datetime.strptime(plan['created_date'], '%Y-%m-%d')).days
                }
        
        # 保存到历史记录
        self.tracking_history[today] = daily_report
        self.save_tracking_data()
        
        return daily_report
    
    def print_summary_and_actions(self, daily_report):
        """打印总结和行动建议"""
        print(f"\n📋 今日总结与行动建议")
        print("=" * 70)
        
        # 投资组合总结
        portfolio = daily_report['portfolio_summary']
        print(f"💰 投资组合状况:")
        print(f"   总资产: ${portfolio['total_value']:.0f}")
        print(f"   持仓价值: ${portfolio['holdings_value']:.0f}")
        print(f"   现金仓位: ${portfolio['cash_position']:.0f} ({portfolio['cash_percentage']:.1f}%)")
        
        # 持仓表现
        best_performer = max(daily_report['holdings_performance'].items(), 
                           key=lambda x: x[1]['daily_change'])
        worst_performer = min(daily_report['holdings_performance'].items(), 
                            key=lambda x: x[1]['daily_change'])
        
        print(f"\n📈 今日表现:")
        print(f"   最佳: {best_performer[0]} {best_performer[1]['daily_change']:+.1f}%")
        print(f"   最差: {worst_performer[0]} {worst_performer[1]['daily_change']:+.1f}%")
        
        # 买入计划状态
        print(f"\n🎯 买入计划状态:")
        for symbol, status in daily_report['buying_plans_status'].items():
            ready_status = "🟢 就绪" if status['overall_ready'] else "🟡 等待"
            print(f"   {symbol}: {ready_status} (等待{status['days_waiting']}天)")
            
            if status['overall_ready']:
                print(f"      ✅ 条件满足，建议开始分批买入！")
            else:
                pending_conditions = []
                if not status['price_condition_met']:
                    pending_conditions.append(f"价格回调至${status['target_price_range'][0]}-${status['target_price_range'][1]}")
                if not status['rsi_condition_met']:
                    pending_conditions.append(f"RSI回落至{status['target_rsi_range'][0]}-{status['target_rsi_range'][1]}")
                print(f"      ⏳ 等待: {', '.join(pending_conditions)}")
        
        # 明日关注点
        print(f"\n👀 明日关注点:")
        for symbol, status in daily_report['buying_plans_status'].items():
            current_price = status['current_price']
            target_max = status['target_price_range'][1]
            distance = (current_price - target_max) / current_price * 100
            
            if distance <= 5:  # 距离目标价格5%以内
                print(f"   • {symbol}: 接近买入区间，密切关注")
            elif distance <= 10:
                print(f"   • {symbol}: 可能即将进入买入区间")
            else:
                print(f"   • {symbol}: 继续等待回调")
    
    def run_daily_analysis(self):
        """运行每日分析"""
        print("📊 每日投资组合分析")
        print("=" * 70)
        print(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 获取市场数据
        market_data = self.get_current_market_data()
        
        if not market_data:
            print("❌ 无法获取市场数据")
            return
        
        # 分析当前持仓
        holdings_analysis, total_value, cash_position = self.analyze_current_holdings(market_data)
        
        # 监控买入计划
        self.monitor_buying_plans(market_data)
        
        # 生成每日报告
        daily_report = self.generate_daily_report(market_data, holdings_analysis, total_value, cash_position)
        
        # 打印总结和行动建议
        self.print_summary_and_actions(daily_report)
        
        print(f"\n💾 数据已保存到: {self.tracking_file}")
        print(f"🎊 每日分析完成！")

if __name__ == "__main__":
    tracker = DailyPortfolioTracker()
    tracker.run_daily_analysis() 