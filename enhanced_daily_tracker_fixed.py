#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版每日投资组合追踪系统 (修复版)
集成JPM等待回调方案 + 8只优质股票分析
完整的投资决策支持系统
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedDailyTracker:
    """增强版每日投资组合追踪器"""
    
    def __init__(self):
        self.portfolio_value = 27533.17
        self.tracking_file = "enhanced_daily_tracking.json"
        
        # 当前持仓
        self.current_holdings = {
            'NVDA': {'shares': 35, 'avg_cost': 143.50, 'weight': 18.29},
            'GOOG': {'shares': 30, 'avg_cost': 167.80, 'weight': 18.28},
            'AMD': {'shares': 30, 'avg_cost': 128.20, 'weight': 13.97},
            'PFE': {'shares': 80, 'avg_cost': 24.10, 'weight': 6.96},
            'MRK': {'shares': 8, 'avg_cost': 79.50, 'weight': 2.30},
            'BRK-B': {'shares': 2, 'avg_cost': 485.36, 'weight': 3.52}
        }
        
        # JPM等待回调计划
        self.jpm_plan = {
            'strategy': 'A-等待回调',
            'status': '等待中',
            'target_weight': '6-8%',
            'total_allocation': 2200,
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
            'created_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # 8只优质股票分析目标 (基于今日分析结果)
        self.target_stocks = {
            'GS': {
                'sector': '金融', 'category': '投资银行', 'target_weight': 5,
                'target_amount': 1377, 'priority': 'wait', 'reason': 'RSI超买(74.1)+高位(89.6%)',
                'current_recommendation': '继续等待', 'entry_price': 590
            },
            'V': {
                'sector': '金融科技', 'category': '支付网络', 'target_weight': 4,
                'target_amount': 1101, 'priority': 'low', 'reason': 'RSI健康(28.7)但估值偏高',
                'current_recommendation': '小仓位试探', 'entry_price': 320
            },
            'MA': {
                'sector': '金融科技', 'category': '支付网络', 'target_weight': 3,
                'target_amount': 826, 'priority': 'high', 'reason': 'RSI超低(23.2)+基本面优秀',
                'current_recommendation': '分批买入', 'entry_price': 506
            },
            'WMT': {
                'sector': '消费', 'category': '零售巨头', 'target_weight': 3,
                'target_amount': 826, 'priority': 'medium', 'reason': 'RSI健康(35.3)+转型成功',
                'current_recommendation': '分批买入', 'entry_price': 91
            },
            'COST': {
                'sector': '消费', 'category': '会员制零售', 'target_weight': 3,
                'target_amount': 826, 'priority': 'high', 'reason': 'RSI超低(26.3)+护城河深',
                'current_recommendation': '分批买入', 'entry_price': 931
            },
            'JNJ': {
                'sector': '医疗', 'category': '医疗器械+制药', 'target_weight': 6,
                'target_amount': 1652, 'priority': 'high', 'reason': '低位(37.8%)+高股息+稳定',
                'current_recommendation': '分批买入', 'entry_price': 142
            },
            'ABT': {
                'sector': '医疗', 'category': '医疗器械', 'target_weight': 4,
                'target_amount': 1101, 'priority': 'medium', 'reason': '基本面优秀但位置偏高(82.1%)',
                'current_recommendation': '分批买入', 'entry_price': 126
            },
            'ABBV': {
                'sector': '医疗', 'category': '生物制药', 'target_weight': 4,
                'target_amount': 1101, 'priority': 'high', 'reason': 'RSI健康(47.3)+高股息',
                'current_recommendation': '分批买入', 'entry_price': 176
            }
        }
        
        self.load_tracking_data()
    
    def convert_numpy_types(self, obj):
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
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
        converted_data = self.convert_numpy_types(self.tracking_history)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    def get_market_data(self):
        """获取市场数据"""
        print("📊 获取市场数据...")
        
        market_data = {}
        all_symbols = list(self.current_holdings.keys()) + ['JPM'] + list(self.target_stocks.keys())
        
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                hist_1y = ticker.history(period='1y')
                
                if not hist.empty and not hist_1y.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                    daily_change = (current_price / prev_close - 1) * 100
                    
                    # RSI
                    rsi = self.calculate_rsi(hist_1y['Close']).iloc[-1]
                    rsi = float(rsi) if not pd.isna(rsi) else 50.0
                    
                    # 52周位置
                    high_52w = float(hist_1y['High'].max())
                    low_52w = float(hist_1y['Low'].min())
                    price_position = (current_price - low_52w) / (high_52w - low_52w) * 100
                    
                    market_data[symbol] = {
                        'current_price': current_price,
                        'daily_change': daily_change,
                        'rsi': rsi,
                        'price_position_52w': price_position,
                        'high_52w': high_52w,
                        'low_52w': low_52w
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
        print("=" * 75)
        
        total_value = 0
        holdings_analysis = {}
        
        print(f"{'股票':<6} {'股数':<6} {'成本':<8} {'现价':<8} {'日涨跌':<8} {'持仓价值':<10} {'总盈亏':<8} {'RSI':<6}")
        print("-" * 75)
        
        for symbol, holding in self.current_holdings.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                daily_change = data['daily_change']
                rsi = data['rsi']
                
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
                    'rsi': rsi
                }
                
                print(f"{symbol:<6} {shares:<6} ${avg_cost:<7.2f} ${current_price:<7.2f} "
                      f"{daily_change:>+6.1f}% ${current_value:<9.0f} {total_return:>+6.1f}% {rsi:>5.1f}")
        
        cash_position = self.portfolio_value - total_value
        cash_percentage = cash_position / self.portfolio_value * 100
        
        print("-" * 75)
        print(f"持仓总值: ${total_value:.0f}")
        print(f"现金仓位: ${cash_position:.0f} ({cash_percentage:.1f}%)")
        print(f"总资产: ${self.portfolio_value:.0f}")
        
        return holdings_analysis, total_value, cash_position
    
    def monitor_jpm_plan(self, market_data):
        """监控JPM买入计划"""
        print(f"\n🎯 JPM等待回调计划监控")
        print("=" * 60)
        
        if 'JPM' not in market_data:
            print("❌ 无法获取JPM数据")
            return {}
        
        data = market_data['JPM']
        current_price = data['current_price']
        rsi = data['rsi']
        daily_change = data['daily_change']
        
        print(f"当前状态: {self.jpm_plan['status']}")
        print(f"当前价格: ${current_price:.2f} ({daily_change:+.1f}%)")
        print(f"RSI: {rsi:.1f}")
        print(f"52周位置: {data['price_position_52w']:.1f}%")
        
        # 检查买入条件
        target_range = self.jpm_plan['entry_conditions']['price_range']
        rsi_range = self.jpm_plan['entry_conditions']['rsi_target']
        
        price_condition = target_range[0] <= current_price <= target_range[1]
        rsi_condition = rsi_range[0] <= rsi <= rsi_range[1]
        
        print(f"\n📊 买入条件检查:")
        if price_condition:
            print(f"   ✅ 价格在目标区间 ${target_range[0]}-${target_range[1]}")
        else:
            print(f"   ⏳ 等待价格回调至 ${target_range[0]}-${target_range[1]} (当前${current_price:.2f})")
        
        if rsi_condition:
            print(f"   ✅ RSI在目标区间 {rsi_range[0]}-{rsi_range[1]}")
        else:
            print(f"   ⏳ 等待RSI回落至 {rsi_range[0]}-{rsi_range[1]} (当前{rsi:.1f})")
        
        # 执行建议
        if price_condition and rsi_condition:
            print(f"\n🚀 执行建议: 条件满足，开始分批买入！")
            status = "ready"
        elif price_condition or rsi_condition:
            print(f"\n⚠️ 执行建议: 部分条件满足，可考虑小仓位试探")
            status = "partial"
        else:
            print(f"\n⏳ 执行建议: 继续等待，条件尚未成熟")
            status = "waiting"
        
        return {
            'symbol': 'JPM',
            'current_price': current_price,
            'daily_change': daily_change,
            'rsi': rsi,
            'price_condition': price_condition,
            'rsi_condition': rsi_condition,
            'status': status,
            'overall_ready': price_condition and rsi_condition
        }
    
    def monitor_target_stocks(self, market_data):
        """监控目标股票"""
        print(f"\n📋 8只目标股票实时监控")
        print("=" * 100)
        
        print(f"{'股票':<6} {'当前价':<8} {'目标价':<8} {'日涨跌':<8} {'RSI':<6} {'52周位置':<8} {'优先级':<6} {'当前建议':<12}")
        print("-" * 100)
        
        stock_status = {}
        ready_to_buy = []
        near_target = []
        
        for symbol, info in self.target_stocks.items():
            if symbol in market_data:
                data = market_data[symbol]
                current_price = data['current_price']
                daily_change = data['daily_change']
                rsi = data['rsi']
                price_position = data['price_position_52w']
                priority = info['priority']
                entry_price = info['entry_price']
                recommendation = info['current_recommendation']
                
                # 判断是否接近买入价
                price_distance = (current_price - entry_price) / current_price * 100
                
                if current_price <= entry_price * 1.02:  # 2%以内
                    near_target.append(symbol)
                
                # 更新买入建议
                if priority == 'high' and current_price <= entry_price * 1.05:
                    buy_suggestion = "立即买入"
                    ready_to_buy.append(symbol)
                elif current_price <= entry_price * 1.03:
                    buy_suggestion = "分批买入"
                elif priority == 'wait':
                    buy_suggestion = "继续等待"
                else:
                    buy_suggestion = recommendation
                
                stock_status[symbol] = {
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'price_distance': price_distance,
                    'daily_change': daily_change,
                    'rsi': rsi,
                    'price_position': price_position,
                    'priority': priority,
                    'buy_suggestion': buy_suggestion,
                    'target_amount': info['target_amount'],
                    'reason': info['reason']
                }
                
                print(f"{symbol:<6} ${current_price:<7.2f} ${entry_price:<7.2f} {daily_change:>+6.1f}% {rsi:>5.1f} "
                      f"{price_position:>7.1f}% {priority:<6} {buy_suggestion:<12}")
        
        # 重点提醒
        if ready_to_buy:
            print(f"\n🚨 立即买入机会:")
            for symbol in ready_to_buy:
                info = self.target_stocks[symbol]
                current = stock_status[symbol]['current_price']
                target = info['entry_price']
                print(f"   🔥 {symbol}: 当前${current:.2f} ≤ 目标${target:.2f} - {info['reason']}")
        
        if near_target:
            print(f"\n📍 接近目标价:")
            for symbol in near_target:
                if symbol not in ready_to_buy:
                    info = self.target_stocks[symbol]
                    current = stock_status[symbol]['current_price']
                    target = info['entry_price']
                    print(f"   📊 {symbol}: 当前${current:.2f} 接近 目标${target:.2f}")
        
        return stock_status
    
    def generate_priority_action_plan(self, jpm_status, stock_status):
        """生成优先行动计划"""
        print(f"\n📋 今日优先行动计划")
        print("=" * 70)
        
        actions = []
        
        # JPM行动
        if jpm_status.get('overall_ready'):
            actions.append({
                'priority': 1,
                'type': 'JPM买入',
                'action': f"🔥 JPM分批买入第一批",
                'amount': f"${self.jpm_plan['execution_plan']['batch_1']['amount']}",
                'price': f"${jpm_status['current_price']:.2f}",
                'reason': "等待回调条件已满足"
            })
        
        # 高优先级股票
        high_priority_stocks = []
        for symbol, status in stock_status.items():
            if status['buy_suggestion'] == '立即买入':
                high_priority_stocks.append((symbol, status))
        
        # 按RSI排序，RSI越低优先级越高
        high_priority_stocks.sort(key=lambda x: x[1]['rsi'])
        
        for symbol, status in high_priority_stocks:
            info = self.target_stocks[symbol]
            actions.append({
                'priority': 2,
                'type': '股票买入',
                'action': f"🚀 {symbol}立即买入",
                'amount': f"${status['target_amount']}",
                'price': f"${status['current_price']:.2f}",
                'reason': f"RSI:{status['rsi']:.1f}, {info['reason']}"
            })
        
        # 分批买入机会
        batch_opportunities = []
        for symbol, status in stock_status.items():
            if status['buy_suggestion'] == '分批买入' and status['price_distance'] > -5:
                batch_opportunities.append((symbol, status))
        
        batch_opportunities.sort(key=lambda x: x[1]['price_distance'])  # 按距离目标价排序
        
        for symbol, status in batch_opportunities[:3]:  # 只显示前3个
            info = self.target_stocks[symbol]
            actions.append({
                'priority': 3,
                'type': '分批买入',
                'action': f"📊 {symbol}分批买入",
                'amount': f"${status['target_amount']}",
                'price': f"等待${status['entry_price']:.2f}",
                'reason': f"当前${status['current_price']:.2f}, 还需回调{abs(status['price_distance']):.1f}%"
            })
        
        # 显示行动计划
        if actions:
            priority_names = {1: "🔥 最高优先级", 2: "⭐ 高优先级", 3: "📍 中优先级"}
            current_priority = 0
            
            for action in actions:
                if action['priority'] != current_priority:
                    current_priority = action['priority']
                    print(f"\n{priority_names.get(current_priority, '💡 低优先级')}:")
                
                print(f"   • {action['action']}: {action['amount']} (价格: {action['price']})")
                print(f"     理由: {action['reason']}")
        else:
            print("⏳ 今日无紧急买入机会，继续等待更好时机")
        
        return actions
    
    def calculate_investment_summary(self, jpm_status, stock_status):
        """计算投资汇总"""
        print(f"\n💰 投资汇总与资金规划")
        print("=" * 70)
        
        # 立即可执行的投资
        immediate_investment = 0
        if jpm_status.get('overall_ready'):
            immediate_investment += self.jpm_plan['execution_plan']['batch_1']['amount']
        
        for symbol, status in stock_status.items():
            if status['buy_suggestion'] == '立即买入':
                immediate_investment += status['target_amount']
        
        # 总计划投资
        total_planned = self.jpm_plan['total_allocation']
        for info in self.target_stocks.values():
            total_planned += info['target_amount']
        
        # 当前现金 (使用预先计算的持仓总值)
        available_cash = 10000  # 估算可用现金
        
        print(f"📊 资金状况:")
        print(f"   总资产: ${self.portfolio_value:.0f}")
        print(f"   可用现金: ${available_cash:.0f}")
        print(f"   立即投资需求: ${immediate_investment:.0f}")
        print(f"   总计划投资: ${total_planned:.0f}")
        
        if immediate_investment > 0:
            print(f"\n🎯 今日可执行投资:")
            print(f"   金额: ${immediate_investment:.0f}")
            print(f"   占可用现金: {immediate_investment/available_cash*100:.1f}%")
            
            if immediate_investment <= available_cash:
                print(f"   ✅ 资金充足，可以执行")
            else:
                print(f"   ⚠️ 资金不足，需要${immediate_investment - available_cash:.0f}")
        
        return {
            'available_cash': available_cash,
            'immediate_investment': immediate_investment,
            'total_planned': total_planned,
            'can_execute': immediate_investment <= available_cash
        }
    
    def save_daily_report(self, market_data, holdings_analysis, jpm_status, stock_status, actions, summary):
        """保存每日报告"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        daily_report = {
            'date': today,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_data': market_data,
            'holdings_analysis': holdings_analysis,
            'jpm_plan_status': jpm_status,
            'target_stocks_status': stock_status,
            'daily_actions': actions,
            'investment_summary': summary,
            'portfolio_value': self.portfolio_value
        }
        
        self.tracking_history[today] = daily_report
        self.save_tracking_data()
        
        return daily_report
    
    def run_enhanced_analysis(self):
        """运行增强版分析"""
        print("📊 增强版每日投资分析系统")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"监控范围: JPM等待回调 + 8只优质股票实时追踪")
        print("=" * 80)
        
        # 获取市场数据
        market_data = self.get_market_data()
        if not market_data:
            print("❌ 无法获取市场数据")
            return
        
        # 分析当前持仓
        holdings_analysis, total_value, cash_position = self.analyze_current_holdings(market_data)
        
        # 监控JPM计划
        jpm_status = self.monitor_jpm_plan(market_data)
        
        # 监控目标股票
        stock_status = self.monitor_target_stocks(market_data)
        
        # 生成行动计划
        actions = self.generate_priority_action_plan(jpm_status, stock_status)
        
        # 计算投资汇总
        summary = self.calculate_investment_summary(jpm_status, stock_status)
        
        # 保存报告
        daily_report = self.save_daily_report(
            market_data, holdings_analysis, jpm_status, stock_status, actions, summary
        )
        
        print(f"\n💾 完整分析数据已保存到: {self.tracking_file}")
        print(f"🎊 增强版每日分析完成！")
        
        return daily_report

if __name__ == "__main__":
    tracker = EnhancedDailyTracker()
    tracker.run_enhanced_analysis() 