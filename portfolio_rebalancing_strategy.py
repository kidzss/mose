import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class PortfolioRebalancingStrategy:
    """投资组合再平衡策略"""
    
    def __init__(self):
        # 当前持仓
        self.current_positions = {
            'AMD': {'shares': 40, 'weight': 18.72, 'value': 5174.40, 'price': 129.36},
            'GOOG': {'shares': 30, 'weight': 18.47, 'value': 5095.20, 'price': 169.84},
            'NVDA': {'shares': 35, 'weight': 18.23, 'value': 5043.15, 'price': 144.09},
            'PFE': {'shares': 80, 'weight': 7.76, 'value': 1907.20, 'price': 23.84}
        }
        
        # 目标配置
        self.target_allocation = {
            'NVDA': {'target_weight': 15.0, 'category': '核心科技'},
            'GOOG': {'target_weight': 12.0, 'category': '核心科技'},
            'AMD': {'target_weight': 10.0, 'category': '核心科技'},
            'MSFT': {'target_weight': 10.0, 'category': '稳健科技'},
            'META': {'target_weight': 8.0, 'category': '稳健科技'},
            'TSLA': {'target_weight': 5.0, 'category': '高波动成长'},
            'PLTR': {'target_weight': 3.0, 'category': '高波动成长'},
            'JPM': {'target_weight': 8.0, 'category': '防御价值'},
            'BRK-B': {'target_weight': 8.0, 'category': '防御价值'},
            'MRK': {'target_weight': 6.0, 'category': '防御价值'},
            'PFE': {'target_weight': 8.0, 'category': '防御价值'},
            'CASH': {'target_weight': 7.0, 'category': '现金储备'}
        }
        
        self.total_assets = 27673.00
        self.available_cash = 4342.00
        
    def calculate_rebalancing_needs(self):
        """计算再平衡需求"""
        print("📊 投资组合再平衡分析")
        print("=" * 80)
        
        rebalancing_plan = {}
        
        print("🎯 目标配置 vs 当前配置:")
        print("-" * 60)
        
        for symbol, target in self.target_allocation.items():
            if symbol == 'CASH':
                continue
                
            target_value = self.total_assets * (target['target_weight'] / 100)
            current_value = self.current_positions.get(symbol, {}).get('value', 0)
            current_weight = (current_value / self.total_assets) * 100
            
            difference = target_value - current_value
            
            print(f"{symbol:6} | 目标: {target['target_weight']:4.1f}% | 当前: {current_weight:4.1f}% | 差额: ${difference:+7.0f}")
            
            rebalancing_plan[symbol] = {
                'target_value': target_value,
                'current_value': current_value,
                'difference': difference,
                'category': target['category']
            }
        
        return rebalancing_plan
    
    def design_selling_plan(self, rebalancing_plan):
        """设计卖出计划"""
        print(f"\n💰 卖出计划 (释放资金用于再配置)")
        print("=" * 60)
        
        selling_plan = {}
        total_proceeds = 0
        
        # AMD减仓：从18.72%降至10%
        amd_target = rebalancing_plan['AMD']['target_value']
        amd_current = rebalancing_plan['AMD']['current_value']
        amd_to_sell = amd_current - amd_target
        amd_shares_to_sell = int(amd_to_sell / self.current_positions['AMD']['price'])
        
        selling_plan['AMD'] = {
            'shares_to_sell': amd_shares_to_sell,
            'estimated_proceeds': amd_shares_to_sell * self.current_positions['AMD']['price'],
            'remaining_shares': 40 - amd_shares_to_sell,
            'reason': '降低科技股集中度'
        }
        
        print(f"🔴 AMD减仓:")
        print(f"  卖出: {amd_shares_to_sell}股 @ ~${self.current_positions['AMD']['price']:.2f}")
        print(f"  收益: ~${selling_plan['AMD']['estimated_proceeds']:.0f}")
        print(f"  剩余: {selling_plan['AMD']['remaining_shares']}股")
        print(f"  权重: {amd_current/self.total_assets*100:.1f}% → 10.0%")
        total_proceeds += selling_plan['AMD']['estimated_proceeds']
        
        # GOOG减仓：从18.47%降至12%
        goog_target = rebalancing_plan['GOOG']['target_value']
        goog_current = rebalancing_plan['GOOG']['current_value']
        goog_to_sell = goog_current - goog_target
        goog_shares_to_sell = int(goog_to_sell / self.current_positions['GOOG']['price'])
        
        selling_plan['GOOG'] = {
            'shares_to_sell': goog_shares_to_sell,
            'estimated_proceeds': goog_shares_to_sell * self.current_positions['GOOG']['price'],
            'remaining_shares': 30 - goog_shares_to_sell,
            'reason': '优化核心科技股权重'
        }
        
        print(f"\n🔴 GOOG减仓:")
        print(f"  卖出: {goog_shares_to_sell}股 @ ~${self.current_positions['GOOG']['price']:.2f}")
        print(f"  收益: ~${selling_plan['GOOG']['estimated_proceeds']:.0f}")
        print(f"  剩余: {selling_plan['GOOG']['remaining_shares']}股")
        print(f"  权重: {goog_current/self.total_assets*100:.1f}% → 12.0%")
        total_proceeds += selling_plan['GOOG']['estimated_proceeds']
        
        # NVDA减仓：从18.23%降至15%
        nvda_target = rebalancing_plan['NVDA']['target_value']
        nvda_current = rebalancing_plan['NVDA']['current_value']
        nvda_to_sell = nvda_current - nvda_target
        nvda_shares_to_sell = int(nvda_to_sell / self.current_positions['NVDA']['price'])
        
        selling_plan['NVDA'] = {
            'shares_to_sell': nvda_shares_to_sell,
            'estimated_proceeds': nvda_shares_to_sell * self.current_positions['NVDA']['price'],
            'remaining_shares': 35 - nvda_shares_to_sell,
            'reason': '控制单一股票风险'
        }
        
        print(f"\n🔴 NVDA减仓:")
        print(f"  卖出: {nvda_shares_to_sell}股 @ ~${self.current_positions['NVDA']['price']:.2f}")
        print(f"  收益: ~${selling_plan['NVDA']['estimated_proceeds']:.0f}")
        print(f"  剩余: {selling_plan['NVDA']['remaining_shares']}股")
        print(f"  权重: {nvda_current/self.total_assets*100:.1f}% → 15.0%")
        total_proceeds += selling_plan['NVDA']['estimated_proceeds']
        
        print(f"\n💵 减仓总收益: ${total_proceeds:.0f}")
        print(f"💰 可用投资资金: ${self.available_cash + total_proceeds:.0f}")
        
        return selling_plan, total_proceeds
    
    def get_stock_data_and_signals(self, symbols):
        """获取股票数据和技术信号"""
        stock_data = {}
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period="6mo")
                info = stock.info
                
                if len(data) > 0:
                    # 计算技术指标
                    current_price = data['Close'].iloc[-1]
                    ma20 = data['Close'].rolling(20).mean().iloc[-1]
                    ma50 = data['Close'].rolling(50).mean().iloc[-1]
                    
                    # RSI
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = (100 - (100 / (1 + rs))).iloc[-1]
                    
                    # 价格位置
                    high_52w = data['High'].rolling(252).max().iloc[-1] if len(data) >= 252 else data['High'].max()
                    low_52w = data['Low'].rolling(252).min().iloc[-1] if len(data) >= 252 else data['Low'].min()
                    price_position = (current_price - low_52w) / (high_52w - low_52w)
                    
                    stock_data[symbol] = {
                        'current_price': current_price,
                        'ma20': ma20,
                        'ma50': ma50,
                        'rsi': rsi,
                        'price_position': price_position,
                        'pe_ratio': info.get('trailingPE', 'N/A'),
                        'market_cap': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 'N/A'
                    }
                    
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                
        return stock_data
    
    def design_buying_plan(self, available_funds, rebalancing_plan):
        """设计买入计划"""
        print(f"\n🟢 买入计划 (总资金: ${available_funds:.0f})")
        print("=" * 60)
        
        # 获取目标股票的当前数据
        target_symbols = ['MSFT', 'META', 'TSLA', 'PLTR', 'JPM', 'BRK-B', 'MRK']
        stock_data = self.get_stock_data_and_signals(target_symbols)
        
        buying_plan = {}
        
        print("📊 目标股票技术分析:")
        print("-" * 50)
        
        for symbol in target_symbols:
            if symbol in stock_data:
                data = stock_data[symbol]
                
                # 评估买入信号
                buy_signal_score = 0
                signals = []
                
                if data['rsi'] < 40:
                    buy_signal_score += 2
                    signals.append("RSI超卖")
                elif data['rsi'] < 50:
                    buy_signal_score += 1
                    signals.append("RSI偏低")
                
                if data['current_price'] > data['ma50']:
                    buy_signal_score += 1
                    signals.append("趋势向上")
                
                if data['price_position'] < 0.7:
                    buy_signal_score += 1
                    signals.append("价格合理")
                
                # 买入建议
                if buy_signal_score >= 3:
                    timing = "🟢 立即买入"
                elif buy_signal_score >= 2:
                    timing = "🟡 谨慎买入"
                else:
                    timing = "🔴 等待回调"
                
                print(f"{symbol:6} | ${data['current_price']:7.2f} | RSI:{data['rsi']:5.1f} | 位置:{data['price_position']:4.1%} | {timing}")
                
                target_value = rebalancing_plan.get(symbol, {}).get('target_value', 0)
                target_shares = int(target_value / data['current_price']) if target_value > 0 else 0
                
                buying_plan[symbol] = {
                    'target_value': target_value,
                    'target_shares': target_shares,
                    'current_price': data['current_price'],
                    'buy_signal_score': buy_signal_score,
                    'timing_advice': timing,
                    'signals': signals,
                    'category': rebalancing_plan.get(symbol, {}).get('category', 'Unknown')
                }
        
        return buying_plan, stock_data
    
    def create_execution_timeline(self, selling_plan, buying_plan):
        """创建执行时间线"""
        print(f"\n📅 执行时间线 (分阶段实施)")
        print("=" * 60)
        
        print("🗓️ 第一阶段 (立即执行 - 1周内):")
        print("  减仓操作:")
        print(f"    • AMD: 卖出{selling_plan['AMD']['shares_to_sell']}股")
        print(f"    • GOOG: 卖出{selling_plan['GOOG']['shares_to_sell']}股")
        print(f"    • NVDA: 卖出{selling_plan['NVDA']['shares_to_sell']}股")
        
        print("\n  立即买入 (技术面良好):")
        immediate_buys = [symbol for symbol, data in buying_plan.items() 
                         if data['buy_signal_score'] >= 3]
        for symbol in immediate_buys:
            data = buying_plan[symbol]
            print(f"    • {symbol}: {data['target_shares']}股 @ ${data['current_price']:.2f}")
        
        print(f"\n🗓️ 第二阶段 (2-4周内):")
        print("  等待回调买入:")
        wait_buys = [symbol for symbol, data in buying_plan.items() 
                    if data['buy_signal_score'] < 3]
        for symbol in wait_buys:
            data = buying_plan[symbol]
            target_price = data['current_price'] * 0.95  # 期待5%回调
            print(f"    • {symbol}: 等待回调至${target_price:.2f}附近")
        
        print(f"\n🗓️ 第三阶段 (1-3个月内):")
        print("  防御性建仓 (等待超卖):")
        print("    • JPM: 等待回调至$270以下")
        print("    • BRK-B: 等待回调至$480以下")
        print("    • MRK: 等待回调至$75以下")
        
        print(f"\n📋 风险控制措施:")
        print("  • 每次交易后设置止损点 (-8%)")
        print("  • 分批建仓，避免一次性投入")
        print("  • 保留15%现金应对突发机会")
        print("  • 定期检查技术指标变化")
    
    def calculate_expected_portfolio(self, selling_plan, buying_plan):
        """计算预期投资组合"""
        print(f"\n🎯 预期投资组合 (完成再平衡后)")
        print("=" * 60)
        
        expected_positions = {}
        total_value = self.total_assets
        
        # 计算减仓后的持仓
        for symbol in ['AMD', 'GOOG', 'NVDA']:
            if symbol in selling_plan:
                remaining_shares = selling_plan[symbol]['remaining_shares']
                current_price = self.current_positions[symbol]['price']
                value = remaining_shares * current_price
                weight = (value / total_value) * 100
                
                expected_positions[symbol] = {
                    'shares': remaining_shares,
                    'value': value,
                    'weight': weight,
                    'category': '核心科技'
                }
        
        # PFE保持不变
        expected_positions['PFE'] = {
            'shares': self.current_positions['PFE']['shares'],
            'value': self.current_positions['PFE']['value'],
            'weight': (self.current_positions['PFE']['value'] / total_value) * 100,
            'category': '防御价值'
        }
        
        # 新买入的股票
        for symbol, data in buying_plan.items():
            if data['target_shares'] > 0:
                value = data['target_shares'] * data['current_price']
                weight = (value / total_value) * 100
                
                expected_positions[symbol] = {
                    'shares': data['target_shares'],
                    'value': value,
                    'weight': weight,
                    'category': data['category']
                }
        
        # 按权重排序显示
        sorted_positions = sorted(expected_positions.items(), 
                                key=lambda x: x[1]['weight'], reverse=True)
        
        print("📊 预期持仓结构:")
        print("-" * 50)
        
        category_totals = {}
        for symbol, data in sorted_positions:
            category = data['category']
            category_totals[category] = category_totals.get(category, 0) + data['weight']
            
            print(f"{symbol:6} | {data['weight']:5.1f}% | {data['shares']:3.0f}股 | ${data['value']:7.0f} | {category}")
        
        print(f"\n📈 按类别汇总:")
        print("-" * 30)
        for category, weight in sorted(category_totals.items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"{category:12} | {weight:5.1f}%")
        
        return expected_positions
    
    def comprehensive_strategy(self):
        """综合策略报告"""
        print("🎯 投资组合再平衡综合策略")
        print("=" * 80)
        print(f"制定时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 计算再平衡需求
        rebalancing_plan = self.calculate_rebalancing_needs()
        
        # 2. 设计卖出计划
        selling_plan, proceeds = self.design_selling_plan(rebalancing_plan)
        
        # 3. 设计买入计划
        total_funds = self.available_cash + proceeds
        buying_plan, stock_data = self.design_buying_plan(total_funds, rebalancing_plan)
        
        # 4. 创建执行时间线
        self.create_execution_timeline(selling_plan, buying_plan)
        
        # 5. 计算预期投资组合
        expected_positions = self.calculate_expected_portfolio(selling_plan, buying_plan)
        
        # 6. 风险评估和建议
        print(f"\n⚠️ 风险提示和建议")
        print("=" * 60)
        print("🔸 市场风险:")
        print("  • 科技股仍有较高权重，关注板块轮动")
        print("  • 注意美联储政策变化对成长股的影响")
        
        print("\n🔸 执行风险:")
        print("  • 分批减仓避免冲击成本")
        print("  • 买入时机要结合技术指标")
        print("  • 设置止损保护已实现收益")
        
        print("\n🔸 优化建议:")
        print("  • 考虑增加国际市场ETF分散风险")
        print("  • 适当配置债券或REITs平衡波动")
        print("  • 保持现金比例应对市场变化")
        
        return {
            'selling_plan': selling_plan,
            'buying_plan': buying_plan,
            'expected_positions': expected_positions,
            'total_funds': total_funds
        }

if __name__ == "__main__":
    strategy = PortfolioRebalancingStrategy()
    result = strategy.comprehensive_strategy() 