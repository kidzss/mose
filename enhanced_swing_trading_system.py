import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedSwingTradingSystem:
    """增强版波段操作系统"""
    
    def __init__(self):
        self.swing_parameters = {
            # 用户3/7策略参数
            'user_strategy': {
                'drawdown_levels': [0.15, 0.25, 0.35],  # 调整为更现实的回调幅度
                'buy_weights': [0.3, 0.4, 0.3],         # 对应买入权重
                'profit_levels': [0.20, 0.35, 0.50],    # 盈利卖出点
                'sell_weights': [0.2, 0.3, 0.5]         # 对应卖出权重
            },
            
            # 专业策略参数
            'pro_strategy': {
                'rsi_oversold': 35,      # 调整RSI阈值
                'rsi_overbought': 65,
                'ma_period_short': 20,
                'ma_period_long': 50,
                'volume_threshold': 1.5,  # 成交量放大倍数
                'stop_loss': 0.08,       # 止损8%
                'take_profit': 0.25      # 止盈25%
            }
        }
    
    def get_stock_data(self, symbol, period='2y'):
        """获取股票数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if len(data) < 100:
                return None
            return data
        except:
            return None
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()
        
        # 移动平均线
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 成交量比率
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 价格变动
        df['Returns'] = df['Close'].pct_change()
        
        # 从近期高点的回撤 - 使用滚动窗口
        df['High_20'] = df['High'].rolling(20).max()
        df['Drawdown_20'] = (df['Close'] - df['High_20']) / df['High_20']
        
        # 从近期低点的反弹
        df['Low_20'] = df['Low'].rolling(20).min() 
        df['Rally_20'] = (df['Close'] - df['Low_20']) / df['Low_20']
        
        return df
    
    def implement_enhanced_user_strategy(self, data):
        """实现增强版用户3/7策略"""
        df = data.copy()
        
        # 初始化
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = 0.0
        df['Position_Size'] = 0.0
        
        current_position = 0.0  # 当前持仓比例
        entry_prices = []       # 多次买入的价格记录
        entry_weights = []      # 对应的权重
        
        params = self.swing_parameters['user_strategy']
        
        for i in range(20, len(df)):  # 从第20个数据点开始
            current_price = df['Close'].iloc[i]
            drawdown = df['Drawdown_20'].iloc[i]
            
            # 买入逻辑
            if current_position < 1.0:  # 还没满仓
                for j, level in enumerate(params['drawdown_levels']):
                    if drawdown <= -level and current_position < sum(params['buy_weights'][:j+1]):
                        # 达到买入条件且还没有在这个级别买入
                        buy_weight = params['buy_weights'][j]
                        target_position = sum(params['buy_weights'][:j+1])
                        
                        if current_position < target_position:
                            actual_buy = target_position - current_position
                            df.loc[df.index[i], 'Buy_Signal'] = j + 1
                            df.loc[df.index[i], 'Position_Size'] = actual_buy
                            
                            # 记录买入信息
                            entry_prices.append(current_price)
                            entry_weights.append(actual_buy)
                            current_position = target_position
                            break
            
            # 卖出逻辑
            if current_position > 0 and entry_prices:
                # 计算加权平均成本
                avg_cost = sum(p * w for p, w in zip(entry_prices, entry_weights)) / sum(entry_weights)
                profit_ratio = (current_price - avg_cost) / avg_cost
                
                for j, level in enumerate(params['profit_levels']):
                    if profit_ratio >= level:
                        sell_weight = params['sell_weights'][j]
                        target_position = current_position * (1 - sell_weight)
                        
                        if current_position > target_position:
                            actual_sell = current_position - target_position
                            df.loc[df.index[i], 'Sell_Signal'] = j + 1
                            df.loc[df.index[i], 'Position_Size'] = -actual_sell
                            
                            current_position = target_position
                            
                            # 如果全部卖出，清空记录
                            if current_position <= 0.01:
                                entry_prices = []
                                entry_weights = []
                                current_position = 0
                            break
            
            df.loc[df.index[i], 'Position'] = current_position
            if entry_prices:
                avg_cost = sum(p * w for p, w in zip(entry_prices, entry_weights)) / sum(entry_weights)
                df.loc[df.index[i], 'Entry_Price'] = avg_cost
        
        return df
    
    def implement_professional_strategy(self, data):
        """实现专业波段策略"""
        df = data.copy()
        
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = 0.0
        
        position = 0
        entry_price = 0
        params = self.swing_parameters['pro_strategy']
        
        for i in range(50, len(df)):
            current_price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            ma20 = df['MA20'].iloc[i]
            ma50 = df['MA50'].iloc[i]
            volume_ratio = df['Volume_Ratio'].iloc[i]
            
            # 买入信号
            if position == 0:
                # 多重确认买入
                if (rsi < params['rsi_oversold'] and
                    current_price > ma50 and
                    ma20 > ma50 and
                    volume_ratio > params['volume_threshold']):
                    
                    df.loc[df.index[i], 'Buy_Signal'] = 1
                    position = 1
                    entry_price = current_price
                    df.loc[df.index[i], 'Entry_Price'] = entry_price
            
            # 卖出信号
            elif position == 1:
                profit_loss = (current_price - entry_price) / entry_price
                
                # 止盈或止损
                if (rsi > params['rsi_overbought'] or
                    profit_loss >= params['take_profit'] or
                    profit_loss <= -params['stop_loss'] or
                    current_price < ma20):
                    
                    df.loc[df.index[i], 'Sell_Signal'] = 1
                    position = 0
                    entry_price = 0
            
            df.loc[df.index[i], 'Position'] = position
        
        return df
    
    def calculate_performance_metrics(self, data, strategy_name):
        """计算策略表现指标"""
        df = data.copy()
        
        # 计算策略收益
        df['Strategy_Return'] = 0.0
        df['Cumulative_Strategy'] = 1.0
        
        # 买入持有基准
        df['Buy_Hold'] = df['Close'] / df['Close'].iloc[0]
        
        # 模拟交易
        cash = 1.0
        position_value = 0.0
        shares = 0.0
        total_value = 1.0
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            
            # 买入
            if df['Buy_Signal'].iloc[i] > 0:
                position_size = abs(df['Position_Size'].iloc[i]) if 'Position_Size' in df.columns else 1.0
                invest_amount = cash * position_size
                new_shares = invest_amount / current_price
                shares += new_shares
                cash -= invest_amount
                
            # 卖出
            elif df['Sell_Signal'].iloc[i] > 0:
                if 'Position_Size' in df.columns:
                    sell_ratio = abs(df['Position_Size'].iloc[i])
                else:
                    sell_ratio = 1.0
                
                sell_shares = shares * sell_ratio
                cash += sell_shares * current_price
                shares -= sell_shares
            
            # 计算总价值
            position_value = shares * current_price
            total_value = cash + position_value
            df.loc[df.index[i], 'Cumulative_Strategy'] = total_value
        
        # 计算指标
        final_return = total_value - 1
        buy_hold_return = df['Buy_Hold'].iloc[-1] - 1
        
        # 最大回撤
        peak = df['Cumulative_Strategy'].expanding().max()
        drawdowns = (df['Cumulative_Strategy'] - peak) / peak
        max_drawdown = drawdowns.min()
        
        # 交易统计
        total_trades = len(df[df['Buy_Signal'] > 0])
        
        # 计算年化收益和波动率
        daily_returns = df['Cumulative_Strategy'].pct_change().dropna()
        annual_return = (total_value ** (252 / len(df))) - 1 if len(df) > 0 else 0
        annual_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # 夏普比率
        risk_free_rate = 0.04
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'strategy_name': strategy_name,
            'total_return': final_return,
            'annual_return': annual_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': final_return - buy_hold_return,
            'max_drawdown': max_drawdown,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'final_value': total_value
        }
    
    def analyze_swing_characteristics(self, data, symbol):
        """分析个股波段特征"""
        df = data.copy()
        
        print(f"\n📊 {symbol} 波段特征分析:")
        print("-" * 50)
        
        # 基本统计
        avg_daily_return = df['Returns'].mean() * 100
        daily_volatility = df['Returns'].std() * 100
        
        print(f"平均日收益: {avg_daily_return:.2f}%")
        print(f"日波动率: {daily_volatility:.2f}%")
        
        # 回调和反弹分析
        significant_drawdowns = df[df['Drawdown_20'] <= -0.15]
        significant_rallies = df[df['Rally_20'] >= 0.15]
        
        print(f"显著回调(>15%)次数: {len(significant_drawdowns)}")
        print(f"显著反弹(>15%)次数: {len(significant_rallies)}")
        
        if len(significant_drawdowns) > 0:
            avg_drawdown = significant_drawdowns['Drawdown_20'].mean()
            print(f"平均显著回调幅度: {avg_drawdown:.1%}")
        
        if len(significant_rallies) > 0:
            avg_rally = significant_rallies['Rally_20'].mean()
            print(f"平均显著反弹幅度: {avg_rally:.1%}")
        
        # RSI分析
        oversold_count = len(df[df['RSI'] <= 30])
        overbought_count = len(df[df['RSI'] >= 70])
        
        print(f"RSI超卖(<30)次数: {oversold_count}")
        print(f"RSI超买(>70)次数: {overbought_count}")
        
        # 成交量分析
        high_volume_days = len(df[df['Volume_Ratio'] >= 2.0])
        print(f"高成交量(>2倍均量)天数: {high_volume_days}")
        
        return {
            'avg_daily_return': avg_daily_return,
            'daily_volatility': daily_volatility,
            'drawdown_opportunities': len(significant_drawdowns),
            'rally_opportunities': len(significant_rallies),
            'oversold_opportunities': oversold_count,
            'overbought_opportunities': overbought_count
        }
    
    def comprehensive_swing_analysis(self, symbols):
        """综合波段分析"""
        print("🎯 增强版波段操作分析")
        print("=" * 80)
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n📈 分析股票: {symbol}")
            print("-" * 40)
            
            # 获取数据
            data = self.get_stock_data(symbol)
            if data is None:
                print(f"❌ {symbol}: 无法获取数据")
                continue
            
            print(f"✓ 获取 {len(data)} 个交易日数据")
            
            # 计算技术指标
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # 分析波段特征
            characteristics = self.analyze_swing_characteristics(data_with_indicators, symbol)
            
            # 实施用户策略
            user_data = self.implement_enhanced_user_strategy(data_with_indicators)
            user_performance = self.calculate_performance_metrics(user_data, "增强用户3/7策略")
            
            # 实施专业策略
            pro_data = self.implement_professional_strategy(data_with_indicators)
            pro_performance = self.calculate_performance_metrics(pro_data, "专业波段策略")
            
            # 输出结果
            print(f"\n📊 {symbol} 策略对比:")
            print("-" * 40)
            
            for perf in [user_performance, pro_performance]:
                print(f"\n{perf['strategy_name']}:")
                print(f"  总收益率: {perf['total_return']:+.1%}")
                print(f"  年化收益: {perf['annual_return']:+.1%}")
                print(f"  买入持有: {perf['buy_hold_return']:+.1%}")
                print(f"  超额收益: {perf['outperformance']:+.1%}")
                print(f"  最大回撤: {perf['max_drawdown']:.1%}")
                print(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
                print(f"  交易次数: {perf['total_trades']}")
            
            all_results[symbol] = {
                'characteristics': characteristics,
                'user_performance': user_performance,
                'pro_performance': pro_performance
            }
        
        # 综合分析
        self.generate_swing_trading_insights(all_results)
        
        return all_results
    
    def generate_swing_trading_insights(self, results):
        """生成波段操作洞察"""
        print(f"\n🎓 专业波段操作教学")
        print("=" * 80)
        
        # 计算平均表现
        user_returns = [r['user_performance']['total_return'] for r in results.values()]
        pro_returns = [r['pro_performance']['total_return'] for r in results.values()]
        
        if user_returns:
            avg_user_return = np.mean(user_returns)
            avg_pro_return = np.mean(pro_returns)
            
            print(f"\n📊 策略表现汇总:")
            print("-" * 40)
            print(f"增强用户3/7策略平均收益: {avg_user_return:+.1%}")
            print(f"专业波段策略平均收益: {avg_pro_return:+.1%}")
            
            better_strategy = "用户策略" if avg_user_return > avg_pro_return else "专业策略"
            print(f"表现更好的策略: {better_strategy}")
        
        print(f"\n🎯 波段操作实战指南:")
        print("-" * 40)
        
        print("1. 买入时机识别:")
        print("   ✓ 股价从近期高点回调15-35%")
        print("   ✓ RSI指标低于35")
        print("   ✓ 价格仍在长期趋势线上方")
        print("   ✓ 成交量放大确认")
        
        print("\n2. 分批建仓策略:")
        print("   ✓ 第一批：回调15%时买入30%")
        print("   ✓ 第二批：回调25%时买入40%")
        print("   ✓ 第三批：回调35%时买入30%")
        
        print("\n3. 卖出时机选择:")
        print("   ✓ 盈利20%时卖出20%")
        print("   ✓ 盈利35%时卖出30%")
        print("   ✓ 盈利50%时卖出50%")
        print("   ✓ RSI超过65考虑减仓")
        
        print("\n4. 风险管理原则:")
        print("   ⚠️ 单只股票最大亏损不超过总资金3%")
        print("   ⚠️ 跌破止损线(8%)立即止损")
        print("   ⚠️ 避免在财报前大举建仓")
        print("   ⚠️ 保持20%现金以应对机会")
        
        print("\n5. 选股标准:")
        print("   📈 选择长期趋势向上的股票")
        print("   📈 日成交额大于1000万美元")
        print("   📈 基本面良好，业绩稳定增长")
        print("   📈 避开即将公布财报的股票")
        
        print(f"\n💡 你的3/7策略优化建议:")
        print("-" * 40)
        print("✅ 优点:")
        print("   • 分批建仓降低风险")
        print("   • 分批卖出锁定利润")
        print("   • 逻辑清晰易于执行")
        
        print("\n🔧 改进建议:")
        print("   • 结合RSI等技术指标确认")
        print("   • 增加成交量条件")
        print("   • 设置明确的止损点")
        print("   • 考虑市场整体环境")
        
        return True

if __name__ == "__main__":
    swing_system = EnhancedSwingTradingSystem()
    
    # 分析组合中的股票
    test_symbols = ['NVDA', 'GOOG', 'META', 'TSLA', 'JPM']
    
    results = swing_system.comprehensive_swing_analysis(test_symbols) 