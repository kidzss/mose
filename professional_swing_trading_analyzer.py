import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalSwingTradingAnalyzer:
    """专业波段操作分析器"""
    
    def __init__(self):
        # 波段操作策略配置
        self.swing_strategies = {
            'user_37_strategy': {
                'name': '用户3/7策略',
                'buy_signals': {
                    'drawdown_threshold': 0.30,    # 从高点回调30%开始关注
                    'entry_points': [0.30, 0.50, 0.70],  # 分批买入点
                    'position_sizes': [0.3, 0.4, 0.3]    # 对应仓位大小
                },
                'sell_signals': {
                    'profit_threshold': 0.30,      # 盈利30%开始卖出
                    'exit_points': [0.30, 0.50, 0.70],   # 分批卖出点
                    'position_sizes': [0.2, 0.3, 0.5]    # 对应卖出比例
                }
            },
            'professional_strategy': {
                'name': '专业波段策略',
                'buy_signals': {
                    'rsi_oversold': 30,            # RSI超卖
                    'drawdown_threshold': 0.20,    # 回调20%关注
                    'moving_average_support': True, # MA支撑确认
                },
                'sell_signals': {
                    'rsi_overbought': 70,          # RSI超买
                    'profit_threshold': 0.25,      # 盈利25%关注
                    'moving_average_resistance': True # MA阻力确认
                }
            },
            'momentum_strategy': {
                'name': '动量波段策略',
                'buy_signals': {
                    'macd_bullish': True,          # MACD金叉
                    'volume_confirmation': True,    # 成交量确认
                    'bollinger_lower': True        # 布林带下轨支撑
                },
                'sell_signals': {
                    'macd_bearish': True,          # MACD死叉
                    'bollinger_upper': True,       # 布林带上轨阻力
                    'volume_divergence': True      # 成交量背离
                }
            }
        }
        
        # 技术指标参数
        self.technical_params = {
            'sma_short': 20,    # 短期均线
            'sma_long': 50,     # 长期均线
            'rsi_period': 14,   # RSI周期
            'macd_fast': 12,    # MACD快线
            'macd_slow': 26,    # MACD慢线
            'macd_signal': 9,   # MACD信号线
            'bollinger_period': 20,  # 布林带周期
            'bollinger_std': 2       # 布林带标准差
        }
        
    def get_stock_data(self, symbol, period='2y'):
        """获取股票数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if len(data) < 100:
                print(f"警告: {symbol} 数据不足")
                return None
            print(f"✓ {symbol}: 获取 {len(data)} 个交易日数据")
            return data
        except Exception as e:
            print(f"✗ {symbol}: 获取失败 - {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        df = data.copy()
        
        # 移动平均线
        df['SMA_20'] = df['Close'].rolling(window=self.technical_params['sma_short']).mean()
        df['SMA_50'] = df['Close'].rolling(window=self.technical_params['sma_long']).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.technical_params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.technical_params['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=self.technical_params['macd_fast']).mean()
        exp2 = df['Close'].ewm(span=self.technical_params['macd_slow']).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=self.technical_params['macd_signal']).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(window=self.technical_params['bollinger_period']).mean()
        bb_std = df['Close'].rolling(window=self.technical_params['bollinger_period']).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.technical_params['bollinger_std'])
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.technical_params['bollinger_std'])
        
        # 价格变动百分比
        df['Price_Change'] = df['Close'].pct_change()
        
        # 从最近高点的回撤
        df['Rolling_Max'] = df['Close'].rolling(window=50, min_periods=1).max()
        df['Drawdown'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
        
        # 成交量均线
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def implement_user_37_strategy(self, data):
        """实现用户3/7策略"""
        df = data.copy()
        
        # 初始化信号
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Position'] = 0
        df['Position_Size'] = 0.0
        
        position = 0
        entry_price = 0
        current_position_size = 0
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            drawdown = df['Drawdown'].iloc[i]
            
            # 买入逻辑 - 基于回撤幅度
            if position == 0 or position == 1:  # 未满仓
                if drawdown <= -0.30 and position == 0:  # 回调30%，第一次买入
                    df.loc[df.index[i], 'Buy_Signal'] = 1
                    position = 1
                    entry_price = current_price
                    current_position_size = 0.3
                    df.loc[df.index[i], 'Position_Size'] = current_position_size
                    
                elif drawdown <= -0.50 and position == 1:  # 回调50%，第二次买入
                    df.loc[df.index[i], 'Buy_Signal'] = 2
                    position = 2
                    # 加权平均成本
                    entry_price = (entry_price * 0.3 + current_price * 0.4) / 0.7
                    current_position_size = 0.7
                    df.loc[df.index[i], 'Position_Size'] = current_position_size
                    
                elif drawdown <= -0.70 and position == 2:  # 回调70%，第三次买入
                    df.loc[df.index[i], 'Buy_Signal'] = 3
                    position = 3
                    # 加权平均成本
                    entry_price = (entry_price * 0.7 + current_price * 0.3) / 1.0
                    current_position_size = 1.0
                    df.loc[df.index[i], 'Position_Size'] = current_position_size
            
            # 卖出逻辑 - 基于盈利幅度
            if position > 0 and entry_price > 0:
                profit_ratio = (current_price - entry_price) / entry_price
                
                if profit_ratio >= 0.30 and current_position_size >= 0.8:  # 盈利30%，开始减仓
                    df.loc[df.index[i], 'Sell_Signal'] = 1
                    current_position_size *= 0.8  # 卖出20%
                    
                elif profit_ratio >= 0.50 and current_position_size >= 0.5:  # 盈利50%，继续减仓
                    df.loc[df.index[i], 'Sell_Signal'] = 2
                    current_position_size *= 0.7  # 再卖出30%
                    
                elif profit_ratio >= 0.70 and current_position_size > 0.2:  # 盈利70%，大幅减仓
                    df.loc[df.index[i], 'Sell_Signal'] = 3
                    current_position_size *= 0.2  # 保留20%
            
            df.loc[df.index[i], 'Position'] = position
        
        return df
    
    def implement_professional_strategy(self, data):
        """实现专业波段策略"""
        df = data.copy()
        
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Position'] = 0
        
        position = 0
        entry_price = 0
        
        for i in range(50, len(df)):  # 从第50个数据点开始，确保技术指标有效
            current_price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            drawdown = df['Drawdown'].iloc[i]
            sma_20 = df['SMA_20'].iloc[i]
            sma_50 = df['SMA_50'].iloc[i]
            
            # 买入信号
            if position == 0:
                # 多重条件确认买入
                if (rsi < 30 and  # RSI超卖
                    drawdown <= -0.20 and  # 回调20%以上
                    current_price > sma_50 and  # 价格在长期均线上方
                    sma_20 > sma_50):  # 短期均线在长期均线上方
                    
                    df.loc[df.index[i], 'Buy_Signal'] = 1
                    position = 1
                    entry_price = current_price
            
            # 卖出信号
            elif position == 1:
                profit_ratio = (current_price - entry_price) / entry_price
                
                # 多重条件确认卖出
                if (rsi > 70 or  # RSI超买
                    profit_ratio >= 0.25 or  # 盈利25%
                    current_price < sma_20):  # 跌破短期均线
                    
                    df.loc[df.index[i], 'Sell_Signal'] = 1
                    position = 0
                    entry_price = 0
            
            df.loc[df.index[i], 'Position'] = position
        
        return df
    
    def calculate_strategy_performance(self, data, strategy_name):
        """计算策略表现"""
        df = data.copy()
        
        # 计算收益
        df['Strategy_Return'] = 0.0
        df['Cumulative_Return'] = 1.0
        df['Buy_and_Hold_Return'] = df['Close'] / df['Close'].iloc[0]
        
        position = 0
        entry_price = 0
        total_trades = 0
        winning_trades = 0
        
        for i in range(len(df)):
            if df['Buy_Signal'].iloc[i] > 0 and position == 0:
                entry_price = df['Close'].iloc[i]
                position = 1
                
            elif df['Sell_Signal'].iloc[i] > 0 and position == 1:
                exit_price = df['Close'].iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                df.loc[df.index[i], 'Strategy_Return'] = trade_return
                
                total_trades += 1
                if trade_return > 0:
                    winning_trades += 1
                
                position = 0
                entry_price = 0
        
        # 计算累积收益
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        
        # 计算策略指标
        total_return = df['Cumulative_Return'].iloc[-1] - 1
        buy_hold_return = df['Buy_and_Hold_Return'].iloc[-1] - 1
        
        # 计算最大回撤
        peak = df['Cumulative_Return'].expanding().max()
        strategy_drawdown = (df['Cumulative_Return'] - peak) / peak
        max_drawdown = strategy_drawdown.min()
        
        # 计算胜率
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算夏普比率
        returns = df['Strategy_Return'][df['Strategy_Return'] != 0]
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        performance = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio
        }
        
        return df, performance
    
    def analyze_swing_patterns(self, data, symbol):
        """分析波段模式"""
        df = data.copy()
        
        print(f"\n📊 {symbol} 波段模式分析:")
        print("-" * 60)
        
        # 识别高点和低点
        highs = df['Close'].rolling(window=20, center=True).max() == df['Close']
        lows = df['Close'].rolling(window=20, center=True).min() == df['Close']
        
        high_points = df[highs]['Close'].dropna()
        low_points = df[lows]['Close'].dropna()
        
        print(f"识别到 {len(high_points)} 个高点, {len(low_points)} 个低点")
        
        # 计算平均波段幅度
        if len(high_points) > 1 and len(low_points) > 1:
            # 计算高点间的平均回调
            high_to_low_changes = []
            low_to_high_changes = []
            
            for i in range(len(high_points) - 1):
                high_price = high_points.iloc[i]
                # 找到这个高点后的第一个低点
                next_lows = low_points[low_points.index > high_points.index[i]]
                if len(next_lows) > 0:
                    low_price = next_lows.iloc[0]
                    change = (low_price - high_price) / high_price
                    high_to_low_changes.append(change)
            
            for i in range(len(low_points) - 1):
                low_price = low_points.iloc[i]
                # 找到这个低点后的第一个高点
                next_highs = high_points[high_points.index > low_points.index[i]]
                if len(next_highs) > 0:
                    high_price = next_highs.iloc[0]
                    change = (high_price - low_price) / low_price
                    low_to_high_changes.append(change)
            
            if high_to_low_changes:
                avg_decline = np.mean(high_to_low_changes)
                print(f"平均回调幅度: {avg_decline:.1%}")
                
            if low_to_high_changes:
                avg_rally = np.mean(low_to_high_changes)
                print(f"平均上涨幅度: {avg_rally:.1%}")
        
        # RSI分析
        oversold_opportunities = len(df[df['RSI'] < 30])
        overbought_opportunities = len(df[df['RSI'] > 70])
        
        print(f"RSI超卖机会: {oversold_opportunities} 次")
        print(f"RSI超买机会: {overbought_opportunities} 次")
        
        return {
            'high_points': len(high_points),
            'low_points': len(low_points),
            'avg_decline': np.mean(high_to_low_changes) if high_to_low_changes else 0,
            'avg_rally': np.mean(low_to_high_changes) if low_to_high_changes else 0,
            'oversold_opportunities': oversold_opportunities,
            'overbought_opportunities': overbought_opportunities
        }
    
    def comprehensive_swing_analysis(self, symbols):
        """综合波段分析"""
        print("🎯 专业波段操作分析")
        print("=" * 80)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n分析股票: {symbol}")
            print("-" * 40)
            
            # 获取数据
            data = self.get_stock_data(symbol)
            if data is None:
                continue
            
            # 计算技术指标
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # 分析波段模式
            pattern_analysis = self.analyze_swing_patterns(data_with_indicators, symbol)
            
            # 实施用户3/7策略
            user_strategy_data, user_performance = self.calculate_strategy_performance(
                self.implement_user_37_strategy(data_with_indicators), "用户3/7策略")
            
            # 实施专业策略
            pro_strategy_data, pro_performance = self.calculate_strategy_performance(
                self.implement_professional_strategy(data_with_indicators), "专业波段策略")
            
            print(f"\n📈 {symbol} 策略表现对比:")
            print("-" * 50)
            
            strategies = [user_performance, pro_performance]
            for perf in strategies:
                print(f"\n{perf['strategy_name']}:")
                print(f"  总收益率: {perf['total_return']:+.1%}")
                print(f"  买入持有: {perf['buy_hold_return']:+.1%}")
                print(f"  超额收益: {perf['outperformance']:+.1%}")
                print(f"  最大回撤: {perf['max_drawdown']:.1%}")
                print(f"  交易次数: {perf['total_trades']}")
                print(f"  胜率: {perf['win_rate']:.1%}")
                print(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
            
            results[symbol] = {
                'pattern_analysis': pattern_analysis,
                'user_strategy': user_performance,
                'professional_strategy': pro_performance,
                'data': data_with_indicators
            }
        
        return results
    
    def generate_swing_trading_guide(self, results):
        """生成波段操作指南"""
        print(f"\n🎓 专业波段操作指南")
        print("=" * 80)
        
        print("\n📚 波段操作核心原则:")
        print("-" * 40)
        print("1. 趋势确认: 只在主趋势方向做波段")
        print("2. 技术指标组合: 不依赖单一指标")
        print("3. 分批建仓: 降低时点选择风险")
        print("4. 严格止损: 控制单次亏损幅度")
        print("5. 盈利保护: 及时锁定部分利润")
        
        print(f"\n🎯 基于你的3/7策略的改进建议:")
        print("-" * 40)
        
        # 基于分析结果给出建议
        avg_strategies_performance = []
        for symbol, result in results.items():
            avg_strategies_performance.append(result['user_strategy']['total_return'])
        
        if avg_strategies_performance:
            avg_return = np.mean(avg_strategies_performance) * 100
            
            if avg_return > 15:
                print("✅ 你的策略表现良好，建议:")
            elif avg_return > 0:
                print("⚠️ 你的策略需要优化，建议:")
            else:
                print("❌ 你的策略需要重新设计，建议:")
        
        print("  • 买入信号优化:")
        print("    - 结合RSI < 30 的超卖信号")
        print("    - 确认价格在长期均线(50日)上方")
        print("    - 等待成交量放大确认")
        
        print("  • 卖出信号优化:")
        print("    - RSI > 70 时开始关注卖出")
        print("    - 价格触及布林带上轨考虑减仓")
        print("    - MACD死叉时果断止损")
        
        print("  • 风险管理:")
        print("    - 单只股票最大亏损限制在总资金的2%")
        print("    - 设置移动止损，保护已实现利润")
        print("    - 避免在重要财报前建立大仓位")
        
        print(f"\n📊 推荐的波段操作流程:")
        print("-" * 40)
        print("1. 选股: 选择趋势向上、流动性好的股票")
        print("2. 时机: 等待技术指标给出明确信号")
        print("3. 建仓: 分2-3次建仓，控制单次风险")
        print("4. 持有: 根据技术指标调整持仓")
        print("5. 卖出: 达到目标利润或止损点及时出场")
        
        return True

if __name__ == "__main__":
    analyzer = ProfessionalSwingTradingAnalyzer()
    
    # 分析目标股票
    test_symbols = ['NVDA', 'GOOG', 'TSLA', 'JPM', 'ABBV']
    
    results = analyzer.comprehensive_swing_analysis(test_symbols)
    analyzer.generate_swing_trading_guide(results) 