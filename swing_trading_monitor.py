import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SwingTradingMonitor:
    """波段操作实时监控器"""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.monitor_data = {}
        
    def get_real_time_data(self, symbol, period='3mo'):
        """获取实时数据"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except:
            return None
    
    def calculate_swing_indicators(self, data):
        """计算波段指标"""
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
        
        # 布林带
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # 成交量指标
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 价格位置
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # 回调幅度
        df['Drawdown_10'] = (df['Close'] - df['High'].rolling(10).max()) / df['High'].rolling(10).max()
        df['Drawdown_20'] = (df['Close'] - df['High'].rolling(20).max()) / df['High'].rolling(20).max()
        
        return df
    
    def generate_swing_signals(self, data):
        """生成波段信号"""
        latest = data.iloc[-1]
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'warnings': [],
            'score': 0
        }
        
        # 买入信号检测
        if latest['RSI'] < 35:
            signals['buy_signals'].append(f"RSI超卖 ({latest['RSI']:.1f})")
            signals['score'] += 2
            
        if latest['Close'] < latest['BB_Lower']:
            signals['buy_signals'].append("触及布林带下轨")
            signals['score'] += 2
            
        if latest['Drawdown_20'] <= -0.15:
            signals['buy_signals'].append(f"回调幅度 ({latest['Drawdown_20']:.1%})")
            signals['score'] += 3
            
        if latest['Close'] > latest['MA50']:
            signals['buy_signals'].append("价格在50日均线上方")
            signals['score'] += 1
            
        if latest['Volume_Ratio'] > 1.5:
            signals['buy_signals'].append(f"成交量放大 ({latest['Volume_Ratio']:.1f}倍)")
            signals['score'] += 2
            
        # 卖出信号检测
        if latest['RSI'] > 70:
            signals['sell_signals'].append(f"RSI超买 ({latest['RSI']:.1f})")
            signals['score'] -= 2
            
        if latest['Close'] > latest['BB_Upper']:
            signals['sell_signals'].append("触及布林带上轨")
            signals['score'] -= 2
            
        if latest['Price_Position'] > 0.9:
            signals['sell_signals'].append("价格接近近期高点")
            signals['score'] -= 1
            
        # 风险警告
        if latest['Close'] < latest['MA20']:
            signals['warnings'].append("跌破20日均线")
            
        if latest['Volume_Ratio'] < 0.5:
            signals['warnings'].append("成交量萎缩")
            
        return signals
    
    def monitor_portfolio(self):
        """监控投资组合"""
        print("🎯 波段操作实时监控")
        print("=" * 80)
        print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        all_signals = {}
        
        for symbol in self.symbols:
            print(f"📊 {symbol} 分析")
            print("-" * 50)
            
            # 获取数据
            data = self.get_real_time_data(symbol)
            if data is None:
                print(f"❌ 无法获取 {symbol} 数据")
                continue
                
            # 计算指标
            data_with_indicators = self.calculate_swing_indicators(data)
            
            # 生成信号
            signals = self.generate_swing_signals(data_with_indicators)
            all_signals[symbol] = signals
            
            # 显示当前状态
            latest = data_with_indicators.iloc[-1]
            print(f"当前价格: ${latest['Close']:.2f}")
            print(f"RSI: {latest['RSI']:.1f}")
            print(f"MA20: ${latest['MA20']:.2f} | MA50: ${latest['MA50']:.2f}")
            print(f"20日回调: {latest['Drawdown_20']:.1%}")
            print(f"成交量比率: {latest['Volume_Ratio']:.1f}x")
            
            # 信号评分
            score = signals['score']
            if score >= 5:
                status = "🟢 强烈买入"
            elif score >= 3:
                status = "🟡 考虑买入"
            elif score <= -3:
                status = "🔴 考虑卖出"
            elif score <= -5:
                status = "🔴 强烈卖出"
            else:
                status = "⚪ 中性观望"
                
            print(f"综合评分: {score} - {status}")
            
            # 具体信号
            if signals['buy_signals']:
                print("🟢 买入信号:")
                for signal in signals['buy_signals']:
                    print(f"  ✓ {signal}")
                    
            if signals['sell_signals']:
                print("🔴 卖出信号:")
                for signal in signals['sell_signals']:
                    print(f"  ✓ {signal}")
                    
            if signals['warnings']:
                print("⚠️ 风险警告:")
                for warning in signals['warnings']:
                    print(f"  ! {warning}")
            
            print()
        
        # 总结建议
        self.generate_action_recommendations(all_signals)
        
        return all_signals
    
    def generate_action_recommendations(self, all_signals):
        """生成行动建议"""
        print("💡 今日行动建议")
        print("=" * 50)
        
        buy_candidates = []
        sell_candidates = []
        watch_list = []
        
        for symbol, signals in all_signals.items():
            score = signals['score']
            if score >= 5:
                buy_candidates.append((symbol, score))
            elif score <= -3:
                sell_candidates.append((symbol, score))
            elif abs(score) >= 2:
                watch_list.append((symbol, score))
        
        if buy_candidates:
            print("🟢 建议买入:")
            buy_candidates.sort(key=lambda x: x[1], reverse=True)
            for symbol, score in buy_candidates:
                print(f"  {symbol}: 评分 {score}")
                
        if sell_candidates:
            print("\n🔴 建议卖出:")
            sell_candidates.sort(key=lambda x: x[1])
            for symbol, score in sell_candidates:
                print(f"  {symbol}: 评分 {score}")
                
        if watch_list:
            print("\n👀 重点关注:")
            for symbol, score in watch_list:
                print(f"  {symbol}: 评分 {score}")
        
        if not (buy_candidates or sell_candidates or watch_list):
            print("📊 当前无明显信号，保持现有仓位")
            
        print(f"\n⏰ 下次监控建议: {(datetime.now() + timedelta(hours=4)).strftime('%H:%M')}")
    
    def detailed_analysis(self, symbol):
        """详细分析单只股票"""
        print(f"🔍 {symbol} 详细分析")
        print("=" * 60)
        
        data = self.get_real_time_data(symbol, period='6mo')
        if data is None:
            print("❌ 无法获取数据")
            return
            
        data_with_indicators = self.calculate_swing_indicators(data)
        latest = data_with_indicators.iloc[-1]
        
        print("📊 技术指标现状:")
        print("-" * 30)
        print(f"当前价格: ${latest['Close']:.2f}")
        print(f"20日均线: ${latest['MA20']:.2f} ({'上方' if latest['Close'] > latest['MA20'] else '下方'})")
        print(f"50日均线: ${latest['MA50']:.2f} ({'上方' if latest['Close'] > latest['MA50'] else '下方'})")
        print(f"RSI: {latest['RSI']:.1f} ({'超买' if latest['RSI'] > 70 else '超卖' if latest['RSI'] < 30 else '正常'})")
        print(f"布林带位置: {'上轨' if latest['Close'] > latest['BB_Upper'] else '下轨' if latest['Close'] < latest['BB_Lower'] else '中轨'}")
        
        print(f"\n📈 价格动向:")
        print("-" * 30)
        print(f"10日内回调: {latest['Drawdown_10']:.1%}")
        print(f"20日内回调: {latest['Drawdown_20']:.1%}")
        print(f"价格相对位置: {latest['Price_Position']:.1%} (0%=低点, 100%=高点)")
        
        print(f"\n📊 成交量分析:")
        print("-" * 30)
        print(f"今日成交量: {latest['Volume']:,.0f}")
        print(f"20日均量: {latest['Volume_MA']:,.0f}")
        print(f"量比: {latest['Volume_Ratio']:.1f}x")
        
        # 生成具体建议
        signals = self.generate_swing_signals(data_with_indicators)
        
        print(f"\n💡 操作建议:")
        print("-" * 30)
        
        if signals['score'] >= 5:
            print("🟢 强烈建议买入")
            print("  建议分批建仓：")
            print("  - 第一批：当前价位买入30%")
            print("  - 第二批：若再跌5%买入40%")
            print("  - 第三批：若再跌10%买入30%")
            
        elif signals['score'] >= 3:
            print("🟡 可以考虑买入")
            print("  建议小仓位试探，等待更好时机")
            
        elif signals['score'] <= -5:
            print("🔴 强烈建议卖出")
            print("  建议立即减仓或清仓")
            
        elif signals['score'] <= -3:
            print("🟠 考虑减仓")
            print("  可以卖出部分仓位锁定利润")
            
        else:
            print("⚪ 保持观望")
            print("  暂无明确信号，维持现有仓位")
        
        return data_with_indicators

if __name__ == "__main__":
    # 你的投资组合
    portfolio_symbols = ['NVDA', 'GOOG', 'AMD', 'PFE']
    
    # 创建监控器
    monitor = SwingTradingMonitor(portfolio_symbols)
    
    print("启动波段操作监控系统...")
    print("分析你的当前持仓和候选股票\n")
    
    # 执行监控
    signals = monitor.monitor_portfolio()
    
    # 如果需要详细分析某只股票，取消下面的注释
    # print("\n" + "="*80)
    # monitor.detailed_analysis('NVDA') 