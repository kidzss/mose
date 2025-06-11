#!/usr/bin/env python3
"""
日常交易助手 - 边用边优化实施方案
整合策略分析、投资组合监控、股票筛选
"""

import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DailyTradingAssistant:
    """日常交易助手"""
    
    def __init__(self):
        """初始化"""
        from monitor.enhanced_portfolio_advisor import EnhancedPortfolioAdvisor
        from monitor.enhanced_stock_screener import EnhancedStockScreener
        from strategy.strategy_factory import StrategyFactory
        
        self.portfolio_advisor = EnhancedPortfolioAdvisor()
        self.stock_screener = EnhancedStockScreener()
        self.strategy_factory = StrategyFactory()
        
        # 你的实际投资组合
        self.portfolio = {
            'AMD': {'shares': 48, 'cost_basis': 126.214},
            'GOOGL': {'shares': 34, 'cost_basis': 170.54},
            'PFE': {'shares': 80, 'cost_basis': 25.899},
            'NVDA': {'shares': 40, 'cost_basis': 138.843},
            'TSLA': {'shares': 8, 'cost_basis': 254.096},
            'ADBE': {'shares': 5, 'cost_basis': 346.896}
        }
        
        print("✅ 日常交易助手初始化完成")
    
    def analyze_portfolio(self):
        """分析投资组合"""
        print("\n📊 投资组合分析")
        print("="*40)
        
        current_prices = {}
        total_cost = 0
        total_value = 0
        
        # 获取实时价格
        for symbol in self.portfolio.keys():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                if not hist.empty:
                    current_prices[symbol] = hist['Close'].iloc[-1]
            except Exception as e:
                print(f"⚠️  {symbol}: 获取价格失败")
        
        # 分析每只股票
        recommendations = []
        for symbol, info in self.portfolio.items():
            if symbol in current_prices:
                cost = info['shares'] * info['cost_basis']
                value = info['shares'] * current_prices[symbol]
                pnl = value - cost
                pnl_pct = (pnl / cost) * 100
                
                total_cost += cost
                total_value += value
                
                # 获取策略信号
                signal = self._get_strategy_signal(symbol)
                
                print(f"{symbol:5}: ${current_prices[symbol]:7.2f} | "
                      f"盈亏{pnl_pct:+6.2f}% | {signal}")
                
                # 生成建议
                if pnl_pct < -10:
                    recommendations.append(f"⚠️  {symbol}: 亏损较大，考虑止损或加仓")
                elif pnl_pct > 20:
                    recommendations.append(f"🎯 {symbol}: 收益良好，考虑部分止盈")
        
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        print(f"\n总盈亏: ${total_pnl:+,.0f} ({total_pnl_pct:+.2f}%)")
        
        if recommendations:
            print("\n💡 投资建议:")
            for rec in recommendations:
                print(f"   {rec}")
        
        return total_pnl_pct
    
    def screen_opportunities(self):
        """筛选投资机会"""
        print("\n🔍 筛选投资机会")
        print("="*40)
        
        # 候选股票池
        candidates = [
            'AAPL', 'MSFT', 'AMZN', 'NFLX', 'META',
            'CRM', 'INTC', 'ORCL', 'IBM', 'QCOM'
        ]
        
        opportunities = []
        for symbol in candidates:
            try:
                signal = self._get_strategy_signal(symbol)
                if "买入" in signal:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                        opportunities.append((symbol, price, signal))
            except:
                continue
        
        if opportunities:
            print("📈 发现投资机会:")
            for symbol, price, signal in opportunities:
                print(f"   {symbol}: ${price:.2f} - {signal}")
        else:
            print("   暂无明显投资机会")
        
        return opportunities
    
    def get_daily_strategy_signals(self):
        """获取今日策略信号"""
        print("\n🎯 今日策略信号")
        print("="*40)
        
        signals = {}
        for symbol in list(self.portfolio.keys()) + ['SPY', 'QQQ']:
            signal = self._get_strategy_signal(symbol)
            signals[symbol] = signal
            print(f"   {symbol:5}: {signal}")
        
        return signals
    
    def _get_strategy_signal(self, symbol):
        """获取策略信号（简化版）"""
        try:
            # 获取价格数据
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d')
            
            if hist.empty:
                return "无数据"
            
            # 使用TDI策略分析
            tdi_strategy = self.strategy_factory.create_strategy('TDI')
            
            # 标准化数据格式
            data = hist.copy()
            data.columns = [col.lower() for col in data.columns]
            
            # 生成信号
            signals_data = tdi_strategy.generate_signals(data)
            
            if 'signal' in signals_data.columns:
                latest_signal = signals_data['signal'].iloc[-1]
                if latest_signal > 0:
                    return "🟢 买入信号"
                elif latest_signal < 0:
                    return "🔴 卖出信号"
                else:
                    return "⚪ 观望"
            else:
                return "无信号"
                
        except Exception as e:
            return f"分析失败: {str(e)[:20]}"
    
    def save_daily_log(self, portfolio_pnl, opportunities, signals):
        """保存日常记录"""
        log_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_pnl_pct': portfolio_pnl,
            'opportunities': len(opportunities),
            'signals': signals
        }
        
        # 保存到文件
        try:
            with open('daily_trading_log.json', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
            print("\n📝 日常记录已保存")
        except Exception as e:
            print(f"⚠️  保存记录失败: {e}")
    
    def run_daily_analysis(self):
        """运行日常分析"""
        print("🚀 开始今日分析")
        print("="*50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 分析投资组合
        portfolio_pnl = self.analyze_portfolio()
        
        # 筛选机会
        opportunities = self.screen_opportunities()
        
        # 获取策略信号
        signals = self.get_daily_strategy_signals()
        
        # 保存记录
        self.save_daily_log(portfolio_pnl, opportunities, signals)
        
        print(f"\n✅ 今日分析完成！")
        print(f"💡 建议: 基于以上分析制定今日交易计划")

def main():
    """主函数"""
    try:
        assistant = DailyTradingAssistant()
        assistant.run_daily_analysis()
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 