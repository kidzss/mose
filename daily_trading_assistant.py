#!/usr/bin/env python3
"""
每日交易助手 - 使用data模块接口
提供投资组合分析、股票筛选和市场洞察
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_interface import DataInterface
from strategy.tdi_strategy import TDIStrategy

class DailyTradingAssistant:
    def __init__(self):
        """初始化交易助手"""
        try:
            self.data_interface = DataInterface()
            self.tdi_strategy = TDIStrategy()
            print("✅ 每日交易助手初始化成功")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def get_available_stocks(self):
        """获取所有可用股票代码 - 使用data模块接口"""
        try:
            symbols = self.data_interface.get_available_symbols()
            print(f"📊 从数据库获取到 {len(symbols)} 只可用股票")
            return symbols
        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return []
    
    def analyze_portfolio(self):
        """分析投资组合"""
        print("\n📊 投资组合分析")
        print("="*40)
        
        # 从统一配置文件加载实际投资组合
        try:
            from utils.portfolio_config_loader import get_portfolio_config
            config_loader = get_portfolio_config()
            positions = config_loader.get_positions()
            
            portfolio = {}
            for symbol, info in positions.items():
                portfolio[symbol] = {
                    'shares': info.get('shares', 0),
                    'cost_basis': info.get('cost_basis', 0)
                }
            print("✅ 已从统一配置文件加载最新持仓信息")
        except Exception as e:
            print(f"⚠️  加载统一配置失败，使用默认配置: {e}")
            # 保留最新的持仓信息作为后备
            portfolio = {
                'AMD': {'shares': 48, 'cost_basis': 126.214},
                'GOOGL': {'shares': 34, 'cost_basis': 170.54},
                'PFE': {'shares': 80, 'cost_basis': 25.899},
                'NVDA': {'shares': 40, 'cost_basis': 138.843},
                'TSLA': {'shares': 4, 'cost_basis': 254.096},
                'EOG': {'shares': 5, 'cost_basis': 122.119}
            }
        
        current_prices = {}
        total_cost = 0
        total_value = 0
        
        # 获取实时价格
        for symbol in portfolio.keys():
            try:
                latest_data = self.data_interface.get_latest_data(symbol, n_bars=1)
                if not latest_data.empty:
                    current_prices[symbol] = latest_data['close'].iloc[-1]
            except Exception as e:
                print(f"⚠️  {symbol}: 获取价格失败")
        
        # 分析每只股票
        recommendations = []
        for symbol, info in portfolio.items():
            if symbol in current_prices:
                cost = info['shares'] * info['cost_basis']
                value = info['shares'] * current_prices[symbol]
                pnl = value - cost
                pnl_pct = (pnl / cost) * 100
                
                total_cost += cost
                total_value += value
                
                # 获取综合分析
                analysis = self._get_comprehensive_analysis(symbol)
                
                print(f"{symbol:5}: ${current_prices[symbol]:7.2f} | "
                      f"盈亏{pnl_pct:+6.2f}% | {analysis['signal']} | {analysis['fundamental']}")
                
                # 生成建议
                if pnl_pct < -10:
                    recommendations.append(f"⚠️  {symbol}: 亏损较大，考虑止损或加仓")
                elif pnl_pct > 20:
                    recommendations.append(f"🎯 {symbol}: 收益良好，考虑部分止盈")
                
                # 基于综合分析的额外建议
                if analysis['score'] < -0.5 and pnl_pct > 0:
                    recommendations.append(f"📊 {symbol}: 技术+基本面转弱，考虑减仓")
                elif analysis['score'] > 0.5 and pnl_pct < 0:
                    recommendations.append(f"💡 {symbol}: 技术+基本面良好，逢低可加仓")
        
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
        
        # 候选股票池 - 从数据库中的前50只股票
        try:
            available_stocks = self.get_available_stocks()
            candidates = available_stocks[:50] if available_stocks else []
        except:
            candidates = [
                'AAPL', 'MSFT', 'AMZN', 'NFLX', 'META',
                'CRM', 'INTC', 'ORCL', 'IBM', 'QCOM'
            ]
        
        opportunities = []
        for symbol in candidates:
            try:
                analysis = self._get_comprehensive_analysis(symbol)
                
                # 综合评分 > 0.3 才推荐
                if analysis['score'] > 0.3:
                    latest_data = self.data_interface.get_latest_data(symbol, n_bars=1)
                    if not latest_data.empty:
                        price = latest_data['close'].iloc[-1]
                        opportunities.append((symbol, price, f"{analysis['signal']} | {analysis['fundamental']}", analysis['score']))
            except:
                continue
        
        if opportunities:
            print("📈 发现投资机会:")
            # 按评分排序
            opportunities.sort(key=lambda x: x[3], reverse=True)
            for symbol, price, analysis, score in opportunities:
                print(f"   {symbol}: ${price:.2f} - {analysis} (评分:{score:.2f})")
        else:
            print("   暂无明显投资机会")
        
        return opportunities
    
    def get_daily_strategy_signals(self):
        """获取今日策略信号"""
        print("\n🎯 今日策略信号")
        print("="*40)
        
        # 组合股票 + 主要指数 - 从统一配置获取
        try:
            from utils.portfolio_config_loader import get_portfolio_config
            config_loader = get_portfolio_config()
            portfolio_symbols = config_loader.get_portfolio_symbols()
        except:
            # 后备方案
            portfolio_symbols = ['AMD', 'GOOGL', 'PFE', 'NVDA', 'TSLA', 'EOG']
            
        signals = {}
        
        for symbol in portfolio_symbols + ['SPY', 'QQQ']:
            analysis = self._get_comprehensive_analysis(symbol)
            signals[symbol] = analysis
            print(f"   {symbol:5}: {analysis['signal']} | {analysis['fundamental']} (评分:{analysis['score']:.2f})")
        
        return signals
    
    def _get_comprehensive_analysis(self, symbol):
        """获取综合分析（技术信号+基本面）"""
        try:
            # 获取价格数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            hist = self.data_interface.get_historical_data(symbol, start_date, end_date)
            
            if hist.empty:
                return {"signal": "无数据", "fundamental": "无数据", "score": 0}
            
            # 1. 技术信号分析
            signals_data = self.tdi_strategy.generate_signals(hist)
            
            if 'signal' in signals_data.columns and not signals_data.empty:
                latest_signal = signals_data['signal'].iloc[-1]
                if latest_signal > 0:
                    tech_signal = "🟢 买入"
                    tech_score = 1
                elif latest_signal < 0:
                    tech_signal = "🔴 卖出"
                    tech_score = -1
                else:
                    tech_signal = "⚪ 观望"
                    tech_score = 0
            else:
                tech_signal = "无信号"
                tech_score = 0
            
            # 2. 基本面简化评估（待接入基本面数据源）
            fundamental_score = 3  # 默认中性评分
            fundamental_status = "🟡 待评估"
            
            # 综合评分
            total_score = tech_score + (fundamental_score - 3) * 0.5  # 基本面影响权重较小
            
            return {
                "signal": tech_signal,
                "fundamental": fundamental_status,
                "score": total_score
            }
                
        except Exception as e:
            return {"signal": f"分析失败", "fundamental": "无数据", "score": 0}
    
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