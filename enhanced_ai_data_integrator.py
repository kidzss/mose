#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强AI数据集成器 - 使用每日持股分析结果
Enhanced AI Data Integrator - Using Daily Holdings Analysis Results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedAIDataIntegrator:
    def __init__(self):
        """初始化数据集成器"""
        self.portfolio_config = self.load_portfolio_config()
        self.daily_analyzer = None
        
    def load_portfolio_config(self):
        """加载投资组合配置"""
        try:
            with open('portfolio_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 无法读取portfolio_config.json: {e}")
            return {}
    
    def get_daily_holdings_analysis(self):
        """获取每日持股分析结果"""
        try:
            # 导入每日持股分析器
            from daily_holdings_analysis import DailyHoldingsAnalyzer
            
            analyzer = DailyHoldingsAnalyzer()
            
            # 获取所有需要的股票数据
            all_symbols = list(analyzer.portfolio.keys()) + analyzer.market_indices + analyzer.watchlist
            all_symbols = list(set(all_symbols))  # 去重
            
            data = analyzer.get_today_data(all_symbols)
            
            if not data:
                return "无法获取市场数据"
            
            # 分析投资组合表现
            portfolio_analysis = analyzer.analyze_portfolio_performance(data)
            
            # 分析市场环境
            market_analysis = self._format_market_analysis(data)
            
            # 生成交易信号
            trading_signals = self._format_trading_signals(data, analyzer)
            
            # 组合分析结果
            analysis_summary = {
                'portfolio_analysis': portfolio_analysis,
                'market_analysis': market_analysis,
                'trading_signals': trading_signals,
                'current_data': data
            }
            
            return analysis_summary
            
        except Exception as e:
            return f"每日持股分析失败: {e}"
    
    def _format_market_analysis(self, data):
        """格式化市场分析"""
        market_summary = []
        
        # 主要指数表现
        indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        for index in indices:
            if index in data:
                index_data = data[index]
                market_summary.append(f"{index}: {index_data['change_pct']:+.2f}%")
        
        # VIX恐慌指数分析
        vix_analysis = ""
        if '^VIX' in data:
            vix_value = data['^VIX']['price']
            if vix_value < 15:
                vix_analysis = "市场恐慌情绪低，风险偏好较高"
            elif vix_value < 25:
                vix_analysis = "市场恐慌情绪正常"
            else:
                vix_analysis = "市场恐慌情绪较高，需要谨慎"
        
        return {
            'indices_performance': market_summary,
            'vix_analysis': vix_analysis,
            'vix_value': data.get('^VIX', {}).get('price', 0)
        }
    
    def _format_trading_signals(self, data, analyzer):
        """格式化交易信号"""
        signals = []
        
        for symbol in analyzer.portfolio:
            if symbol in data:
                stock_data = data[symbol]
                rsi = stock_data['rsi']
                change_pct = stock_data['change_pct']
                position_52w = stock_data['position_52w']
                
                # 技术状态评估
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
                if rsi > 70:
                    suggestion = "考虑减仓或设置止损"
                elif rsi < 30:
                    suggestion = "技术面支持持有或加仓"
                elif position_52w > 80:
                    suggestion = "接近年高，谨慎操作"
                elif position_52w < 20:
                    suggestion = "接近年低，关注机会"
                else:
                    suggestion = "维持当前仓位"
                
                signals.append({
                    'symbol': symbol,
                    'price': stock_data['price'],
                    'change_pct': change_pct,
                    'rsi': rsi,
                    'tech_status': tech_status,
                    'suggestion': suggestion,
                    'position_52w': position_52w
                })
        
        return signals
    
    def get_comprehensive_data_for_ai(self, symbol=None):
        """获取综合数据用于AI分析"""
        print("📊 获取每日持股分析数据...")
        
        # 获取每日持股分析结果
        daily_analysis = self.get_daily_holdings_analysis()
        
        if isinstance(daily_analysis, str):
            return f"数据获取失败: {daily_analysis}"
        
        # 构建AI输入数据
        ai_input = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'daily_analysis': daily_analysis,
            'portfolio_config': self.portfolio_config
        }
        
        return ai_input
    
    def format_ai_prompt_data(self, ai_input):
        """格式化AI提示数据"""
        if isinstance(ai_input, str):
            return ai_input
        
        daily_analysis = ai_input['daily_analysis']
        portfolio_config = ai_input['portfolio_config']
        
        # 构建结构化提示
        prompt_data = {
            'market_environment': daily_analysis['market_analysis'],
            'portfolio_status': daily_analysis['portfolio_analysis'],
            'trading_signals': daily_analysis['trading_signals'],
            'current_prices': daily_analysis['current_data']
        }
        
        return prompt_data

def main():
    """测试数据集成器"""
    integrator = EnhancedAIDataIntegrator()
    
    print("🧪 测试增强AI数据集成器...")
    
    # 获取综合数据
    ai_data = integrator.get_comprehensive_data_for_ai()
    
    if isinstance(ai_data, str):
        print(f"❌ {ai_data}")
        return
    
    # 格式化提示数据
    prompt_data = integrator.format_ai_prompt_data(ai_data)
    
    print("✅ 数据集成成功!")
    print(f"📊 市场环境: {prompt_data['market_environment']}")
    print(f"💼 投资组合状态: {len(prompt_data['portfolio_status'])} 个持仓")
    print(f"🎯 交易信号: {len(prompt_data['trading_signals'])} 个信号")
    print(f"📈 当前价格数据: {len(prompt_data['current_prices'])} 个股票")

if __name__ == "__main__":
    main() 