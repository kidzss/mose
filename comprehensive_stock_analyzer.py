#!/usr/bin/env python3
"""
综合股票分析器 - 多重验证投资决策框架
避免单一指标盲从，建立多层次确认机制
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveStockAnalyzer:
    """综合股票分析器"""
    
    def __init__(self, symbol):
        """初始化"""
        from strategy.strategy_factory import StrategyFactory
        
        self.symbol = symbol.upper()
        self.strategy_factory = StrategyFactory()
        
        print(f"🎯 {self.symbol} 综合分析器初始化完成")
    
    def get_multi_timeframe_data(self):
        """获取多时间框架数据"""
        print(f"\n📡 获取 {self.symbol} 多时间框架数据...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            
            # 获取基本信息
            info = ticker.info
            
            # 多时间框架数据
            data = {
                'info': info,
                'daily_1y': ticker.history(period='1y'),
                'daily_6m': ticker.history(period='6mo'),
                'daily_3m': ticker.history(period='3mo'),
                'daily_1m': ticker.history(period='1mo')
            }
            
            print(f"✅ 数据获取完成")
            return data
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return None
    
    def analyze_multiple_strategies(self, data):
        """多策略分析验证"""
        print(f"\n📊 多策略技术分析验证")
        print("="*45)
        
        strategies = ['TDI', 'CPGW', 'NiuniuV3']
        timeframes = ['daily_6m', 'daily_3m', 'daily_1m']
        
        analysis_results = {}
        
        for timeframe in timeframes:
            print(f"\n⏰ {timeframe} 时间框架:")
            
            timeframe_results = {}
            for strategy_name in strategies:
                try:
                    strategy = self.strategy_factory.create_strategy(strategy_name)
                    
                    # 标准化数据
                    analysis_data = data[timeframe].copy()
                    analysis_data.columns = [col.lower() for col in analysis_data.columns]
                    
                    # 生成信号
                    result = strategy.generate_signals(analysis_data)
                    
                    if 'signal' in result.columns:
                        signals = result['signal'].dropna()
                        if len(signals) > 0:
                            latest_signal = signals.iloc[-1]
                            recent_signals = signals.tail(5).mean()  # 最近5个信号的平均
                            
                            timeframe_results[strategy_name] = {
                                'latest_signal': latest_signal,
                                'recent_avg': recent_signals,
                                'strength': abs(latest_signal),
                                'consistency': abs(recent_signals)
                            }
                            
                            signal_text = "🟢 买入" if latest_signal > 0 else "🔴 卖出" if latest_signal < 0 else "⚪ 中性"
                            consistency_text = "📈 一致" if abs(recent_signals) > 0.3 else "📊 波动"
                            
                            print(f"   {strategy_name:8}: {signal_text} | 强度:{abs(latest_signal):.3f} | {consistency_text}")
                        else:
                            timeframe_results[strategy_name] = {'latest_signal': 0, 'recent_avg': 0, 'strength': 0, 'consistency': 0}
                            print(f"   {strategy_name:8}: ⚪ 无信号数据")
                    else:
                        timeframe_results[strategy_name] = {'latest_signal': 0, 'recent_avg': 0, 'strength': 0, 'consistency': 0}
                        print(f"   {strategy_name:8}: ❌ 无信号列")
                        
                except Exception as e:
                    timeframe_results[strategy_name] = {'latest_signal': 0, 'recent_avg': 0, 'strength': 0, 'consistency': 0}
                    print(f"   {strategy_name:8}: ❌ 分析失败")
            
            analysis_results[timeframe] = timeframe_results
        
        return analysis_results
    
    def analyze_fundamental_factors(self, data):
        """基本面分析"""
        print(f"\n📈 基本面分析")
        print("="*45)
        
        try:
            info = data['info']
            
            # 关键指标
            factors = {}
            
            # 估值指标
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            factors['pe_ratio'] = pe_ratio
            factors['pb_ratio'] = pb_ratio
            
            # 盈利能力
            roe = info.get('returnOnEquity', 0)
            profit_margin = info.get('profitMargins', 0)
            factors['roe'] = roe
            factors['profit_margin'] = profit_margin
            
            # 成长性
            revenue_growth = info.get('revenueGrowth', 0)
            earnings_growth = info.get('earningsGrowthRate', 0)
            factors['revenue_growth'] = revenue_growth
            factors['earnings_growth'] = earnings_growth
            
            # 财务健康
            debt_to_equity = info.get('debtToEquity', 0)
            current_ratio = info.get('currentRatio', 0)
            factors['debt_to_equity'] = debt_to_equity
            factors['current_ratio'] = current_ratio
            
            # 市场表现
            beta = info.get('beta', 1.0)
            market_cap = info.get('marketCap', 0)
            factors['beta'] = beta
            factors['market_cap'] = market_cap
            
            print(f"📊 估值指标:")
            print(f"   P/E 比率: {pe_ratio:.2f}")
            print(f"   P/B 比率: {pb_ratio:.2f}")
            
            print(f"\n💰 盈利能力:")
            print(f"   ROE: {roe*100:.2f}%")
            print(f"   利润率: {profit_margin*100:.2f}%")
            
            print(f"\n📈 成长性:")
            print(f"   营收增长: {revenue_growth*100:.2f}%")
            print(f"   盈利增长: {earnings_growth*100:.2f}%")
            
            print(f"\n⚖️ 财务健康:")
            print(f"   负债权益比: {debt_to_equity:.2f}")
            print(f"   流动比率: {current_ratio:.2f}")
            
            print(f"\n🎯 市场特征:")
            print(f"   Beta: {beta:.2f}")
            print(f"   市值: ${market_cap/1e9:.1f}B")
            
            return factors
            
        except Exception as e:
            print(f"❌ 基本面分析失败: {e}")
            return {}
    
    def analyze_market_sentiment(self, data):
        """市场情绪分析"""
        print(f"\n📊 市场情绪分析")
        print("="*45)
        
        try:
            hist = data['daily_3m']
            
            # 价格动量
            current_price = hist['Close'].iloc[-1]
            price_1w = hist['Close'].iloc[-5] if len(hist) >= 5 else current_price
            price_1m = hist['Close'].iloc[-20] if len(hist) >= 20 else current_price
            
            momentum_1w = (current_price / price_1w - 1) * 100
            momentum_1m = (current_price / price_1m - 1) * 100
            
            # 波动率
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            # 交易量
            avg_volume = hist['Volume'].tail(20).mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            print(f"📈 价格动量:")
            print(f"   1周涨跌: {momentum_1w:+.2f}%")
            print(f"   1月涨跌: {momentum_1m:+.2f}%")
            
            print(f"\n📊 波动率:")
            print(f"   年化波动率: {volatility:.2f}%")
            
            print(f"\n📈 交易量:")
            print(f"   量比: {volume_ratio:.2f}")
            
            print(f"\n🎯 技术指标:")
            print(f"   RSI: {current_rsi:.2f}")
            
            sentiment = {
                'momentum_1w': momentum_1w,
                'momentum_1m': momentum_1m,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'rsi': current_rsi
            }
            
            return sentiment
            
        except Exception as e:
            print(f"❌ 市场情绪分析失败: {e}")
            return {}
    
    def calculate_confidence_score(self, strategy_results):
        """计算综合置信度得分"""
        print(f"\n🎯 综合置信度评估")
        print("="*45)
        
        # 收集所有策略信号
        all_signals = []
        strategy_count = 0
        
        for timeframe, strategies in strategy_results.items():
            for strategy_name, result in strategies.items():
                if result['latest_signal'] != 0:
                    all_signals.append(result['latest_signal'])
                    strategy_count += 1
        
        if not all_signals:
            print("⚠️ 没有有效的策略信号")
            return 0, "🔴 无信号", []
        
        # 计算信号一致性
        avg_signal = np.mean(all_signals)
        signal_std = np.std(all_signals) if len(all_signals) > 1 else 0
        
        # 一致性得分 (信号方向一致性)
        positive_signals = sum(1 for s in all_signals if s > 0)
        negative_signals = sum(1 for s in all_signals if s < 0)
        direction_consistency = max(positive_signals, negative_signals) / len(all_signals)
        
        # 强度得分
        avg_strength = np.mean([abs(s) for s in all_signals])
        
        # 综合得分计算
        confidence = (direction_consistency * 0.5 + avg_strength * 0.3 + (1 - signal_std) * 0.2) * 100
        confidence = min(100, max(0, confidence))
        
        print(f"📊 信号分析:")
        print(f"   有效信号数: {len(all_signals)}")
        print(f"   平均信号: {avg_signal:.3f}")
        print(f"   方向一致性: {direction_consistency:.2f}")
        print(f"   平均强度: {avg_strength:.3f}")
        print(f"   信号稳定性: {1-signal_std:.3f}")
        
        print(f"\n🎯 综合置信度: {confidence:.1f}%")
        
        # 置信度等级
        if confidence >= 80:
            level = "🟢 高置信度"
        elif confidence >= 60:
            level = "🟡 中等置信度"
        elif confidence >= 40:
            level = "🟠 低置信度"
        else:
            level = "🔴 极低置信度"
        
        print(f"   置信等级: {level}")
        
        details = [
            f"信号数量: {len(all_signals)}",
            f"方向一致性: {direction_consistency:.2f}",
            f"平均强度: {avg_strength:.3f}"
        ]
        
        return confidence, level, details
    
    def generate_investment_recommendation(self, confidence, level, strategy_results):
        """生成投资建议"""
        print(f"\n💡 投资建议")
        print("="*45)
        
        # 计算整体信号方向
        all_signals = []
        for timeframe, strategies in strategy_results.items():
            for strategy_name, result in strategies.items():
                if result['latest_signal'] != 0:
                    all_signals.append(result['latest_signal'])
        
        if all_signals:
            avg_signal = np.mean(all_signals)
            signal_direction = "买入" if avg_signal > 0 else "卖出"
        else:
            signal_direction = "观望"
        
        # 基于置信度给出建议
        if confidence >= 80:
            action = f"🟢 强烈推荐{signal_direction}"
            position_size = "20-30%"
        elif confidence >= 60:
            action = f"🟡 谨慎{signal_direction}"
            position_size = "10-20%"
        elif confidence >= 40:
            action = f"🟠 观望为主"
            position_size = "5-10%"
        else:
            action = "🔴 不建议操作"
            position_size = "0%"
        
        print(f"操作建议: {action}")
        print(f"建议仓位: {position_size}")
        print(f"置信度: {confidence:.1f}% ({level})")
        
        return {
            'action': action,
            'confidence': confidence,
            'level': level,
            'position_size': position_size
        }
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print(f"🚀 开始 {self.symbol} 综合分析")
        print("="*60)
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取数据
        data = self.get_multi_timeframe_data()
        if not data:
            return None
        
        # 多策略分析
        strategy_results = self.analyze_multiple_strategies(data)
        
        # 基本面分析
        fundamental_factors = self.analyze_fundamental_factors(data)
        
        # 市场情绪分析
        sentiment = self.analyze_market_sentiment(data)
        
        # 计算置信度
        confidence, level, details = self.calculate_confidence_score(strategy_results)
        
        # 生成投资建议
        recommendation = self.generate_investment_recommendation(confidence, level, strategy_results)
        
        # 保存分析报告
        report = {
            'symbol': self.symbol,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': confidence,
            'level': level,
            'recommendation': recommendation,
            'strategy_results': strategy_results,
            'fundamental_factors': fundamental_factors,
            'sentiment': sentiment
        }
        
        # 保存到文件
        filename = f"{self.symbol}_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📝 详细报告已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 报告保存失败: {e}")
        
        print(f"\n✅ {self.symbol} 综合分析完成！")
        
        return report

def analyze_stock(symbol):
    """分析单只股票"""
    analyzer = ComprehensiveStockAnalyzer(symbol)
    return analyzer.run_comprehensive_analysis()

def main():
    """主函数"""
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = input("请输入股票代码: ").strip().upper()
    
    if symbol:
        analyze_stock(symbol)
    else:
        print("❌ 请提供有效的股票代码")

if __name__ == "__main__":
    main() 