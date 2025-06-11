#!/usr/bin/env python3
"""
CRM (Salesforce) 深度投资分析报告
基于多策略分析提供具体买入建议
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

class CRMAnalyzer:
    """CRM深度分析器"""
    
    def __init__(self):
        """初始化"""
        from strategy.strategy_factory import StrategyFactory
        
        self.symbol = 'CRM'
        self.strategy_factory = StrategyFactory()
        
        print("🎯 CRM (Salesforce) 深度分析器初始化完成")
    
    def get_comprehensive_data(self):
        """获取综合数据"""
        print("\n📡 获取CRM实时数据...")
        
        try:
            # 获取多个时间段的数据
            ticker = yf.Ticker(self.symbol)
            
            # 获取基本信息
            info = ticker.info
            
            # 获取历史价格数据
            hist_1y = ticker.history(period='1y')
            hist_6m = ticker.history(period='6mo') 
            hist_3m = ticker.history(period='3mo')
            hist_1m = ticker.history(period='1mo')
            
            print(f"✅ 数据获取完成 - 1年数据: {len(hist_1y)}天")
            
            return {
                'info': info,
                'hist_1y': hist_1y,
                'hist_6m': hist_6m,
                'hist_3m': hist_3m,
                'hist_1m': hist_1m
            }
            
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return None
    
    def analyze_technical_signals(self, data):
        """技术分析信号"""
        print("\n📊 技术指标分析")
        print("="*40)
        
        signals = {}
        
        try:
            # 使用各种策略分析
            strategies = ['TDI', 'CPGW', 'NiuniuV3']
            
            for strategy_name in strategies:
                try:
                    strategy = self.strategy_factory.create_strategy(strategy_name)
                    
                    # 标准化数据格式
                    analysis_data = data['hist_6m'].copy()
                    analysis_data.columns = [col.lower() for col in analysis_data.columns]
                    
                    # 生成信号
                    result = strategy.generate_signals(analysis_data)
                    
                    if 'signal' in result.columns:
                        latest_signal = result['signal'].iloc[-1]
                        signals[strategy_name] = {
                            'signal': latest_signal,
                            'strength': abs(latest_signal) if pd.notna(latest_signal) else 0
                        }
                        
                        signal_text = "🟢 买入" if latest_signal > 0 else "🔴 卖出" if latest_signal < 0 else "⚪ 中性"
                        print(f"   {strategy_name:8}: {signal_text} (强度: {abs(latest_signal):.3f})")
                    else:
                        signals[strategy_name] = {'signal': 0, 'strength': 0}
                        print(f"   {strategy_name:8}: ⚪ 无信号")
                        
                except Exception as e:
                    print(f"   {strategy_name:8}: ❌ 分析失败 - {str(e)[:30]}")
                    signals[strategy_name] = {'signal': 0, 'strength': 0}
            
            return signals
            
        except Exception as e:
            print(f"❌ 技术分析失败: {e}")
            return {}
    
    def calculate_support_resistance(self, data):
        """计算支撑阻力位"""
        print("\n📈 支撑阻力位分析")
        print("="*40)
        
        try:
            hist = data['hist_3m']
            closes = hist['Close']
            highs = hist['High']
            lows = hist['Low']
            
            current_price = closes.iloc[-1]
            
            # 计算关键价位
            # 支撑位：近期低点
            support_levels = []
            resistance_levels = []
            
            # 简单的峰谷检测
            for i in range(5, len(lows)-5):
                # 支撑位：局部最低点
                if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                    support_levels.append(lows.iloc[i])
                
                # 阻力位：局部最高点  
                if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                    resistance_levels.append(highs.iloc[i])
            
            # 取最近的几个关键位
            support_levels = sorted(support_levels, reverse=True)[:3]
            resistance_levels = sorted(resistance_levels)[:3]
            
            # 移动平均线作为动态支撑阻力
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_50 = closes.rolling(50).mean().iloc[-1]
            
            print(f"   当前价格: ${current_price:.2f}")
            print(f"   MA20: ${ma_20:.2f}")
            print(f"   MA50: ${ma_50:.2f}")
            
            if support_levels:
                print(f"   关键支撑位: {[f'${s:.2f}' for s in support_levels[:2]]}")
            
            if resistance_levels:
                print(f"   关键阻力位: {[f'${r:.2f}' for r in resistance_levels[:2]]}")
            
            return {
                'current_price': current_price,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'ma_20': ma_20,
                'ma_50': ma_50
            }
            
        except Exception as e:
            print(f"❌ 支撑阻力计算失败: {e}")
            return None
    
    def calculate_volatility_risk(self, data):
        """计算波动率和风险指标"""
        print("\n⚠️  风险评估")
        print("="*40)
        
        try:
            hist = data['hist_3m']
            returns = hist['Close'].pct_change().dropna()
            
            # 计算各种风险指标
            volatility_daily = returns.std()
            volatility_annual = volatility_daily * np.sqrt(252)
            
            # VaR计算（95%置信度）
            var_95 = returns.quantile(0.05)
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            print(f"   日波动率: {volatility_daily:.3f} ({volatility_daily*100:.2f}%)")
            print(f"   年化波动率: {volatility_annual:.3f} ({volatility_annual*100:.2f}%)")
            print(f"   VaR(95%): {var_95:.3f} ({var_95*100:.2f}%)")
            print(f"   最大回撤: {max_drawdown:.3f} ({max_drawdown*100:.2f}%)")
            
            # 风险等级
            if volatility_annual < 0.2:
                risk_level = "低风险"
            elif volatility_annual < 0.4:
                risk_level = "中等风险"
            else:
                risk_level = "高风险"
            
            print(f"   风险等级: {risk_level}")
            
            return {
                'volatility_daily': volatility_daily,
                'volatility_annual': volatility_annual,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'risk_level': risk_level
            }
            
        except Exception as e:
            print(f"❌ 风险计算失败: {e}")
            return None
    
    def generate_trading_plan(self, data, signals, levels, risk_metrics):
        """生成具体交易计划"""
        print("\n🎯 交易计划制定")
        print("="*50)
        
        try:
            current_price = levels['current_price']
            
            # 计算信号强度
            total_signals = sum([s['signal'] for s in signals.values()])
            avg_strength = np.mean([s['strength'] for s in signals.values()])
            
            print(f"📊 综合分析:")
            print(f"   信号总分: {total_signals:.2f}")
            print(f"   平均强度: {avg_strength:.3f}")
            
            # 买入建议
            if total_signals > 0.5 and avg_strength > 0.1:
                action = "🟢 建议买入"
                confidence = min(90, 50 + avg_strength * 100)
            elif total_signals > 0:
                action = "🟡 谨慎买入"
                confidence = 30 + avg_strength * 50
            else:
                action = "⚪ 观望等待"
                confidence = 20
            
            print(f"   操作建议: {action}")
            print(f"   信心度: {confidence:.0f}%")
            
            # 价格建议
            print(f"\n💰 价格策略:")
            
            # 买入价格区间
            if levels['support_levels']:
                buy_price_low = max(levels['support_levels'][0], current_price * 0.97)
                buy_price_high = current_price * 1.02
            else:
                buy_price_low = current_price * 0.98
                buy_price_high = current_price * 1.01
            
            print(f"   建议买入区间: ${buy_price_low:.2f} - ${buy_price_high:.2f}")
            print(f"   当前价格: ${current_price:.2f}")
            
            # 止损价格 (基于波动率)
            stop_loss = current_price * (1 - risk_metrics['volatility_daily'] * 2)
            if levels['support_levels']:
                stop_loss = max(stop_loss, levels['support_levels'][0] * 0.98)
            
            print(f"   建议止损价: ${stop_loss:.2f} ({((stop_loss/current_price-1)*100):+.2f}%)")
            
            # 止盈价格
            if levels['resistance_levels']:
                take_profit_1 = levels['resistance_levels'][0]
                take_profit_2 = current_price * 1.15 if len(levels['resistance_levels']) < 2 else levels['resistance_levels'][1]
            else:
                take_profit_1 = current_price * 1.08
                take_profit_2 = current_price * 1.15
            
            print(f"   第一止盈位: ${take_profit_1:.2f} ({((take_profit_1/current_price-1)*100):+.2f}%)")
            print(f"   第二止盈位: ${take_profit_2:.2f} ({((take_profit_2/current_price-1)*100):+.2f}%)")
            
            # 仓位建议
            risk_factor = min(1.0, max(0.3, 1 - risk_metrics['volatility_annual']))
            position_size = f"{risk_factor*100:.0f}%"
            
            print(f"\n📊 仓位管理:")
            print(f"   建议仓位: {position_size} (基于风险调整)")
            print(f"   分批建仓: 建议分2-3次买入")
            
            # 生成报告
            report = {
                'symbol': 'CRM',
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': float(current_price),
                'action': action,
                'confidence': float(confidence),
                'buy_range': [float(buy_price_low), float(buy_price_high)],
                'stop_loss': float(stop_loss),
                'take_profit': [float(take_profit_1), float(take_profit_2)],
                'position_size': position_size,
                'risk_level': risk_metrics['risk_level'],
                'signals': {k: {'signal': float(v['signal']), 'strength': float(v['strength'])} for k, v in signals.items()}
            }
            
            return report
            
        except Exception as e:
            print(f"❌ 交易计划生成失败: {e}")
            return None
    
    def save_analysis_report(self, report):
        """保存分析报告"""
        try:
            filename = f"CRM_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n📝 详细报告已保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"⚠️  报告保存失败: {e}")
            return None
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始CRM深度分析")
        print("="*60)
        print(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取数据
        data = self.get_comprehensive_data()
        if not data:
            return None
        
        # 技术分析
        signals = self.analyze_technical_signals(data)
        
        # 支撑阻力分析
        levels = self.calculate_support_resistance(data)
        
        # 风险评估
        risk_metrics = self.calculate_volatility_risk(data)
        
        # 生成交易计划
        if signals and levels and risk_metrics:
            report = self.generate_trading_plan(data, signals, levels, risk_metrics)
            
            if report:
                # 保存报告
                filename = self.save_analysis_report(report)
                
                print(f"\n✅ CRM分析完成！")
                print(f"💡 基于分析结果，请谨慎做出投资决策")
                
                return report, filename
        
        return None, None

def main():
    """主函数"""
    try:
        analyzer = CRMAnalyzer()
        report, filename = analyzer.run_analysis()
        
        if report:
            print(f"\n🎉 分析报告生成成功！")
            if filename:
                print(f"📄 详细报告文件: {filename}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 