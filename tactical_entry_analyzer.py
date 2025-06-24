#!/usr/bin/env python3
"""
战术买入点分析器
为阶段二分批买入提供具体的价位和时机指导
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TacticalEntryAnalyzer:
    """战术买入点分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 阶段二买入股票
        self.phase2_stocks = {
            # 第1批：防御性资产
            'batch_1_defensive': {
                'BRK-B': {'target_shares': 4, 'target_investment': 1941},
                'GS': {'target_shares': 3, 'target_investment': 1906},
                'ABT': {'target_shares': 14, 'target_investment': 1854},
                'MRK': {'target_shares': 24, 'target_investment': 1903}
            },
            # 第2批：科技股
            'batch_2_tech': {
                'PLTR': {'target_shares': 13, 'target_investment': 1819},
                'ORCL': {'target_shares': 9, 'target_investment': 1898},
                'IBM': {'target_shares': 6, 'target_investment': 1699}
            },
            # 第3批：消费+ETF
            'batch_3_diversified': {
                'COST': {'target_shares': 1, 'target_investment': 975},
                'XLK': {'target_shares': 3, 'target_investment': 725}
            }
        }
        
        # 市场基准
        self.market_benchmark = 'SPY'
        
        logger.info("📈 战术买入点分析器初始化完成")
    
    def get_technical_analysis(self, symbol, period="6mo"):
        """获取技术分析数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # 计算技术指标
            current_price = hist['Close'].iloc[-1]
            
            # 移动平均线
            ma5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma10 = hist['Close'].rolling(10).mean().iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            ma50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 布林带
            bb_period = 20
            bb_std = 2
            bb_middle = hist['Close'].rolling(bb_period).mean()
            bb_std_dev = hist['Close'].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_middle = bb_middle.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            
            # 支撑阻力位
            recent_high = hist['High'].rolling(60).max().iloc[-1]  # 3个月高点
            recent_low = hist['Low'].rolling(60).min().iloc[-1]    # 3个月低点
            
            # 成交量分析
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # 波动率
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            return {
                'current_price': current_price,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'ma50': ma50,
                'rsi': current_rsi,
                'bb_upper': current_bb_upper,
                'bb_middle': current_bb_middle,
                'bb_lower': current_bb_lower,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'price_vs_ma20': (current_price / ma20 - 1) * 100,
                'price_vs_ma50': (current_price / ma50 - 1) * 100,
                'distance_from_high': (current_price / recent_high - 1) * 100,
                'distance_from_low': (current_price / recent_low - 1) * 100
            }
            
        except Exception as e:
            logger.warning(f"获取{symbol}技术分析失败: {e}")
            return None
    
    def calculate_entry_levels(self, symbol, tech_data):
        """计算买入价位"""
        if not tech_data:
            return None
        
        current_price = tech_data['current_price']
        
        # 买入价位计算
        entry_levels = {
            'aggressive_buy': current_price * 0.98,  # 激进买入：当前价格下方2%
            'conservative_buy': current_price * 0.95,  # 保守买入：当前价格下方5%
            'value_buy': min(tech_data['bb_lower'], tech_data['ma20'] * 0.97),  # 价值买入：布林带下轨或20日均线下方3%
            'panic_buy': tech_data['recent_low'] * 1.02,  # 恐慌买入：近期低点上方2%
        }
        
        # 止损价位
        stop_loss = current_price * 0.85
        
        # 目标价位
        target_price = current_price * 1.20
        
        # 技术信号评分
        signal_score = 0
        signals = []
        
        # RSI信号
        if tech_data['rsi'] < 30:
            signal_score += 3
            signals.append("RSI超卖")
        elif tech_data['rsi'] < 40:
            signal_score += 2
            signals.append("RSI偏低")
        elif tech_data['rsi'] > 70:
            signal_score -= 2
            signals.append("RSI超买")
        
        # 价格相对均线
        if tech_data['price_vs_ma20'] < -5:
            signal_score += 2
            signals.append("跌破20日均线")
        elif tech_data['price_vs_ma20'] > 5:
            signal_score -= 1
            signals.append("远离20日均线")
        
        # 布林带位置
        bb_position = (current_price - tech_data['bb_lower']) / (tech_data['bb_upper'] - tech_data['bb_lower'])
        if bb_position < 0.2:
            signal_score += 2
            signals.append("接近布林带下轨")
        elif bb_position > 0.8:
            signal_score -= 2
            signals.append("接近布林带上轨")
        
        # 距离高点回调
        if tech_data['distance_from_high'] < -15:
            signal_score += 2
            signals.append(f"距高点回调{-tech_data['distance_from_high']:.1f}%")
        
        # 成交量确认
        if tech_data['volume_ratio'] > 1.5:
            signal_score += 1
            signals.append("成交量放大")
        
        # 综合评级
        if signal_score >= 5:
            rating = "STRONG_BUY"
        elif signal_score >= 3:
            rating = "BUY"
        elif signal_score >= 0:
            rating = "HOLD"
        else:
            rating = "WAIT"
        
        return {
            'entry_levels': entry_levels,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'signal_score': signal_score,
            'signals': signals,
            'rating': rating,
            'bb_position': bb_position,
            'current_price': current_price
        }
    
    def analyze_market_scenarios(self):
        """分析不同市场情景"""
        # 获取SPY技术分析
        spy_tech = self.get_technical_analysis('SPY')
        
        if not spy_tech:
            return None
        
        spy_current = spy_tech['current_price']
        spy_ma20 = spy_tech['ma20']
        spy_ma50 = spy_tech['ma50']
        spy_rsi = spy_tech['rsi']
        
        scenarios = {
            'bull_scenario': {
                'name': '牛市情景',
                'probability': 0.30,
                'spy_target_range': (spy_current * 1.02, spy_current * 1.08),
                'description': 'SPY突破阻力，继续上涨',
                'strategy': '追涨买入，关注动量股',
                'timing': '立即执行，分2批买入'
            },
            'correction_scenario': {
                'name': '回调情景',
                'probability': 0.40,
                'spy_target_range': (spy_current * 0.92, spy_current * 0.98),
                'description': 'SPY回调至支撑位',
                'strategy': '逢低买入，关注价值股',
                'timing': '等待回调，分3批买入'
            },
            'sideways_scenario': {
                'name': '震荡情景',
                'probability': 0.25,
                'spy_target_range': (spy_current * 0.97, spy_current * 1.03),
                'description': 'SPY在区间震荡',
                'strategy': '区间操作，低买高卖',
                'timing': '分批买入，控制节奏'
            },
            'panic_scenario': {
                'name': '恐慌情景',
                'probability': 0.05,
                'spy_target_range': (spy_current * 0.85, spy_current * 0.92),
                'description': 'SPY大幅下跌',
                'strategy': '抄底买入，重仓配置',
                'timing': '快速买入，单批执行'
            }
        }
        
        # 根据当前技术面判断最可能的情景
        current_scenario_prob = {}
        
        # 基于RSI判断
        if spy_rsi > 70:
            current_scenario_prob['correction_scenario'] = 0.5
            current_scenario_prob['sideways_scenario'] = 0.3
        elif spy_rsi < 30:
            current_scenario_prob['bull_scenario'] = 0.6
            current_scenario_prob['sideways_scenario'] = 0.2
        else:
            current_scenario_prob['sideways_scenario'] = 0.4
            current_scenario_prob['bull_scenario'] = 0.3
            current_scenario_prob['correction_scenario'] = 0.2
        
        # 基于价格相对均线判断
        if spy_tech['price_vs_ma20'] > 3:
            current_scenario_prob['correction_scenario'] = current_scenario_prob.get('correction_scenario', 0) + 0.2
        elif spy_tech['price_vs_ma20'] < -3:
            current_scenario_prob['bull_scenario'] = current_scenario_prob.get('bull_scenario', 0) + 0.2
        
        return {
            'spy_current': spy_current,
            'spy_ma20': spy_ma20,
            'spy_ma50': spy_ma50,
            'spy_rsi': spy_rsi,
            'scenarios': scenarios,
            'current_scenario_prob': current_scenario_prob
        }
    
    def generate_tactical_plan(self):
        """生成战术买入计划"""
        # 分析市场情景
        market_analysis = self.analyze_market_scenarios()
        
        # 分析各股票
        stock_analysis = {}
        
        all_stocks = {}
        for batch_name, stocks in self.phase2_stocks.items():
            all_stocks.update(stocks)
        
        for symbol in all_stocks.keys():
            logger.info(f"分析 {symbol} 技术面...")
            tech_data = self.get_technical_analysis(symbol)
            entry_analysis = self.calculate_entry_levels(symbol, tech_data)
            
            if tech_data and entry_analysis:
                stock_analysis[symbol] = {
                    'tech_data': tech_data,
                    'entry_analysis': entry_analysis,
                    'target_investment': all_stocks[symbol]['target_investment'],
                    'target_shares': all_stocks[symbol]['target_shares']
                }
        
        return {
            'market_analysis': market_analysis,
            'stock_analysis': stock_analysis,
            'timestamp': datetime.now()
        }
    
    def generate_tactical_report(self, tactical_plan):
        """生成战术买入报告"""
        market_analysis = tactical_plan['market_analysis']
        stock_analysis = tactical_plan['stock_analysis']
        
        report = []
        report.append("=" * 120)
        report.append("🎯 阶段二战术买入计划")
        report.append("📈 基于技术分析的精准入场时机")
        report.append(f"📅 分析时间: {tactical_plan['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 120)
        
        # 市场环境分析
        if market_analysis:
            report.append(f"\n📊 当前市场环境分析:")
            report.append("-" * 100)
            report.append(f"• SPY当前价格: ${market_analysis['spy_current']:.2f}")
            report.append(f"• SPY 20日均线: ${market_analysis['spy_ma20']:.2f} "
                         f"(距离{(market_analysis['spy_current']/market_analysis['spy_ma20']-1)*100:+.1f}%)")
            report.append(f"• SPY 50日均线: ${market_analysis['spy_ma50']:.2f} "
                         f"(距离{(market_analysis['spy_current']/market_analysis['spy_ma50']-1)*100:+.1f}%)")
            report.append(f"• SPY RSI: {market_analysis['spy_rsi']:.1f}")
            
            # 市场情景分析
            report.append(f"\n📈 市场情景分析:")
            report.append("-" * 100)
            for scenario_name, scenario in market_analysis['scenarios'].items():
                spy_range = scenario['spy_target_range']
                report.append(f"\n• {scenario['name']} (概率{scenario['probability']:.0%}):")
                report.append(f"  SPY目标区间: ${spy_range[0]:.0f} - ${spy_range[1]:.0f}")
                report.append(f"  策略: {scenario['strategy']}")
                report.append(f"  时机: {scenario['timing']}")
        
        # 分批买入详细计划
        report.append(f"\n💼 分批买入详细计划:")
        report.append("=" * 120)
        
        # 第1批：防御性资产
        report.append(f"\n🛡️ 第1批 - 防御性资产 (优先级: 最高)")
        report.append("-" * 100)
        report.append(f"{'股票':<8} {'现价':<10} {'评级':<12} {'激进买入':<10} {'保守买入':<10} {'价值买入':<10} {'目标投资':<10}")
        report.append("-" * 80)
        
        batch1_stocks = ['BRK-B', 'GS', 'ABT', 'MRK']
        for symbol in batch1_stocks:
            if symbol in stock_analysis:
                data = stock_analysis[symbol]
                entry = data['entry_analysis']
                
                report.append(f"{symbol:<8} ${entry['current_price']:<9.2f} "
                             f"{entry['rating']:<12} "
                             f"${entry['entry_levels']['aggressive_buy']:<9.2f} "
                             f"${entry['entry_levels']['conservative_buy']:<9.2f} "
                             f"${entry['entry_levels']['value_buy']:<9.2f} "
                             f"${data['target_investment']:<9.0f}")
        
        # 第2批：科技股
        report.append(f"\n💻 第2批 - 科技股 (优先级: 中等)")
        report.append("-" * 100)
        report.append(f"{'股票':<8} {'现价':<10} {'评级':<12} {'激进买入':<10} {'保守买入':<10} {'价值买入':<10} {'目标投资':<10}")
        report.append("-" * 80)
        
        batch2_stocks = ['PLTR', 'ORCL', 'IBM']
        for symbol in batch2_stocks:
            if symbol in stock_analysis:
                data = stock_analysis[symbol]
                entry = data['entry_analysis']
                
                report.append(f"{symbol:<8} ${entry['current_price']:<9.2f} "
                             f"{entry['rating']:<12} "
                             f"${entry['entry_levels']['aggressive_buy']:<9.2f} "
                             f"${entry['entry_levels']['conservative_buy']:<9.2f} "
                             f"${entry['entry_levels']['value_buy']:<9.2f} "
                             f"${data['target_investment']:<9.0f}")
        
        # 第3批：消费+ETF
        report.append(f"\n🛒 第3批 - 消费+ETF (优先级: 较低)")
        report.append("-" * 100)
        report.append(f"{'股票':<8} {'现价':<10} {'评级':<12} {'激进买入':<10} {'保守买入':<10} {'价值买入':<10} {'目标投资':<10}")
        report.append("-" * 80)
        
        batch3_stocks = ['COST', 'XLK']
        for symbol in batch3_stocks:
            if symbol in stock_analysis:
                data = stock_analysis[symbol]
                entry = data['entry_analysis']
                
                report.append(f"{symbol:<8} ${entry['current_price']:<9.2f} "
                             f"{entry['rating']:<12} "
                             f"${entry['entry_levels']['aggressive_buy']:<9.2f} "
                             f"${entry['entry_levels']['conservative_buy']:<9.2f} "
                             f"${entry['entry_levels']['value_buy']:<9.2f} "
                             f"${data['target_investment']:<9.0f}")
        
        # 详细技术分析
        report.append(f"\n📊 个股详细技术分析:")
        report.append("=" * 120)
        
        for symbol, data in stock_analysis.items():
            tech = data['tech_data']
            entry = data['entry_analysis']
            
            report.append(f"\n🎯 {symbol}")
            report.append("-" * 80)
            report.append(f"• 当前价格: ${entry['current_price']:.2f}")
            report.append(f"• 技术评级: {entry['rating']} (评分: {entry['signal_score']})")
            report.append(f"• RSI: {tech['rsi']:.1f}")
            report.append(f"• 距20日均线: {tech['price_vs_ma20']:+.1f}%")
            report.append(f"• 距3个月高点: {tech['distance_from_high']:+.1f}%")
            report.append(f"• 布林带位置: {entry['bb_position']:.1%} (0%=下轨, 100%=上轨)")
            
            if entry['signals']:
                report.append(f"• 技术信号: {', '.join(entry['signals'])}")
            
            report.append(f"• 买入价位指导:")
            report.append(f"  - 激进买入: ${entry['entry_levels']['aggressive_buy']:.2f} (当前-2%)")
            report.append(f"  - 保守买入: ${entry['entry_levels']['conservative_buy']:.2f} (当前-5%)")
            report.append(f"  - 价值买入: ${entry['entry_levels']['value_buy']:.2f} (技术支撑位)")
            report.append(f"  - 恐慌买入: ${entry['entry_levels']['panic_buy']:.2f} (近期低点+2%)")
            report.append(f"• 止损价格: ${entry['stop_loss']:.2f}")
            report.append(f"• 目标价格: ${entry['target_price']:.2f}")
        
        # 不同市场情景下的执行策略
        if market_analysis:
            report.append(f"\n📈 不同市场情景执行策略:")
            report.append("-" * 100)
            
            spy_current = market_analysis['spy_current']
            
            report.append(f"\n🐂 牛市情景 (SPY > ${spy_current*1.02:.0f}):")
            report.append(f"• 立即买入第1批防御性资产的50%")
            report.append(f"• 追涨买入PLTR、ORCL等成长股")
            report.append(f"• 使用激进买入价位")
            
            report.append(f"\n📉 回调情景 (SPY ${spy_current*0.92:.0f} - ${spy_current*0.98:.0f}):")
            report.append(f"• 等待SPY跌至${spy_current*0.95:.0f}以下开始买入")
            report.append(f"• 优先买入BRK-B、ABT等防御股")
            report.append(f"• 使用保守买入价位")
            
            report.append(f"\n📊 震荡情景 (SPY ${spy_current*0.97:.0f} - ${spy_current*1.03:.0f}):")
            report.append(f"• 分3批等量买入，每周一批")
            report.append(f"• 关注个股技术面，逢低买入")
            report.append(f"• 混合使用激进和保守价位")
            
            report.append(f"\n😱 恐慌情景 (SPY < ${spy_current*0.92:.0f}):")
            report.append(f"• 立即大量买入所有目标股票")
            report.append(f"• 使用恐慌买入价位")
            report.append(f"• 重点关注PLTR、COST等高成长股")
        
        # 风险管理
        report.append(f"\n⚠️ 风险管理建议:")
        report.append("-" * 100)
        report.append(f"• 每只股票设置15%止损")
        report.append(f"• 单日买入不超过总资金的30%")
        report.append(f"• 关注成交量确认，避免假突破")
        report.append(f"• 如遇重大利空，暂停买入计划")
        
        # 执行时间表
        report.append(f"\n📅 建议执行时间表:")
        report.append("-" * 100)
        report.append(f"• 第1周: 买入第1批防御性资产 (BRK-B, GS, ABT, MRK)")
        report.append(f"• 第2周: 买入第2批科技股 (PLTR, ORCL, IBM)")
        report.append(f"• 第3周: 买入第3批消费+ETF (COST, XLK)")
        report.append(f"• 灵活调整: 根据市场变化和技术信号调整顺序")
        
        report.append("\n" + "=" * 120)
        report.append("📋 声明: 技术分析仅供参考，实际执行需结合市场实时情况调整")
        report.append("=" * 120)
        
        return '\n'.join(report)

def main():
    """主函数"""
    analyzer = TacticalEntryAnalyzer()
    
    # 生成战术计划
    tactical_plan = analyzer.generate_tactical_plan()
    
    # 生成报告
    report = analyzer.generate_tactical_report(tactical_plan)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'tactical_entry_plan_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(tactical_plan, f, ensure_ascii=False, indent=2, default=str)
    
    with open(f'tactical_entry_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("✅ 战术买入计划生成完成")

if __name__ == "__main__":
    main() 