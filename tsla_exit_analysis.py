import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class TSLAExitAnalysis:
    """TSLA清仓分析"""
    
    def __init__(self):
        self.exit_details = {
            'symbol': 'TSLA',
            'shares': 4,
            'cost_basis': 179.841,
            'exit_price': 324.00,
            'exit_date': '2025-06-20',
            'total_investment': 719.364,  # 4 * 179.841
            'total_proceeds': 1296.00,    # 4 * 324.00
            'gross_profit': 576.636,      # 1296 - 719.364
            'return_percentage': 80.16    # (1296 - 719.364) / 719.364
        }
    
    def analyze_exit_timing(self):
        """分析退出时机"""
        print("🎯 TSLA清仓时机分析")
        print("=" * 60)
        
        # 获取TSLA近期数据
        tsla = yf.Ticker("TSLA")
        data = tsla.history(period="1y")
        
        if len(data) == 0:
            print("❌ 无法获取TSLA数据")
            return
        
        # 计算技术指标
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        
        # 分析退出时的市场状态
        latest = data.iloc[-1]
        
        print(f"📊 退出时技术指标:")
        print(f"退出价格: ${self.exit_details['exit_price']:.2f}")
        print(f"当前价格: ${latest['Close']:.2f}")
        print(f"RSI: {latest['RSI']:.1f}")
        print(f"20日均线: ${latest['MA20']:.2f}")
        print(f"50日均线: ${latest['MA50']:.2f}")
        
        # 判断退出时机
        timing_score = 0
        timing_analysis = []
        
        if self.exit_details['exit_price'] > latest['MA20']:
            timing_score += 2
            timing_analysis.append("✓ 在20日均线上方退出")
        
        if self.exit_details['exit_price'] > latest['MA50']:
            timing_score += 2
            timing_analysis.append("✓ 在50日均线上方退出")
            
        if latest['RSI'] > 70:
            timing_score += 3
            timing_analysis.append("✓ RSI超买区域退出")
        elif latest['RSI'] > 60:
            timing_score += 2
            timing_analysis.append("✓ RSI偏高区域退出")
        
        # 价格位置分析
        recent_high = data['High'].rolling(60).max().iloc[-1]
        recent_low = data['Low'].rolling(60).min().iloc[-1]
        price_position = (self.exit_details['exit_price'] - recent_low) / (recent_high - recent_low)
        
        if price_position > 0.8:
            timing_score += 3
            timing_analysis.append(f"✓ 在60日区间高位退出 ({price_position:.1%})")
        elif price_position > 0.6:
            timing_score += 2
            timing_analysis.append(f"✓ 在60日区间中高位退出 ({price_position:.1%})")
        
        print(f"\n💡 退出时机评估 (评分: {timing_score}/10):")
        for analysis in timing_analysis:
            print(f"  {analysis}")
        
        if timing_score >= 8:
            print("🟢 退出时机：优秀")
        elif timing_score >= 6:
            print("🟡 退出时机：良好")
        elif timing_score >= 4:
            print("🟠 退出时机：一般")
        else:
            print("🔴 退出时机：需要改进")
        
        return timing_score
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analyze_portfolio_impact(self):
        """分析对投资组合的影响"""
        print(f"\n💰 投资组合影响分析")
        print("=" * 60)
        
        print(f"📈 交易表现:")
        print(f"投资成本: ${self.exit_details['total_investment']:.2f}")
        print(f"退出收益: ${self.exit_details['total_proceeds']:.2f}")
        print(f"绝对收益: ${self.exit_details['gross_profit']:.2f}")
        print(f"收益率: {self.exit_details['return_percentage']:.1f}%")
        
        # 资金释放分析
        print(f"\n💵 资金变化:")
        print(f"释放资金: ${self.exit_details['total_proceeds']:.2f}")
        print(f"现金占比提升: 约4.6% (从8.4%到12.8%)")
        print(f"股票仓位降低: 约6% (从70%到64%)")
        
        # 风险分散效果
        print(f"\n🛡️ 风险管理效果:")
        print(f"• 减少单一股票集中风险")
        print(f"• 降低汽车行业暴露")
        print(f"• 增加现金灵活性")
        print(f"• 为新投资机会腾出空间")
        
        return self.exit_details['gross_profit']
    
    def strategy_evaluation(self):
        """策略评估"""
        print(f"\n🎓 波段操作策略评估")
        print("=" * 60)
        
        # 符合波段操作原则分析
        strategy_points = []
        
        # 右侧交易原则
        if self.exit_details['return_percentage'] > 20:
            strategy_points.append("✓ 符合右侧交易原则：盈利20%以上退出")
        
        # 波段幅度合理
        if 50 <= self.exit_details['return_percentage'] <= 100:
            strategy_points.append("✓ 波段幅度合理：50%-100%区间")
        elif self.exit_details['return_percentage'] > 100:
            strategy_points.append("⚠️ 收益率偏高：可能存在更早退出机会")
        
        # 持仓时间
        strategy_points.append("✓ 持仓时间适中：约6个月，符合波段操作周期")
        
        # 市场环境
        strategy_points.append("✓ 高位获利了结：避免后续回调风险")
        
        print("📊 策略执行评估:")
        for point in strategy_points:
            print(f"  {point}")
        
        # 改进建议
        print(f"\n💡 策略优化建议:")
        print("  1. 可考虑分批卖出：30%盈利卖出1/3，50%盈利卖出1/3，70%盈利卖出1/3")
        print("  2. 设置移动止损：保护已实现利润")
        print("  3. 关注技术指标：RSI>70时开始减仓")
        print("  4. 考虑再入时机：等待回调至$250-280区间")
        
    def generate_next_actions(self):
        """生成后续行动建议"""
        print(f"\n🎯 后续行动建议")
        print("=" * 60)
        
        print("💰 资金配置建议 (新增$1,296现金):")
        print("  方案一：防御性配置")
        print("    • $600 买入 BRK-B (1股，防御性)")
        print("    • $400 买入 MRK (5股，高股息)")
        print("    • $296 保留现金等待机会")
        
        print("\n  方案二：成长性配置")
        print("    • $650 加仓 GOOG (4股，补强仓位)")
        print("    • $646 保留现金等待科技股回调")
        
        print("\n  方案三：平衡配置")
        print("    • $400 买入 JPM (1股，金融)")
        print("    • $400 买入 MRK (5股，医疗)")
        print("    • $496 保留现金机动")
        
        print(f"\n⏰ 时机选择:")
        print("  • 不急于立即投入")
        print("  • 等待市场回调或技术信号")
        print("  • 关注财报季表现")
        print("  • 保持12.8%现金比例的灵活性")
        
        print(f"\n🔄 TSLA再入机会:")
        print("  • 目标价位: $250-280")
        print("  • 技术条件: RSI<40 + 突破50日均线")
        print("  • 基本面: 交付量数据改善")
        print("  • 仓位控制: 不超过总资产5%")
    
    def comprehensive_analysis(self):
        """综合分析报告"""
        print("🎯 TSLA清仓综合分析报告")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 执行各项分析
        timing_score = self.analyze_exit_timing()
        profit = self.analyze_portfolio_impact()
        self.strategy_evaluation()
        self.generate_next_actions()
        
        # 总结评价
        print(f"\n📋 总结评价")
        print("=" * 60)
        
        overall_score = 0
        
        # 收益率评分 (40%)
        if self.exit_details['return_percentage'] > 70:
            profit_score = 10
        elif self.exit_details['return_percentage'] > 50:
            profit_score = 8
        elif self.exit_details['return_percentage'] > 30:
            profit_score = 6
        else:
            profit_score = 4
        
        overall_score += profit_score * 0.4
        
        # 时机评分 (30%)
        overall_score += timing_score * 0.3
        
        # 策略执行评分 (30%)
        strategy_score = 8  # 基于策略符合度
        overall_score += strategy_score * 0.3
        
        print(f"💯 综合评分: {overall_score:.1f}/10")
        
        if overall_score >= 9:
            grade = "A+ 优秀"
        elif overall_score >= 8:
            grade = "A 良好"
        elif overall_score >= 7:
            grade = "B+ 中上"
        elif overall_score >= 6:
            grade = "B 中等"
        else:
            grade = "C 需改进"
        
        print(f"🏆 交易等级: {grade}")
        
        print(f"\n✅ 成功要素:")
        print("  • 严格执行波段操作纪律")
        print("  • 在高位区域获利了结")
        print("  • 实现了显著的绝对收益")
        print("  • 降低了投资组合风险")
        
        print(f"\n📚 经验总结:")
        print("  • 右侧交易策略有效")
        print("  • 80%收益率符合波段操作目标")
        print("  • 资金管理得当，现金比例提升")
        print("  • 为后续投资创造了机会")

if __name__ == "__main__":
    analyzer = TSLAExitAnalysis()
    analyzer.comprehensive_analysis() 