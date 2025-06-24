#!/usr/bin/env python3
"""
增强每日分析系统 - 集成右侧交易分析
防止左侧抄底被套，强化趋势跟随思维
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.smart_daily_report import SmartDailyReportGenerator
from right_side_trading_system import generate_right_side_trading_alerts, format_right_side_trading_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_colored(text, color='white'):
    """打印彩色文本"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def load_portfolio_config():
    """加载投资组合配置"""
    try:
        with open('portfolio_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载投资组合配置失败: {e}")
        return None

def run_enhanced_daily_analysis():
    """运行增强每日分析"""
    print_colored("=" * 80, 'cyan')
    print_colored("🎯 增强每日分析系统 - 集成右侧交易分析", 'cyan')
    print_colored("=" * 80, 'cyan')
    print_colored(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'blue')
    
    # 加载投资组合配置
    print_colored("\n📊 加载投资组合配置...", 'yellow')
    portfolio_config = load_portfolio_config()
    
    if not portfolio_config:
        print_colored("❌ 无法加载投资组合配置，使用默认配置", 'red')
        return False
    
    positions = portfolio_config.get('positions', {})
    watchlist = portfolio_config.get('watchlist', {})
    
    print_colored(f"✅ 成功加载配置 - 持仓: {len(positions)}只, 观察: {len(watchlist)}只", 'green')
    
    # 生成右侧交易分析报告
    print_colored("\n🎯 生成右侧交易分析报告...", 'yellow')
    try:
        right_side_alerts = generate_right_side_trading_alerts(positions, watchlist)
        right_side_report = format_right_side_trading_report(right_side_alerts)
        
        # 保存右侧交易报告
        report_filename = f"right_side_trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(right_side_report)
        
        print_colored(f"✅ 右侧交易报告已保存: {report_filename}", 'green')
        
        # 显示汇总信息
        summary = right_side_alerts['summary']
        print_colored("\n📈 右侧交易分析汇总:", 'cyan')
        print_colored(f"   总分析股票: {summary['total_analyzed']}只", 'white')
        print_colored(f"   买入机会: {summary['buy_opportunities']}只", 'green')
        print_colored(f"   卖出警告: {summary['sell_warnings']}只", 'red')
        print_colored(f"   等待建议: {summary['wait_recommendations']}只", 'yellow')
        
    except Exception as e:
        print_colored(f"❌ 右侧交易分析失败: {e}", 'red')
        logger.error(f"右侧交易分析失败: {e}")
    
    # 生成智能日报（已集成右侧交易分析）
    print_colored("\n📊 生成智能每日分析报告...", 'yellow')
    try:
        # 创建智能日报生成器
        generator = SmartDailyReportGenerator(
            auto_update_data=True
        )
        
        # 生成报告
        report_file = generator.generate_report()
        
        print_colored(f"✅ 智能日报已生成: {report_file}", 'green')
        print_colored("📊 报告特性:", 'cyan')
        print_colored("   ✓ 使用真实市场数据", 'white')
        print_colored("   ✓ 包含持仓成本和盈亏分析", 'white')
        print_colored("   ✓ 集成右侧交易分析系统", 'green')
        print_colored("   ✓ 防抄底风险警告", 'yellow')
        print_colored("   ✓ 趋势确认信号分析", 'blue')
        print_colored("   ✓ 宏观环境分析", 'white')
        print_colored("   ✓ 技术指标分析", 'white')
        print_colored("   ✓ 可视化图表", 'white')
        
    except Exception as e:
        print_colored(f"❌ 智能日报生成失败: {e}", 'red')
        logger.error(f"智能日报生成失败: {e}")
        return False
    
    # 显示持仓股票右侧交易状态
    print_colored("\n💼 持仓股票右侧交易状态:", 'cyan')
    print_colored("-" * 60, 'white')
    
    for symbol, analysis in right_side_alerts.get('portfolio_alerts', {}).items():
        if 'error' not in analysis:
            trend_status = analysis['trend_status']
            entry_signals = analysis['entry_signals']
            
            status_color = 'green' if trend_status['confirmed'] and trend_status['direction'] == '上升' else 'yellow'
            print_colored(f"\n📈 {symbol}:", 'cyan')
            print_colored(f"   趋势: {trend_status['direction']} ({trend_status['strength']['level']})", status_color)
            print_colored(f"   确认: {'✅ 已确认' if trend_status['confirmed'] else '❌ 未确认'}", status_color)
            print_colored(f"   持续: {trend_status['trend_days']}天", 'white')
            
            # 显示信号
            if entry_signals['buy_signals']:
                print_colored(f"   🟢 买入信号: {len(entry_signals['buy_signals'])}个", 'green')
            if entry_signals['sell_signals']:
                print_colored(f"   🔴 卖出信号: {len(entry_signals['sell_signals'])}个", 'red')
            if entry_signals['wait_signals']:
                print_colored(f"   🟡 等待信号: {len(entry_signals['wait_signals'])}个", 'yellow')
    
    # 显示观察列表股票状态
    if right_side_alerts.get('watchlist_alerts'):
        print_colored("\n👀 观察列表股票状态:", 'cyan')
        print_colored("-" * 60, 'white')
        
        for symbol, analysis in right_side_alerts['watchlist_alerts'].items():
            if 'error' not in analysis:
                trend_status = analysis['trend_status']
                entry_signals = analysis['entry_signals']
                
                buy_ready = len(entry_signals['buy_signals']) > 0
                status_color = 'green' if buy_ready else 'yellow'
                
                print_colored(f"\n🔍 {symbol}:", 'cyan')
                print_colored(f"   趋势: {trend_status['direction']} ({trend_status['strength']['level']})", status_color)
                print_colored(f"   买入时机: {'✅ 合适' if buy_ready else '❌ 等待'}", status_color)
    
    # 显示右侧交易核心提醒
    print_colored("\n🎯 右侧交易核心提醒:", 'cyan')
    print_colored("=" * 60, 'cyan')
    print_colored("✅ 执行原则:", 'green')
    print_colored("   1. 只在趋势确认后进入", 'white')
    print_colored("   2. 必须有成交量配合", 'white')
    print_colored("   3. 设置明确的止损位", 'white')
    print_colored("   4. 分批建仓，控制风险", 'white')
    print_colored("", 'white')
    print_colored("❌ 避免行为:", 'red')
    print_colored("   1. 不要试图抄底摸顶", 'white')
    print_colored("   2. 不要在下跌趋势中抢反弹", 'white')
    print_colored("   3. 不要追涨无量的股票", 'white')
    print_colored("   4. 不要忽视止损信号", 'white')
    
    print_colored("\n" + "=" * 80, 'cyan')
    print_colored("✅ 增强每日分析完成！", 'green')
    print_colored("💡 记住：右侧交易，趋势为王，严格止损！", 'yellow')
    print_colored("=" * 80, 'cyan')
    
    return True

def show_right_side_trading_principles():
    """显示右侧交易原则说明"""
    print_colored("\n" + "=" * 80, 'cyan')
    print_colored("📚 右侧交易原理说明", 'cyan')
    print_colored("=" * 80, 'cyan')
    
    print_colored("\n🎯 什么是右侧交易？", 'yellow')
    print_colored("右侧交易是指在价格趋势确认后再进入的交易方式：", 'white')
    print_colored("• 左侧交易：在价格下跌过程中买入（抄底）", 'red')
    print_colored("• 右侧交易：在价格上升趋势确认后买入（跟趋势）", 'green')
    
    print_colored("\n💡 右侧交易的优势：", 'yellow')
    print_colored("1. 避免抄底被套：不在下跌趋势中盲目买入", 'white')
    print_colored("2. 提高成功率：跟随确认的趋势操作", 'white')
    print_colored("3. 控制风险：有明确的止损位", 'white')
    print_colored("4. 心理压力小：顺势而为，不逆市操作", 'white')
    
    print_colored("\n⚠️ 左侧交易的风险：", 'yellow')
    print_colored("1. 抄底抄在半山腰：价格可能继续下跌", 'red')
    print_colored("2. 无法预测底部：市场底部难以准确判断", 'red')
    print_colored("3. 心理压力大：持续亏损影响判断", 'red')
    print_colored("4. 资金利用率低：被套资金无法灵活运用", 'red')
    
    print_colored("\n🔧 系统如何帮助您？", 'yellow')
    print_colored("1. 趋势确认分析：多重指标确认趋势方向", 'green')
    print_colored("2. 成交量验证：确保有资金推动价格上涨", 'green')
    print_colored("3. 风险警告：识别左侧交易风险", 'yellow')
    print_colored("4. 入场时机：提供具体的买入信号", 'green')
    print_colored("5. 止损建议：设置合理的风险控制位", 'blue')
    
    print_colored("\n📊 技术指标解读：", 'yellow')
    print_colored("• 均线排列：多头排列确认上升趋势", 'white')
    print_colored("• 成交量配合：放量上涨显示资金推动", 'white')
    print_colored("• RSI指标：避免超买区域追高", 'white')
    print_colored("• MACD信号：金叉确认动量转强", 'white')
    print_colored("• 价格突破：突破关键阻力位确认", 'white')
    
    print_colored("\n" + "=" * 80, 'cyan')

def main():
    """主函数"""
    try:
        # 显示右侧交易原理
        show_right_side_trading_principles()
        
        # 询问是否继续
        user_input = input("\n是否开始增强每日分析？(y/n): ").lower().strip()
        if user_input not in ['y', 'yes', '是', '']:
            print_colored("分析已取消", 'yellow')
            return
        
        # 运行增强分析
        success = run_enhanced_daily_analysis()
        
        if success:
            print_colored("\n🎉 分析完成！请查看生成的报告文件。", 'green')
        else:
            print_colored("\n❌ 分析过程中出现错误，请检查日志。", 'red')
            
    except KeyboardInterrupt:
        print_colored("\n\n用户中断操作", 'yellow')
    except Exception as e:
        print_colored(f"\n❌ 程序执行出错: {e}", 'red')
        logger.error(f"程序执行出错: {e}")

if __name__ == "__main__":
    main() 