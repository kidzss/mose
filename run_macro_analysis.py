#!/usr/bin/env python3
"""
宏观因子分析主入口脚本
执行完整的宏观因子分析并生成报告
"""

import sys
import os
import json
from datetime import datetime

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from analysis.portfolio_macro_integration import PortfolioMacroIntegration


def print_colored(text, color='', end='\n'):
    """打印彩色文本"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    if color in colors:
        print(f"{colors[color]}{text}{colors['end']}", end=end)
    else:
        print(text, end=end)


def display_macro_report(report):
    """显示宏观分析报告"""
    print_colored("\n" + "="*60, 'blue')
    print_colored("         投资组合宏观因子分析报告", 'blue')
    print_colored("="*60, 'blue')
    
    # 报告摘要
    summary = report['executive_summary']
    print_colored(f"\n📊 报告日期: {report['report_date']}", 'cyan')
    print_colored(f"🌡️  宏观得分: {summary['macro_score']:.2f}/1.00", 'cyan')
    print_colored(f"💡 宏观建议: {summary['macro_recommendation']}", 'cyan')
    print_colored(f"⚠️  投资组合风险等级: {summary['portfolio_risk_level'].upper()}", 'yellow')
    
    if summary['key_concerns']:
        print_colored(f"🚨 重点关注: {', '.join(summary['key_concerns'])}", 'red')
    
    # 详细分析
    detailed = report['detailed_analysis']
    macro_analysis = detailed.get('macro_analysis', {})
    sector_impact = detailed.get('sector_impact', {})
    portfolio_impact = detailed.get('portfolio_impact', {})
    
    # 宏观因子分析
    print_colored("\n📈 宏观因子分析:", 'green')
    if 'components' in macro_analysis:
        components = macro_analysis['components']
        for factor, data in components.items():
            if isinstance(data, dict):
                print(f"  • {factor}: {json.dumps(data, indent=4, ensure_ascii=False)}")
    
    # 行业影响分析
    print_colored("\n🏭 行业影响分析:", 'green')
    for sector, score in sector_impact.items():
        color = 'green' if score > 0.6 else 'yellow' if score > 0.4 else 'red'
        print_colored(f"  • {sector}: {score:.2f}", color)
    
    # 投资组合影响
    print_colored("\n💼 投资组合影响分析:", 'green')
    if 'individual_impacts' in portfolio_impact:
        print_colored("  个股影响评估:", 'cyan')
        for symbol, info in portfolio_impact['individual_impacts'].items():
            impact_color = {
                'positive': 'green',
                'neutral': 'yellow', 
                'negative': 'red',
                'very_negative': 'red'
            }.get(info['impact_level'], 'white')
            
            print_colored(f"    {symbol} ({info['sector']}): ", end='')
            print_colored(f"{info['impact_score']:.2f} - {info['impact_description']}", impact_color)
    
    # 行动计划
    action_plan = report['action_plan']
    
    if action_plan['priority_1']:
        print_colored("\n🚨 立即行动项:", 'red')
        for i, action in enumerate(action_plan['priority_1'], 1):
            print_colored(f"  {i}. {action}", 'red')
    
    if action_plan['priority_2']:
        print_colored("\n📋 中期行动项:", 'yellow')
        for i, action in enumerate(action_plan['priority_2'], 1):
            print_colored(f"  {i}. {action}", 'yellow')
    
    if action_plan['risk_management']:
        print_colored("\n🛡️  风险管理建议:", 'purple')
        for i, suggestion in enumerate(action_plan['risk_management'], 1):
            print_colored(f"  {i}. {suggestion}", 'purple')
    
    if action_plan['monitoring']:
        print_colored("\n👀 监控要点:", 'cyan')
        for i, point in enumerate(action_plan['monitoring'], 1):
            print_colored(f"  {i}. {point}", 'cyan')
    
    print_colored("\n" + "="*60, 'blue')


def save_report_to_file(report, filename=None):
    """保存报告到文件"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"macro_analysis_report_{timestamp}.json"
    
    filepath = os.path.join('reports', filename)
    os.makedirs('reports', exist_ok=True)
    
    # 处理datetime序列化问题
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=datetime_handler)
    
    print_colored(f"\n💾 报告已保存到: {filepath}", 'green')
    return filepath


def main():
    """主函数"""
    print_colored("🚀 启动宏观因子分析系统...", 'blue')
    
    try:
        # 初始化分析器
        integration = PortfolioMacroIntegration()
        
        # 生成报告
        print_colored("📊 正在生成宏观分析报告...", 'yellow')
        report = integration.generate_macro_report()
        
        if 'error' in report:
            print_colored(f"❌ 报告生成失败: {report['error']}", 'red')
            return
        
        # 显示报告
        display_macro_report(report)
        
        # 保存报告
        saved_file = save_report_to_file(report)
        
        print_colored("\n✅ 宏观分析完成！", 'green')
        
    except Exception as e:
        print_colored(f"❌ 系统运行出错: {e}", 'red')
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 