#!/usr/bin/env python3
"""
每日持股分析主入口脚本
整合宏观分析，提供详细通俗的每日分析报告
"""

import sys
import os
from datetime import datetime

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'monitor'))
sys.path.append(os.path.dirname(__file__))

# 直接导入
import importlib.util
spec = importlib.util.spec_from_file_location("enhanced_daily_analysis", "monitor/enhanced_daily_analysis.py")
enhanced_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_module)
EnhancedDailyAnalysis = enhanced_module.EnhancedDailyAnalysis


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


def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                    📊 每日持股分析系统 v1.0                          ║
║                                                                      ║
║  🌍 宏观环境分析    📈 个股技术分析    💡 操作建议                    ║
║  📊 行业影响评估    🎯 风险评级       💰 收益跟踪                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, 'cyan')


def show_menu():
    """显示菜单选项"""
    print_colored("\n📋 选择分析模式:", 'yellow')
    print_colored("1. 🚀 完整每日分析报告 (推荐)", 'green')
    print_colored("2. 📊 仅宏观环境分析", 'blue')
    print_colored("3. 💼 仅个股分析", 'blue')
    print_colored("4. 🎯 查看历史报告", 'purple')
    print_colored("0. ❌ 退出系统", 'red')
    print_colored("-" * 50, 'white')


def run_full_analysis():
    """运行完整分析"""
    print_colored("🚀 正在启动完整每日分析...", 'yellow')
    
    try:
        analyzer = EnhancedDailyAnalysis()
        
        print_colored("📊 正在生成综合分析报告...", 'blue')
        print_colored("   • 获取宏观数据...", 'cyan')
        print_colored("   • 分析个股表现...", 'cyan')
        print_colored("   • 生成操作建议...", 'cyan')
        
        report = analyzer.generate_comprehensive_daily_report()
        
        # 显示报告
        print("\n" + report)
        
        # 保存报告
        saved_file = analyzer.save_daily_report(report)
        if saved_file:
            print_colored(f"\n💾 报告已保存到: {saved_file}", 'green')
        
        print_colored("\n✅ 完整分析完成！", 'green')
        
    except Exception as e:
        print_colored(f"❌ 分析过程中出错: {e}", 'red')


def run_macro_only():
    """仅运行宏观分析"""
    print_colored("📊 正在运行宏观环境分析...", 'blue')
    
    try:
        # 导入宏观分析模块
        from analysis.portfolio_macro_integration import PortfolioMacroIntegration
        
        integration = PortfolioMacroIntegration()
        report = integration.generate_macro_report()
        
        if 'error' not in report:
            macro_analysis = report.get('detailed_analysis', {}).get('macro_analysis', {})
            macro_score = macro_analysis.get('macro_score', 0)
            macro_recommendation = macro_analysis.get('recommendation', '')
            
            print_colored("\n🌍 宏观环境分析结果:", 'green')
            print_colored(f"📈 宏观得分: {macro_score:.2f}/1.00 ({int(macro_score*100)}分)", 'cyan')
            print_colored(f"💡 环境建议: {macro_recommendation}", 'cyan')
            
            # 行业影响
            sector_impact = report.get('detailed_analysis', {}).get('sector_impact', {})
            if sector_impact:
                print_colored("\n🏭 行业影响分析:", 'yellow')
                for sector, score in sector_impact.items():
                    color = 'green' if score > 0.6 else 'yellow' if score > 0.4 else 'red'
                    print_colored(f"   • {sector}: {score:.2f}分", color)
            
            print_colored("\n✅ 宏观分析完成！", 'green')
        else:
            print_colored(f"❌ 宏观分析失败: {report['error']}", 'red')
            
    except Exception as e:
        print_colored(f"❌ 宏观分析出错: {e}", 'red')


def run_stocks_only():
    """仅运行个股分析"""
    print_colored("💼 正在运行个股分析...", 'blue')
    
    try:
        analyzer = EnhancedDailyAnalysis()
        
        # 获取投资组合配置
        positions = analyzer.portfolio_config.get('positions', {})
        
        if not positions:
            print_colored("❌ 未找到投资组合配置", 'red')
            return
        
        print_colored("\n💼 个股分析结果:", 'green')
        print_colored("-" * 50, 'white')
        
        for symbol, position_info in positions.items():
            stock_name = analyzer.stock_names.get(symbol, symbol)
            
            # 获取股票数据
            stock_data = analyzer.get_stock_data(symbol)
            if stock_data is not None and not stock_data.empty:
                technical_data = analyzer.calculate_technical_indicators(stock_data)
                
                current_price = technical_data.get('current_price', 0)
                price_change = technical_data.get('price_change', 0)
                cost_basis = position_info.get('cost_basis', current_price)
                
                # 收益计算
                return_pct = (current_price - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
                
                # 显示信息
                print_colored(f"\n📈 {stock_name} ({symbol}):", 'cyan')
                print_colored(f"   当前价格: ${current_price:.2f} ({price_change:+.2f}%)", 'white')
                print_colored(f"   成本价格: ${cost_basis:.2f}", 'white')
                
                # 收益状态
                if return_pct > 0:
                    print_colored(f"   收益率: +{return_pct:.2f}% 📈", 'green')
                else:
                    print_colored(f"   收益率: {return_pct:.2f}% 📉", 'red')
            else:
                print_colored(f"\n❌ {stock_name} ({symbol}): 无法获取数据", 'red')
        
        print_colored("\n✅ 个股分析完成！", 'green')
        
    except Exception as e:
        print_colored(f"❌ 个股分析出错: {e}", 'red')


def show_history():
    """显示历史报告"""
    print_colored("📂 查看历史报告...", 'blue')
    
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            print_colored("❌ 报告目录不存在", 'red')
            return
        
        # 查找报告文件
        report_files = [f for f in os.listdir(reports_dir) if f.startswith('daily_analysis_report_')]
        
        if not report_files:
            print_colored("❌ 未找到历史报告", 'red')
            return
        
        # 按时间排序
        report_files.sort(reverse=True)
        
        print_colored("\n📁 最近的分析报告:", 'green')
        for i, filename in enumerate(report_files[:10], 1):  # 显示最近10个
            # 从文件名提取时间
            try:
                timestamp = filename.replace('daily_analysis_report_', '').replace('.txt', '')
                date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
                file_size = os.path.getsize(os.path.join(reports_dir, filename))
                print_colored(f"   {i}. {date_str} ({file_size:,} bytes)", 'cyan')
            except:
                print_colored(f"   {i}. {filename}", 'cyan')
        
        print_colored(f"\n📊 总共找到 {len(report_files)} 个报告文件", 'yellow')
        print_colored(f"📂 报告目录: {os.path.abspath(reports_dir)}", 'white')
        
    except Exception as e:
        print_colored(f"❌ 查看历史报告出错: {e}", 'red')


def main():
    """主函数"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择操作 (0-4): ").strip()
            
            if choice == '1':
                run_full_analysis()
            elif choice == '2':
                run_macro_only()
            elif choice == '3':
                run_stocks_only()
            elif choice == '4':
                show_history()
            elif choice == '0':
                print_colored("\n👋 感谢使用每日持股分析系统！", 'cyan')
                break
            else:
                print_colored("❌ 无效选择，请重新输入", 'red')
                continue
                
            input("\n按回车键继续...")
            
        except KeyboardInterrupt:
            print_colored("\n\n👋 用户中断，退出系统", 'yellow')
            break
        except Exception as e:
            print_colored(f"\n❌ 系统错误: {e}", 'red')
            input("按回车键继续...")


if __name__ == "__main__":
    main() 