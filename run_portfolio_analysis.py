#!/usr/bin/env python3
"""
持仓股票分析并发送邮件
运行完整的持仓分析，包括盈亏、技术指标、市场情绪等，并发送邮件报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.smart_daily_report import SmartDailyReportGenerator
from datetime import datetime
import logging
import tempfile
from utils.unified_email_api import send_html

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_portfolio_analysis_and_email():
    """运行持仓分析并发送邮件（方案B：统一API）"""
    try:
        print("📊 开始持仓股票分析...")
        print("=" * 60)
        
        # 创建智能日报生成器
        report_generator = SmartDailyReportGenerator()
        
        # 生成分析报告
        print("🔍 正在分析持仓股票...")
        html_report = report_generator.generate_report()
        
        if not html_report:
            print("❌ 报告生成失败")
            return False
        
        print("✅ 持仓分析报告生成完成")
        
        # 直接用统一API发送HTML内容
        subject = f"📈 持仓股票分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        print("📧 正在发送邮件...")
        success = send_html(subject=subject, html_content=html_report)
        
        if success:
            print("✅ 持仓分析邮件发送成功！")
            print(f"📧 邮件主题: {subject}")
            print("📋 邮件包含:")
            print("   • 持仓股票详细分析")
            print("   • 盈亏情况统计")
            print("   • 技术指标分析")
            print("   • 市场环境评估")
            print("   • 投资建议")
            return True
        else:
            print("❌ 邮件发送失败")
            return False
            
    except Exception as e:
        logger.error(f"持仓分析失败: {e}")
        print(f"❌ 运行出错: {e}")
        return False

def show_portfolio_summary():
    """显示持仓概览"""
    print("\n📋 当前持仓概览:")
    print("=" * 60)
    
    # 从统一配置文件加载持仓信息
    try:
        from utils.portfolio_config_loader import get_portfolio_config
        config_loader = get_portfolio_config()
        portfolio = config_loader.to_smart_report_format()
        portfolio_summary = config_loader.get_portfolio_summary()
        watchlist = config_loader.get_watchlist_symbols()
        print("✅ 已从统一配置文件加载最新持仓信息")
    except Exception as e:
        logger.warning(f"加载统一配置失败，使用默认配置: {e}")
        # 保留最新的持仓信息作为后备
        portfolio = {
            'AMD': {'cost': 126.214, 'shares': 48, 'weight': 21.86, 'investment': 4788.89},
            'GOOGL': {'cost': 170.54, 'shares': 34, 'weight': 21.53, 'investment': 4715.83},
            'PFE': {'cost': 25.899, 'shares': 80, 'weight': 6.97, 'investment': 1526.65},
            'NVDA': {'cost': 138.843, 'shares': 40, 'weight': 20.92, 'investment': 4582.24},
            'TSLA': {'cost': 254.096, 'shares': 4, 'weight': 4.74, 'investment': 1038.22},
            'EOG': {'cost': 122.119, 'shares': 5, 'weight': 2.20, 'investment': 481.88}
        }
        watchlist = ['MSFT', 'ADBE', 'PHM', 'CF']
    
    total_investment = sum(p['investment'] for p in portfolio.values())
    
    print(f"总持仓股票: {len(portfolio)} 只")
    print(f"总投资金额: ${total_investment:,.2f}")
    print()
    
    for symbol, info in portfolio.items():
        print(f"{symbol:6s}: {info['shares']:3d}股 | 成本${info['cost']:8.3f} | "
              f"权重{info['weight']:5.2f}% | 投资${info['investment']:8.2f}")
    
    print(f"\n观察股票: {', '.join(watchlist)} (准备买入)")
    try:
        # 显示观察股票的目标价格
        if 'config_loader' in locals():
            watchlist_details = config_loader.get_watchlist()
            print("观察股票详情:")
            for symbol in watchlist:
                details = watchlist_details.get(symbol, {})
                target_price = details.get('target_buy_price', 'N/A')
                reason = details.get('reason', '无描述')
                print(f"  {symbol}: 目标价格 ${target_price} - {reason}")
    except:
        pass
    print("=" * 60)

if __name__ == "__main__":
    try:
        print("🚀 持仓股票分析系统")
        print("=" * 80)
        
        # 显示持仓概览
        show_portfolio_summary()
        
        # 运行分析并发送邮件
        success = run_portfolio_analysis_and_email()
        
        if success:
            print("\n🎉 持仓分析完成！")
            print("📧 请检查您的邮箱查看详细分析报告")
        else:
            print("\n❌ 持仓分析失败，请检查错误信息")
            
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc() 