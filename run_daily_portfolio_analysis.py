#!/usr/bin/env python3
"""
直接运行每日持仓分析
"""
from portfolio_analysis_scheduler import PortfolioAnalysisScheduler

def main():
    print("🚀 开始执行每日持仓分析...")
    print("=" * 60)
    
    scheduler = PortfolioAnalysisScheduler()
    result = scheduler.run_daily_analysis()
    
    if result:
        print("✅ 每日持仓分析完成并发送邮件成功！")
        print("📧 请检查您的邮箱接收分析报告")
    else:
        print("❌ 分析过程中出现问题，请查看日志")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 