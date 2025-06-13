#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import schedule
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_screening.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WeeklyScreening")

def run_weekly_screening():
    """运行每周股票筛选"""
    try:
        logger.info("🚀 开始每周股票池筛选...")
        
        # 运行量化股票筛选器
        import subprocess
        result = subprocess.run(['python', 'monitor/run_stock_screener.py'], 
                              capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        if result.returncode == 0:
            logger.info("✅ 每周股票筛选完成")
            print("✅ 每周股票筛选报告已发送到您的邮箱")
        else:
            logger.error(f"❌ 股票筛选失败: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ 运行每周筛选时出错: {str(e)}")

def main():
    """主函数 - 启动每周筛选定时任务"""
    print("=" * 60)
    print("📈 每周股票池筛选监控服务")
    print("=" * 60)
    print()
    print("📊 服务功能：")
    print("   ✓ 每周日20:00自动筛选股票池")
    print("   ✓ 分析573只股票，推荐top 20")
    print("   ✓ 生成详细分析报告")
    print("   ✓ 自动发送邮件到您的邮箱")
    print()
    print("⏰ 定时安排：")
    print("   - 每周日晚上 20:00 执行")
    print("   - 分析时间约15-30分钟")
    print()
    print("📧 邮件将发送到：kidzss@gmail.com")
    print()
    print("🛑 要停止服务，请按 Ctrl+C")
    print("=" * 60)
    print()
    
    try:
        # 设置定时任务
        schedule.every().sunday.at("20:00").do(run_weekly_screening)
        
        # 显示下次运行时间
        next_run = schedule.next_run()
        if next_run:
            print(f"⏰ 下次运行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 可选：立即运行一次测试
        test_now = input("💡 是否立即运行一次测试？(y/N): ").strip().lower()
        if test_now == 'y':
            print("🧪 开始测试运行...")
            run_weekly_screening()
        
        print("⏳ 等待定时任务触发...")
        print("   (或按 Ctrl+C 停止服务)")
        
        # 保持运行
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
            
    except KeyboardInterrupt:
        print("\n👋 每周筛选服务已停止")
    except Exception as e:
        logger.error(f"服务运行出错: {str(e)}")

if __name__ == "__main__":
    main() 