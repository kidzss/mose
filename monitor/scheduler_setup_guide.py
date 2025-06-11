#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能日报定时任务设置指南
"""

import os
import sys
from datetime import datetime
from smart_daily_email_sender import SmartDailyEmailSender

def main():
    print("=" * 80)
    print("📅 智能日报定时任务设置指南")
    print("=" * 80)
    
    print("\n📚 两个文件的区别：")
    print("=" * 40)
    
    print("📄 smart_daily_report.py:")
    print("   - 📊 核心报告生成器")
    print("   - 🔍 股票分析逻辑")
    print("   - 📈 技术指标计算")
    print("   - 🎨 HTML报告生成")
    print("   - ⚡ 可单独使用")
    
    print("\n📄 smart_daily_email_sender.py:")
    print("   - 🔗 调用 smart_daily_report.py")
    print("   - 📧 邮件发送功能")
    print("   - 🔄 数据更新管理")
    print("   - ⏰ 定时任务控制")
    print("   - 🚨 错误处理和警报")
    
    print("\n🎯 定时任务方案：")
    print("=" * 40)
    
    print("🔥 方案一：使用内置定时器（推荐）")
    print("   ✅ 简单易用，开箱即用")
    print("   ✅ 自动处理交易日判断")
    print("   ✅ 错误处理和重试机制")
    print("   ⏰ 美股收盘后30分钟执行")
    print("   📅 只在交易日运行")
    
    print("\n🛠️ 方案二：系统定时任务")
    print("   ⚡ 更稳定的系统级调度")
    print("   🔧 需要额外配置")
    print("   💻 适合服务器部署")
    
    print("\n⚙️ 当前定时设置：")
    print("=" * 40)
    
    try:
        # 创建邮件发送器来获取当前配置
        sender = SmartDailyEmailSender()
        close_time = sender.get_market_close_time()
        is_market_day = sender.is_market_day()
        
        print(f"📅 今天是否交易日: {'✅ 是' if is_market_day else '❌ 否'}")
        print(f"⏰ 定时执行时间: {close_time} (北京时间)")
        print(f"🕐 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 监控股票数量: {len(sender.report_generator.watchlist)} 只")
        print(f"💼 持仓股票数量: {len(sender.report_generator.portfolio)} 只")
        print(f"💾 数据源: {sender.report_generator.data_source_type}")
        
    except Exception as e:
        print(f"❌ 获取配置失败: {e}")
    
    print("\n🚀 启动方法：")
    print("=" * 40)
    
    print("1️⃣ 使用内置定时器（推荐）：")
    print("   python smart_daily_email_sender.py")
    print("   # 或")
    print("   python -c \"from smart_daily_email_sender import SmartDailyEmailSender; sender = SmartDailyEmailSender(); sender.setup_daily_schedule(); sender.start_scheduler()\"")
    
    print("\n2️⃣ 立即测试运行：")
    print("   python -c \"from smart_daily_email_sender import SmartDailyEmailSender; sender = SmartDailyEmailSender(); sender.run_immediately()\"")
    
    print("\n3️⃣ Windows 任务计划程序：")
    print("   - 打开任务计划程序")
    print("   - 创建基本任务")
    print(f"   - 程序: {sys.executable}")
    print(f"   - 参数: \"{os.path.abspath('smart_daily_email_sender.py')}\"")
    print(f"   - 起始于: {os.path.dirname(os.path.abspath(__file__))}")
    print("   - 触发器: 每日 04:30 (夏令时) / 05:30 (冬令时)")
    
    print("\n4️⃣ Linux Crontab：")
    print("   # 夏令时 (3-10月)")
    print(f"   30 4 * 3-10 1-5 cd {os.path.dirname(os.path.abspath(__file__))} && python smart_daily_email_sender.py")
    print("   # 冬令时 (11-2月)")
    print(f"   30 5 * 11-2 1-5 cd {os.path.dirname(os.path.abspath(__file__))} && python smart_daily_email_sender.py")
    
    print("\n🎛️ 高级配置：")
    print("=" * 40)
    
    print("📧 邮件配置文件位置：")
    print("   monitor/smart_daily_email_sender.py (第51-57行)")
    
    print("\n📊 持仓配置文件位置：")
    print("   monitor/smart_daily_report.py (第49-65行)")
    
    print("\n⏰ 时间配置调整：")
    print("   修改 get_market_close_time() 方法")
    print("   当前: 收盘后30分钟")
    print("   可改为: 收盘后1小时、2小时等")
    
    print("\n💡 使用建议：")
    print("=" * 40)
    print("1️⃣ 首次使用建议先测试运行")
    print("2️⃣ 确认邮件发送正常后再设置定时")
    print("3️⃣ 建议使用内置定时器，最简单可靠")
    print("4️⃣ 服务器部署可考虑系统级定时任务")
    print("5️⃣ 定期检查日志确保系统正常运行")

# 快速启动函数
def quick_start_scheduler():
    """快速启动定时任务"""
    try:
        print("🚀 快速启动智能日报定时任务...")
        
        # 创建并启动邮件发送器
        sender = SmartDailyEmailSender()
        
        print("✅ 系统初始化成功")
        print(f"⏰ 将在每个交易日 {sender.get_market_close_time()} 自动执行")
        print("📧 按 Ctrl+C 停止程序")
        
        # 设置定时任务
        sender.setup_daily_schedule()
        
        # 启动调度器
        sender.start_scheduler()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def test_run():
    """测试运行"""
    try:
        print("🧪 测试运行智能日报...")
        
        sender = SmartDailyEmailSender()
        success = sender.run_immediately()
        
        if success:
            print("✅ 测试运行成功！")
        else:
            print("❌ 测试运行失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            quick_start_scheduler()
        elif sys.argv[1] == "test":
            test_run()
        else:
            print("使用方法:")
            print("  python scheduler_setup_guide.py        # 显示设置指南")
            print("  python scheduler_setup_guide.py start  # 启动定时任务")
            print("  python scheduler_setup_guide.py test   # 测试运行")
    else:
        main() 