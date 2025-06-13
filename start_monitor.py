#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitor.smart_daily_email_sender import SmartDailyEmailSender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MonitorService")

def main():
    """启动持仓分析报告监控服务"""
    print("=" * 60)
    print("🚀 启动持仓分析报告监控服务")
    print("=" * 60)
    
    try:
        # 创建邮件发送器
        logger.info("初始化智能日报邮件系统...")
        email_sender = SmartDailyEmailSender()
        
        # 显示配置信息
        print(f"\n📊 投资组合配置:")
        print(f"   📈 持仓股票数量: {len(email_sender.report_generator.portfolio)}")
        print(f"   👀 观察股票数量: {len(email_sender.report_generator.watch_targets)}")
        print(f"   💰 总投资金额: ${email_sender.report_generator.total_stock_investment:,.2f}")
        print(f"   📧 收件邮箱: {email_sender.email_config['recipient_email']}")
        print(f"   ⏰ 定时发送时间: 每个交易日 {email_sender.get_market_close_time()}")
        
        print(f"\n🕐 服务启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 设置并启动定时任务
        logger.info("设置定时任务...")
        email_sender.setup_daily_schedule()
        
        logger.info("启动定时任务调度器...")
        print("✅ 监控服务已启动！")
        print("📅 将在每个交易日美股收盘后30分钟自动发送持仓分析报告")
        print("🛑 按 Ctrl+C 可停止服务")
        print("=" * 60)
        
        # 启动调度器（这会一直运行）
        email_sender.start_scheduler()
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断服务")
        print("\n👋 监控服务已停止")
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 