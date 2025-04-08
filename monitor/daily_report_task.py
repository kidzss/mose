import json
import time
import schedule
import logging
from datetime import datetime
from pathlib import Path
from monitor.portfolio_monitor import PortfolioMonitor
from monitor.report_generator import ReportGenerator
from monitor.trading_monitor import AlertSystem
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='daily_report.log'
)
logger = logging.getLogger("DailyReport")

def send_daily_report():
    """发送每日投资组合报告"""
    try:
        logger.info("开始生成每日报告...")
        
        # 加载配置文件
        config_path = Path(__file__).parent / "configs" / "portfolio_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # 初始化组件
        portfolio_monitor = PortfolioMonitor(
            positions=config["positions"],
            monitor_config=config["monitor_config"]
        )
        report_generator = ReportGenerator()
        alert_system = AlertSystem(default_config)
        
        # 更新持仓信息
        portfolio_monitor.update_positions()
        
        # 生成报告
        report_html = report_generator.generate_daily_report(portfolio_monitor)
        
        # 发送邮件
        subject = f"每日投资组合报告 - {datetime.now().strftime('%Y-%m-%d')}"
        alert_system.send_email(subject, report_html)
        
        logger.info("每日报告发送成功")
        
    except Exception as e:
        logger.error(f"生成或发送每日报告时出错: {str(e)}")

def main():
    """主函数"""
    logger.info("启动每日报告任务...")
    
    # 设置定时任务 - 每个交易日（周一到周五）下午4点发送
    schedule.every().monday.at("16:00").do(send_daily_report)
    schedule.every().tuesday.at("16:00").do(send_daily_report)
    schedule.every().wednesday.at("16:00").do(send_daily_report)
    schedule.every().thursday.at("16:00").do(send_daily_report)
    schedule.every().friday.at("16:00").do(send_daily_report)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        logger.info("任务被用户中断")
    except Exception as e:
        logger.error(f"运行定时任务时出错: {str(e)}")

if __name__ == "__main__":
    main() 