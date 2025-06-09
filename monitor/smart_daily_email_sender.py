#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .smart_daily_report import SmartDailyReportGenerator
from .alert_system import AlertSystem
from .notification_manager import NotificationManager
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartDailyEmailSender")

class SmartDailyEmailSender:
    """智能日报邮件发送器"""
    
    def __init__(self, portfolio=None, watch_targets=None):
        """
        初始化智能日报邮件发送器
        
        参数:
            portfolio: 用户投资组合信息（可选）
            watch_targets: 观察目标股票（可选）
        """
        try:
            # 初始化智能日报生成器
            self.report_generator = SmartDailyReportGenerator(
                portfolio=portfolio,
                watch_targets=watch_targets
            )
            
            # 初始化邮件发送系统
            # 不使用default_config，直接使用email_config
            self.alert_system = None
            self.notification_manager = None
            
            # 邮件配置
            self.email_config = {
                'sender_email': 'kidzss@gmail.com',
                'recipient_email': 'kidzss@gmail.com',
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_password': 'wlkp dbbz xpgk rkhy'  # App Password
            }
            
            # 简化的邮件发送器
            self._init_simple_email_sender()
            
            logger.info("✅ 智能日报邮件发送器初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 智能日报邮件发送器初始化失败: {e}")
            raise
    
    def _init_simple_email_sender(self):
        """初始化简单的邮件发送器"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        self._smtp_config = self.email_config
        
    def _send_email_direct(self, subject: str, body: str, is_html: bool = False):
        """直接发送邮件，不依赖AlertSystem"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            logger.info(f"准备发送邮件: {subject}")
            
            msg = MIMEMultipart()
            msg['From'] = self._smtp_config['sender_email']
            msg['To'] = self._smtp_config['recipient_email']
            msg['Subject'] = subject
            
            # 根据 is_html 参数决定内容类型
            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            logger.info("正在连接SMTP服务器...")
            with smtplib.SMTP(self._smtp_config['smtp_server'], self._smtp_config['smtp_port']) as server:
                logger.info("正在启动TLS加密...")
                server.starttls()
                logger.info("正在登录SMTP服务器...")
                server.login(self._smtp_config['sender_email'], self._smtp_config['sender_password'])
                logger.info("正在发送邮件...")
                server.send_message(msg)
            
            logger.info(f"✅ 邮件发送成功: {subject}")
            return True
        
        except Exception as e:
            logger.error(f"❌ 发送邮件失败: {str(e)}")
            return False
    
    def generate_and_send_daily_report(self):
        """生成并发送每日智能报告"""
        try:
            logger.info("🚀 开始生成每日智能报告...")
            
            # 生成智能日报
            report_html = self.report_generator.generate_report()
            
            # 如果返回的是文件名，读取文件内容
            if isinstance(report_html, str) and report_html.endswith('.html'):
                with open(report_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                logger.info(f"📄 从文件读取报告内容: {report_html}")
            else:
                html_content = report_html
                logger.info("📄 直接获取报告HTML内容")
            
            # 构建邮件主题
            current_date = datetime.now().strftime('%Y年%m月%d日')
            
            # 统计持仓股票数量
            portfolio_count = len(self.report_generator.portfolio)
            watch_count = len(self.report_generator.watch_targets)
            total_stocks = portfolio_count + watch_count
            
            subject = f"📊 智能投资组合日报 - {current_date} ({total_stocks}只股票分析)"
            
            # 发送HTML邮件
            success = self._send_email_direct(
                subject=subject,
                body=html_content,
                is_html=True
            )
            
            if success:
                logger.info("✅ 每日智能报告发送成功！")
            else:
                logger.error("❌ 每日智能报告发送失败")
                return False
            
            # 生成发送总结
            summary = {
                "发送时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "持仓股票": list(self.report_generator.portfolio.keys()),
                "观察股票": list(self.report_generator.watch_targets.keys()),
                "总投资金额": f"${self.report_generator.total_stock_investment:,.2f}",
                "投资组合配置": f"{self.report_generator.portfolio_allocation}%"
            }
            
            logger.info("📈 发送总结:")
            for key, value in summary.items():
                logger.info(f"  - {key}: {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 生成或发送每日报告失败: {e}")
            
            # 发送错误通知邮件
            try:
                error_subject = f"⚠️ 智能日报生成失败 - {datetime.now().strftime('%Y年%m月%d日')}"
                error_body = f"""
                <html>
                <body>
                    <h2>⚠️ 智能日报生成失败</h2>
                    <p><strong>时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>错误信息:</strong> {str(e)}</p>
                    <p><strong>建议:</strong> 请检查数据源连接和配置设置</p>
                </body>
                </html>
                """
                self._send_email_direct(error_subject, error_body, is_html=True)
            except:
                logger.error("连错误通知邮件都发送失败")
                
            return False
    
    def send_test_email(self):
        """发送测试邮件，验证邮件配置"""
        try:
            logger.info("📧 发送测试邮件...")
            
            test_subject = "🧪 智能日报系统测试邮件"
            test_body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .header {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; }}
                    .content {{ padding: 15px; }}
                    .success {{ color: #28a745; }}
                    .info {{ color: #17a2b8; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>🧪 智能日报系统测试</h2>
                    <p class="success">✅ 邮件发送功能正常</p>
                </div>
                <div class="content">
                    <p><strong>测试时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>系统状态:</strong> <span class="success">运行正常</span></p>
                    <p><strong>监控股票:</strong> {len(self.report_generator.portfolio)} 只持仓 + {len(self.report_generator.watch_targets)} 只观察</p>
                    <p><strong>数据源:</strong> <span class="info">{self.report_generator.data_source_type}</span></p>
                    <p><strong>下次报告时间:</strong> 下个交易日收盘后</p>
                    
                    <h3>📋 当前配置:</h3>
                    <ul>
                        <li>持仓股票: {', '.join(self.report_generator.portfolio.keys())}</li>
                        <li>观察股票: {', '.join(self.report_generator.watch_targets.keys())}</li>
                        <li>总投资金额: ${self.report_generator.total_stock_investment:,.2f}</li>
                        <li>股票配置比例: {self.report_generator.portfolio_allocation}%</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            success = self._send_email_direct(test_subject, test_body, is_html=True)
            if success:
                logger.info("✅ 测试邮件发送成功！")
            return success
            
        except Exception as e:
            logger.error(f"❌ 测试邮件发送失败: {e}")
            return False
    
    def setup_daily_schedule(self):
        """设置每日定时任务"""
        try:
            # 设置为每个交易日美股收盘后30分钟发送 (美东时间16:30)
            # 对应北京时间：夏令时5:30，冬令时6:30
            # 这里设置为北京时间早上6:00，避免夏令时/冬令时的复杂性
            
            schedule.every().monday.at("06:00").do(self.generate_and_send_daily_report)
            schedule.every().tuesday.at("06:00").do(self.generate_and_send_daily_report)
            schedule.every().wednesday.at("06:00").do(self.generate_and_send_daily_report)
            schedule.every().thursday.at("06:00").do(self.generate_and_send_daily_report)
            schedule.every().friday.at("06:00").do(self.generate_and_send_daily_report)
            
            logger.info("⏰ 每日智能报告定时任务已设置")
            logger.info("📅 发送时间: 周一至周五 早上6:00 (北京时间)")
            logger.info("📈 这相当于美股收盘后约2小时")
            
        except Exception as e:
            logger.error(f"❌ 设置定时任务失败: {e}")
            raise
    
    def start_scheduler(self):
        """启动定时任务调度器"""
        try:
            logger.info("🎯 启动智能日报定时任务调度器...")
            logger.info("📋 当前任务列表:")
            for job in schedule.jobs:
                logger.info(f"  - {job}")
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            logger.info("⏹️ 用户中断，停止定时任务")
        except Exception as e:
            logger.error(f"❌ 定时任务运行错误: {e}")
            raise
    
    def run_immediately(self):
        """立即运行一次报告生成（用于测试）"""
        logger.info("🚀 立即执行一次智能日报生成...")
        return self.generate_and_send_daily_report()

def main():
    """主函数 - 用于独立运行"""
    try:
        logger.info("🎯 启动智能日报邮件发送系统...")
        
        # 创建邮件发送器实例（使用默认配置）
        email_sender = SmartDailyEmailSender()
        
        # 发送测试邮件
        logger.info("📧 首先发送测试邮件验证配置...")
        if email_sender.send_test_email():
            logger.info("✅ 测试邮件发送成功，系统配置正常")
        else:
            logger.error("❌ 测试邮件发送失败，请检查配置")
            return
        
        # 立即生成一份报告用于测试
        logger.info("📊 生成测试报告...")
        if email_sender.run_immediately():
            logger.info("✅ 测试报告生成并发送成功")
        else:
            logger.error("❌ 测试报告生成失败")
            return
        
        # 设置定时任务
        email_sender.setup_daily_schedule()
        
        # 启动调度器
        email_sender.start_scheduler()
        
    except Exception as e:
        logger.error(f"❌ 智能日报邮件系统启动失败: {e}")
        raise

if __name__ == "__main__":
    main() 