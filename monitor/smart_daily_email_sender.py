#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import schedule
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.smart_daily_report import SmartDailyReportGenerator
# 简化依赖 - 直接使用内置邮件功能
# from alert_system import AlertSystem
# from notification_manager import NotificationManager
try:
    from config.trading_config import default_config
except ImportError:
    default_config = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartDailyEmailSender")

class SmartDailyEmailSender:
    """智能日报邮件发送器 - 集成数据更新功能"""
    
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
            
            # 数据更新器路径
            self.data_updater_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'data_updater.py'
            )
            
            logger.info("✅ 智能日报邮件发送器初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 智能日报邮件发送器初始化失败: {e}")
            raise
    
    def _init_simple_email_sender(self):
        """初始化简单的邮件发送器"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        self.smtp = None
        logger.info("邮件发送器初始化完成")
    
    def _send_email_direct(self, subject: str, body: str, is_html: bool = False) -> bool:
        """直接发送邮件"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 创建邮件消息
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = subject
            
            # 添加邮件正文
            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                text = msg.as_string()
                server.sendmail(
                    self.email_config['sender_email'],
                    self.email_config['recipient_email'],
                    text
                )
            
            return True
            
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False
    
    def update_market_data(self) -> bool:
        """
        更新市场数据
        
        返回:
            bool: 更新是否成功
        """
        try:
            logger.info("🔄 开始更新市场数据...")
            
            # 检查数据更新器文件是否存在
            if not os.path.exists(self.data_updater_path):
                error_msg = f"数据更新器文件不存在: {self.data_updater_path}"
                logger.error(error_msg)
                self._send_data_update_failure_alert(error_msg)
                return False
            
            # 执行数据更新
            start_time = datetime.now()
            result = subprocess.run(
                [sys.executable, self.data_updater_path],
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ 市场数据更新成功！耗时: {duration:.1f}秒")
                
                # 记录更新日志
                if result.stdout:
                    logger.info(f"更新输出: {result.stdout[-500:]}")  # 只显示最后500字符
                
                return True
            else:
                error_msg = f"数据更新失败，返回码: {result.returncode}"
                if result.stderr:
                    error_msg += f"\n错误信息: {result.stderr}"
                if result.stdout:
                    error_msg += f"\n输出信息: {result.stdout}"
                
                logger.error(error_msg)
                self._send_data_update_failure_alert(error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "数据更新超时（超过30分钟）"
            logger.error(error_msg)
            self._send_data_update_failure_alert(error_msg)
            return False
            
        except Exception as e:
            error_msg = f"数据更新过程发生异常: {str(e)}"
            logger.error(error_msg)
            self._send_data_update_failure_alert(error_msg)
            return False
    
    def _send_data_update_failure_alert(self, error_message: str):
        """发送数据更新失败警报邮件"""
        try:
            subject = "🚨 股票数据更新失败警报"
            
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #dc3545; margin-bottom: 20px;">
                        🚨 数据更新失败警报
                    </h2>
                    
                    <div style="background-color: #fff; padding: 15px; border-left: 4px solid #dc3545; margin: 20px 0;">
                        <h3 style="color: #dc3545; margin-top: 0;">错误详情:</h3>
                        <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;">
{error_message}
                        </pre>
                    </div>
                    
                    <div style="background-color: #fff3cd; padding: 15px; border-radius: 4px; margin: 20px 0;">
                        <h3 style="color: #856404; margin-top: 0;">⚠️ 影响说明:</h3>
                        <ul style="color: #856404;">
                            <li>今日智能日报可能使用昨日数据</li>
                            <li>投资组合盈亏计算可能不准确</li>
                            <li>技术指标分析可能基于过期数据</li>
                        </ul>
                    </div>
                    
                    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 4px; margin: 20px 0;">
                        <h3 style="color: #0c5460; margin-top: 0;">🔧 建议操作:</h3>
                        <ol style="color: #0c5460;">
                            <li>检查网络连接是否正常</li>
                            <li>手动运行数据更新: <code>python data/data_updater.py</code></li>
                            <li>检查数据库连接状态</li>
                            <li>如问题持续，请检查Yahoo Finance API状态</li>
                        </ol>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;">
                        <p style="color: #6c757d; font-size: 12px;">
                            时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                            系统: 智能股票日报邮件系统
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self._send_email_direct(subject, body, is_html=True)
            logger.info("✅ 数据更新失败警报邮件已发送")
            
        except Exception as e:
            logger.error(f"❌ 发送数据更新失败警报时出错: {e}")
    
    def run_daily_report_with_data_update(self) -> bool:
        """
        执行完整的日报流程：数据更新 + 报告生成 + 邮件发送
        
        返回:
            bool: 整个流程是否成功
        """
        try:
            logger.info("🚀 开始执行每日智能报告流程...")
            
            # 步骤1: 更新市场数据
            logger.info("📊 步骤1: 更新市场数据")
            data_update_success = self.update_market_data()
            
            if not data_update_success:
                logger.warning("⚠️ 数据更新失败，但继续生成报告（使用现有数据）")
            
            # 步骤2: 生成智能报告
            logger.info("📈 步骤2: 生成智能报告")
            try:
                html_content = self.report_generator.generate_report()
                if not html_content:
                    logger.error("❌ 报告生成失败")
                    return False
                
                logger.info("✅ 报告生成成功")
                
            except Exception as e:
                logger.error(f"❌ 报告生成失败: {e}")
                return False
            
            # 步骤3: 发送邮件
            logger.info("📧 步骤3: 发送邮件报告")
            
            # 根据数据更新状态调整邮件标题
            if data_update_success:
                subject = f"📊 每日智能股票分析报告 - {datetime.now().strftime('%Y年%m月%d日')}"
            else:
                subject = f"📊 每日智能股票分析报告 - {datetime.now().strftime('%Y年%m月%d日')} [数据更新异常]"
            
            # 发送HTML邮件
            success = self._send_email_direct(
                subject=subject,
                body=html_content,
                is_html=True
            )
            
            if success:
                logger.info("✅ 每日智能报告发送成功！")
                return True
            else:
                logger.error("❌ 每日智能报告发送失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 执行每日报告流程时出错: {e}")
            # 发送错误警报
            try:
                error_subject = "🚨 每日智能报告系统错误"
                error_body = f"""
                <div style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2 style="color: #dc3545;">系统错误</h2>
                    <p>每日智能报告系统运行时发生错误：</p>
                    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">{str(e)}</pre>
                    <p>时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """
                self._send_email_direct(error_subject, error_body, is_html=True)
            except:
                pass
            
            return False
    
    def is_market_day(self) -> bool:
        """
        判断今天是否为交易日（周一到周五，排除美国主要节假日）
        
        返回:
            bool: 是否为交易日
        """
        today = datetime.now()
        
        # 检查是否为周末
        if today.weekday() >= 5:  # 周六=5, 周日=6
            return False
        
        # 简单的美国主要节假日检查（可以后续扩展）
        # 这里可以添加更详细的节假日判断逻辑
        
        return True
    
    def get_market_close_time(self) -> str:
        """
        获取美股收盘后30分钟的北京时间
        
        返回:
            str: 时间字符串 (HH:MM格式)
        """
        # 简单判断夏令时和冬令时
        # 实际应用中可以使用pytz库进行更精确的时区转换
        now = datetime.now()
        month = now.month
        
        # 大致的夏令时判断：3月到10月为夏令时
        if 3 <= month <= 10:
            # 夏令时：美东16:00 = 北京04:00，加30分钟 = 04:30
            return "04:30"
        else:
            # 冬令时：美东16:00 = 北京05:00，加30分钟 = 05:30
            return "05:30"
    
    def setup_daily_schedule(self):
        """设置每日定时任务 - 收盘后30分钟执行"""
        try:
            # 获取收盘后30分钟的时间
            schedule_time = self.get_market_close_time()
            
            # 设置定时任务，只在交易日执行
            def run_if_market_day():
                if self.is_market_day():
                    logger.info(f"📅 今天是交易日，开始执行每日报告流程...")
                    self.run_daily_report_with_data_update()
                else:
                    logger.info(f"📅 今天不是交易日，跳过报告生成")
            
            # 每天在指定时间执行
            schedule.every().day.at(schedule_time).do(run_if_market_day)
            
            logger.info(f"✅ 定时任务已设置：每个交易日 {schedule_time} 执行")
            logger.info(f"🕐 当前系统时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"❌ 设置定时任务失败: {e}")
    
    def start_scheduler(self):
        """启动定时任务调度器"""
        try:
            logger.info("🚀 启动定时任务调度器...")
            logger.info("按 Ctrl+C 可停止程序")
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            logger.info("👋 用户中断，程序退出")
        except Exception as e:
            logger.error(f"❌ 定时任务调度器运行错误: {e}")
    
    def run_immediately(self) -> bool:
        """立即执行一次完整的报告流程（用于测试）"""
        logger.info("🧪 立即执行报告流程（测试模式）...")
        return self.run_daily_report_with_data_update()
    
    def send_test_email(self) -> bool:
        """发送测试邮件"""
        try:
            test_subject = "📧 智能日报邮件系统测试"
            test_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #28a745;">✅ 系统测试成功</h2>
                    <p>智能日报邮件系统运行正常！</p>
                    
                    <h3>📋 系统配置信息:</h3>
                    <ul>
                        <li>持仓股票: {len(self.report_generator.portfolio)} 只</li>
                        <li>观察股票: {len(self.report_generator.watch_targets)} 只</li>
                        <li>总投资金额: ${self.report_generator.total_stock_investment:,.2f}</li>
                        <li>股票配置比例: {self.report_generator.portfolio_allocation}%</li>
                        <li>数据源: {self.report_generator.data_source_type}</li>
                    </ul>
                    
                    <h3>⏰ 定时设置:</h3>
                    <p>每个交易日 <strong>{self.get_market_close_time()}</strong> 自动发送（美股收盘后30分钟）</p>
                    
                    <p style="text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px;">
                        测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            success = self._send_email_direct(test_subject, test_body, is_html=True)
            if success:
                logger.info("✅ 测试邮件发送成功！")
            return success
            
        except Exception as e:
            logger.error(f"❌ 发送测试邮件失败: {e}")
            return False


# 独立运行脚本
if __name__ == "__main__":
    try:
        logger.info("🚀 启动智能日报邮件系统...")
        
        # 创建邮件发送器
        email_sender = SmartDailyEmailSender()
        
        # 设置并启动定时任务
        email_sender.setup_daily_schedule()
        email_sender.start_scheduler()
        
    except Exception as e:
        logger.error(f"❌ 系统启动失败: {e}")
        sys.exit(1)