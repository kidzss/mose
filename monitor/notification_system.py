import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict
import json
import os
import logging
from email.utils import formataddr

class NotificationSystem:
    """邮件通知系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        self.alert_levels = {
            'danger': '🔴',
            'warning': '🟡',
            'info': '🔵',
            'opportunity': '🟢'
        }
    
    def _load_config(self):
        """加载邮件配置"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'configs', 'email_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载邮件配置失败: {e}")
            return None
    
    def _format_alert_message(self, alerts: List[Dict], market_state: Dict) -> str:
        """格式化预警消息"""
        message = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; }}
                .alert {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .danger {{ background-color: #ffe6e6; }}
                .warning {{ background-color: #fff3cd; }}
                .info {{ background-color: #e7f5ff; }}
                .opportunity {{ background-color: #d4edda; }}
                .market-state {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>投资组合预警通知</h2>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="market-state">
                <h3>市场状态</h3>
                <p>市场情绪: {market_state['sentiment']}</p>
                <p>VIX指数: {market_state['vix_level']:.2f} ({market_state['vix_change']:+.2f}%)</p>
                <p>建议: {market_state['suggestion']}</p>
            </div>
            
            <h3>预警信息</h3>
        """
        
        # 按级别分组预警信息
        for level in ['danger', 'warning', 'opportunity', 'info']:
            level_alerts = [a for a in alerts if a['level'] == level]
            if level_alerts:
                message += f'<div class="alert {level}">'
                message += f'<h4>{self.alert_levels[level]} {level.title()}级别预警</h4>'
                for alert in level_alerts:
                    message += f'<p>{alert["message"]}</p>'
                message += '</div>'
        
        message += """
            </body>
            </html>
        """
        return message
    
    async def send_email(self, subject, body, is_html=False):
        """发送邮件通知"""
        if not self.config:
            self.logger.error("邮件配置未加载，无法发送邮件")
            return
            
        try:
            # 创建邮件对象
            msg = MIMEMultipart('alternative')
            msg['From'] = formataddr(("Stock Monitor", self.config['sender_email']))
            msg['To'] = self.config['recipient_email']
            msg['Subject'] = subject
            
            # 添加邮件内容
            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type, 'utf-8'))
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)
                
            self.logger.info(f"邮件发送成功: {subject}")
            
        except Exception as e:
            self.logger.error(f"发送邮件失败: {e}")
            raise
    
    def save_config(self, config: Dict, config_path: str) -> bool:
        """保存邮件配置"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.config = config
            return True
        except Exception as e:
            print(f"保存配置时出错: {e}")
            return False 