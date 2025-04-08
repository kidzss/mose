import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict
import json
import os

class NotificationSystem:
    """邮件通知系统"""
    
    def __init__(self, config_path: str = None):
        self.email_config = self._load_config(config_path)
        self.alert_levels = {
            'danger': '🔴',
            'warning': '🟡',
            'info': '🔵',
            'opportunity': '🟢'
        }
    
    def _load_config(self, config_path: str = None) -> Dict:
        """加载邮件配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',  # 需要配置
            'sender_password': '',  # 需要配置
            'recipient_email': ''  # 需要配置
        }
    
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
    
    def send_notification(self, alerts: List[Dict], market_state: Dict) -> bool:
        """发送预警通知"""
        try:
            if not alerts:
                return True
                
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"投资组合预警通知 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            
            html_content = self._format_alert_message(alerts, market_state)
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"发送通知时出错: {e}")
            return False
    
    def save_config(self, config: Dict, config_path: str) -> bool:
        """保存邮件配置"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.email_config = config
            return True
        except Exception as e:
            print(f"保存配置时出错: {e}")
            return False 