"""
统一邮件发送API
提供简单、统一的邮件发送接口，封装所有邮件配置细节
任何功能只需要调用简单的API就能发送邮件，无需关心邮件配置
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import markdown
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class UnifiedEmailAPI:
    """
    统一邮件发送API
    
    使用方式：
    1. 简单文本邮件：
       UnifiedEmailAPI.send_text(subject="测试", content="Hello World")
    
    2. HTML邮件：
       UnifiedEmailAPI.send_html(subject="报告", html_content="<h1>报告内容</h1>")
    
    3. Markdown邮件：
       UnifiedEmailAPI.send_markdown(subject="分析", md_content="# 分析报告")
    
    4. 带附件的邮件：
       UnifiedEmailAPI.send_with_attachments(subject="数据", content="请查看附件", 
                                           attachments=["file1.pdf", "file2.xlsx"])
    """
    
    # 单例模式，确保全局唯一配置
    _instance = None
    _config_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnifiedEmailAPI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config_loaded:
            self._load_config()
            self._config_loaded = True
    
    def _load_config(self):
        """加载邮件配置"""
        try:
            # 配置优先级：环境变量 > 配置文件 > 默认值
            config = self._load_config_file()
            
            self.smtp_server = os.getenv('EMAIL_SMTP_SERVER') or config.get('smtp_server', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
            self.sender_email = os.getenv('EMAIL_SENDER') or config.get('sender_email', '')
            self.sender_password = os.getenv('EMAIL_PASSWORD') or config.get('sender_password', '')
            self.receiver_email = os.getenv('EMAIL_RECEIVER') or config.get('recipient_email', '')
            
            # 验证必要配置
            if not all([self.sender_email, self.sender_password, self.receiver_email]):
                logger.warning("邮件配置不完整，请检查环境变量或配置文件")
                
        except Exception as e:
            logger.error(f"加载邮件配置失败: {e}")
            # 设置默认值
            self.smtp_server = 'smtp.gmail.com'
            self.smtp_port = 587
            self.sender_email = ''
            self.sender_password = ''
            self.receiver_email = ''
    
    def _load_config_file(self) -> dict:
        """加载配置文件"""
        config_paths = [
            'monitor/configs/email_config.json',
            'configs/email_config.json',
            'email_config.json',
            str(Path.home() / '.mose_email_config.json')
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"读取配置文件 {config_path} 失败: {e}")
        
        return {}
    
    def _send_email(self, subject: str, content: str, content_type: str = 'html', 
                   attachments: List[str] = None, recipient_email: str = None) -> bool:
        """
        核心邮件发送方法
        
        Args:
            subject: 邮件主题
            content: 邮件内容
            content_type: 内容类型 ('html' 或 'plain')
            attachments: 附件文件路径列表
            recipient_email: 收件人邮箱，None则使用默认配置
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 验证配置
            if not all([self.sender_email, self.sender_password, self.receiver_email]):
                logger.error("邮件配置不完整，无法发送邮件")
                return False
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email or self.receiver_email
            msg['Subject'] = subject
            
            # 添加正文
            msg.attach(MIMEText(content, content_type, 'utf-8'))
            
            # 添加附件
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                        
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(file_path)}'
                        )
                        msg.attach(part)
                    else:
                        logger.warning(f"附件文件不存在: {file_path}")
            
            # 发送邮件
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ 邮件发送成功: {subject} -> {recipient_email or self.receiver_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return False
    
    def _create_html_template(self, content: str, title: str = None) -> str:
        """创建带样式的HTML模板"""
        title = title or "系统通知"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                h3 {{
                    color: #2980b9;
                    margin-top: 25px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                tr:hover {{
                    background-color: #e8f4f8;
                }}
                .highlight {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 4px solid #ffc107;
                    margin: 20px 0;
                }}
                .success {{
                    background-color: #d4edda;
                    padding: 15px;
                    border-left: 4px solid #28a745;
                    margin: 20px 0;
                }}
                .info {{
                    background-color: #d1ecf1;
                    padding: 15px;
                    border-left: 4px solid #17a2b8;
                    margin: 20px 0;
                }}
                .warning {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 4px solid #ffc107;
                    margin: 20px 0;
                }}
                .error {{
                    background-color: #f8d7da;
                    padding: 15px;
                    border-left: 4px solid #dc3545;
                    margin: 20px 0;
                }}
                ul, ol {{
                    padding-left: 25px;
                }}
                li {{
                    margin: 8px 0;
                }}
                code {{
                    background-color: #f8f9fa;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                }}
                .emoji {{
                    font-size: 1.2em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {content}
                <div class="footer">
                    <p>📧 此邮件由量化交易系统自动发送 | 生成时间: {timestamp}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template

# 全局API函数，方便直接调用
def send_text(subject: str, content: str, recipient_email: str = None) -> bool:
    """发送纯文本邮件"""
    api = UnifiedEmailAPI()
    return api._send_email(subject, content, 'plain', None, recipient_email)

def send_html(subject: str, html_content: str, recipient_email: str = None) -> bool:
    """发送HTML邮件"""
    api = UnifiedEmailAPI()
    return api._send_email(subject, html_content, 'html', None, recipient_email)

def send_markdown(subject: str, md_content: str, recipient_email: str = None) -> bool:
    """发送Markdown邮件（自动转换为HTML）"""
    try:
        # 转换Markdown为HTML
        extensions = [
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.toc'
        ]
        html_content = markdown.markdown(md_content, extensions=extensions)
        
        # 使用HTML模板
        api = UnifiedEmailAPI()
        styled_html = api._create_html_template(html_content, subject)
        
        return api._send_email(subject, styled_html, 'html', None, recipient_email)
    except Exception as e:
        logger.error(f"Markdown转换失败: {e}")
        return False

def send_with_attachments(subject: str, content: str, attachments: List[str], 
                         content_type: str = 'html', recipient_email: str = None) -> bool:
    """发送带附件的邮件"""
    api = UnifiedEmailAPI()
    return api._send_email(subject, content, content_type, attachments, recipient_email)

def send_report(subject: str, report_data: Dict[str, Any], report_type: str = 'html', 
                recipient_email: str = None) -> bool:
    """
    发送报告邮件
    
    Args:
        subject: 邮件主题
        report_data: 报告数据字典
        report_type: 报告类型 ('html', 'markdown', 'json')
        recipient_email: 收件人邮箱
    """
    try:
        if report_type == 'json':
            # JSON格式报告
            content = json.dumps(report_data, indent=2, ensure_ascii=False)
            return send_text(subject, content, recipient_email)
        
        elif report_type == 'markdown':
            # Markdown格式报告
            md_content = _dict_to_markdown(report_data)
            return send_markdown(subject, md_content, recipient_email)
        
        else:  # html
            # HTML格式报告
            html_content = _dict_to_html(report_data)
            api = UnifiedEmailAPI()
            styled_html = api._create_html_template(html_content, subject)
            return api._send_email(subject, styled_html, 'html', None, recipient_email)
            
    except Exception as e:
        logger.error(f"发送报告邮件失败: {e}")
        return False

def _dict_to_markdown(data: Dict[str, Any], level: int = 0) -> str:
    """将字典转换为Markdown格式"""
    md_content = ""
    indent = "  " * level
    
    for key, value in data.items():
        if isinstance(value, dict):
            md_content += f"{indent}- **{key}**:\n"
            md_content += _dict_to_markdown(value, level + 1)
        elif isinstance(value, list):
            md_content += f"{indent}- **{key}**:\n"
            for item in value:
                if isinstance(item, dict):
                    md_content += _dict_to_markdown(item, level + 1)
                else:
                    md_content += f"{indent}  - {item}\n"
        else:
            md_content += f"{indent}- **{key}**: {value}\n"
    
    return md_content

def _dict_to_html(data: Dict[str, Any]) -> str:
    """将字典转换为HTML格式"""
    def _format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:,.2f}" if isinstance(value, float) else str(value)
        return str(value)
    
    html_content = "<div class='report'>"
    
    for key, value in data.items():
        html_content += f"<h3>{key}</h3>"
        
        if isinstance(value, dict):
            html_content += "<table><tr><th>项目</th><th>值</th></tr>"
            for k, v in value.items():
                html_content += f"<tr><td>{k}</td><td>{_format_value(v)}</td></tr>"
            html_content += "</table>"
        
        elif isinstance(value, list):
            html_content += "<ul>"
            for item in value:
                if isinstance(item, dict):
                    html_content += "<li><table><tr><th>项目</th><th>值</th></tr>"
                    for k, v in item.items():
                        html_content += f"<tr><td>{k}</td><td>{_format_value(v)}</td></tr>"
                    html_content += "</table></li>"
                else:
                    html_content += f"<li>{_format_value(item)}</li>"
            html_content += "</ul>"
        
        else:
            html_content += f"<p>{_format_value(value)}</p>"
    
    html_content += "</div>"
    return html_content

def test_email_config() -> bool:
    """测试邮件配置"""
    try:
        api = UnifiedEmailAPI()
        
        if not all([api.sender_email, api.sender_password, api.receiver_email]):
            logger.error("邮件配置不完整")
            return False
        
        # 发送测试邮件
        test_subject = "🧪 邮件配置测试"
        test_content = f"""
        <h2>邮件配置测试</h2>
        <p>如果您收到这封邮件，说明邮件配置正确！</p>
        <div class="info">
            <strong>配置信息：</strong><br>
            SMTP服务器: {api.smtp_server}<br>
            SMTP端口: {api.smtp_port}<br>
            发件人: {api.sender_email}<br>
            收件人: {api.receiver_email}<br>
            测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """
        
        styled_html = api._create_html_template(test_content, "邮件配置测试")
        success = api._send_email(test_subject, styled_html, 'html')
        
        if success:
            logger.info("✅ 邮件配置测试成功")
        else:
            logger.error("❌ 邮件配置测试失败")
        
        return success
        
    except Exception as e:
        logger.error(f"邮件配置测试失败: {e}")
        return False

# 便捷的类方法调用
class EmailAPI:
    """便捷的邮件API类"""
    
    @staticmethod
    def text(subject: str, content: str, recipient_email: str = None) -> bool:
        """发送文本邮件"""
        return send_text(subject, content, recipient_email)
    
    @staticmethod
    def html(subject: str, html_content: str, recipient_email: str = None) -> bool:
        """发送HTML邮件"""
        return send_html(subject, html_content, recipient_email)
    
    @staticmethod
    def markdown(subject: str, md_content: str, recipient_email: str = None) -> bool:
        """发送Markdown邮件"""
        return send_markdown(subject, md_content, recipient_email)
    
    @staticmethod
    def with_attachments(subject: str, content: str, attachments: List[str], 
                        content_type: str = 'html', recipient_email: str = None) -> bool:
        """发送带附件的邮件"""
        return send_with_attachments(subject, content, attachments, content_type, recipient_email)
    
    @staticmethod
    def report(subject: str, report_data: Dict[str, Any], report_type: str = 'html', 
               recipient_email: str = None) -> bool:
        """发送报告邮件"""
        return send_report(subject, report_data, report_type, recipient_email)
    
    @staticmethod
    def test() -> bool:
        """测试邮件配置"""
        return test_email_config()

# 导出主要接口
__all__ = [
    'UnifiedEmailAPI',
    'EmailAPI',
    'send_text',
    'send_html', 
    'send_markdown',
    'send_with_attachments',
    'send_report',
    'test_email_config'
]
