"""
邮件发送模块 - 支持股票筛选结果的HTML邮件发送
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import markdown
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EmailSender:
    """邮件发送器"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = None):
        """
        初始化邮件发送器
        
        Args:
            smtp_server: SMTP服务器地址
            smtp_port: SMTP端口
        """
        # 首先尝试从配置文件读取
        config = self._load_email_config()
        
        self.smtp_server = smtp_server or config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = smtp_port or config.get('smtp_port', 587)
        
        # 优先使用配置文件，然后是环境变量
        self.sender_email = config.get('sender_email') or os.getenv('EMAIL_SENDER', '')
        self.sender_password = config.get('sender_password') or os.getenv('EMAIL_PASSWORD', '')
        self.receiver_email = config.get('recipient_email') or os.getenv('EMAIL_RECEIVER', '')
    
    def _load_email_config(self) -> dict:
        """加载邮件配置文件"""
        try:
            import json
            config_paths = [
                'monitor/configs/email_config.json',
                'configs/email_config.json',
                'email_config.json'
            ]
            
            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            return {}
        except Exception as e:
            logger.warning(f"加载邮件配置失败: {e}")
            return {}
        
    def convert_md_to_html(self, md_content: str) -> str:
        """将Markdown内容转换为HTML"""
        try:
            # 配置markdown扩展
            extensions = [
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.toc'
            ]
            
            # 转换为HTML
            html_content = markdown.markdown(md_content, extensions=extensions)
            
            # 添加CSS样式
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>股票筛选报告</title>
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
                    {html_content}
                    <div class="footer">
                        <p>📧 此邮件由量化交易系统自动发送 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return styled_html
            
        except Exception as e:
            logger.error(f"Markdown转HTML失败: {e}")
            return f"<html><body><h1>转换失败</h1><pre>{md_content}</pre></body></html>"
    
    def create_screening_report_html(self, results: List[Dict], summary: Dict) -> str:
        """创建股票筛选报告的HTML内容"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 构建HTML报告
            html_content = f"""
            <h1>🚀 股票筛选报告</h1>
            
            <div class="info">
                <h3>📊 筛选摘要</h3>
                <ul>
                    <li><strong>筛选时间</strong>: {timestamp}</li>
                    <li><strong>分析股票数</strong>: {summary.get('total_stocks_analyzed', 0)} 只</li>
                    <li><strong>发现优质股票</strong>: {summary.get('qualified_stocks_found', 0)} 只</li>
                    <li><strong>高质量股票</strong>: {summary.get('high_quality_stocks', 0)} 只</li>
                    <li><strong>最佳股票</strong>: {summary.get('best_stock', 'N/A')} (评分: {summary.get('best_score', 0):.1f})</li>
                </ul>
            </div>
            
            <h2>🏆 TOP 筛选结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>股票代码</th>
                        <th>多因子评分</th>
                        <th>质量因子</th>
                        <th>夏普比率</th>
                        <th>最大回撤</th>
                        <th>当前价格</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # 添加股票数据
            for i, stock in enumerate(results[:15], 1):  # 显示前15只
                quality_class = "success" if stock['quality_factor'] > 0.6 else ""
                html_content += f"""
                    <tr class="{quality_class}">
                        <td>{i}</td>
                        <td><strong>{stock['symbol']}</strong></td>
                        <td>{stock['multifactor_score']:.1f}</td>
                        <td>{stock['quality_factor']:.3f}</td>
                        <td>{stock['sharpe_ratio']:.2f}</td>
                        <td>{stock['max_drawdown']:.2%}</td>
                        <td>${stock['current_price']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </tbody>
            </table>
            """
            
            # 添加最佳股票分析
            if results:
                best_stock = results[0]
                html_content += f"""
                <div class="highlight">
                    <h3>🎯 最佳投资标的: {best_stock['symbol']}</h3>
                    <ul>
                        <li><strong>综合评分</strong>: {best_stock['multifactor_score']:.1f}/100</li>
                        <li><strong>质量因子</strong>: {best_stock['quality_factor']:.3f} (权重30%)</li>
                        <li><strong>夏普比率</strong>: {best_stock['sharpe_ratio']:.2f}</li>
                        <li><strong>最大回撤</strong>: {best_stock['max_drawdown']:.2%}</li>
                        <li><strong>当前价格</strong>: ${best_stock['current_price']:.2f}</li>
                    </ul>
                </div>
                """
            
            # 添加投资建议
            html_content += """
            <h2>💡 投资建议</h2>
            <div class="success">
                <h4>🎯 核心策略</h4>
                <ol>
                    <li><strong>重点关注</strong>: 质量因子>0.6的高质量股票</li>
                    <li><strong>风险控制</strong>: 优先选择最大回撤<20%的股票</li>
                    <li><strong>分散投资</strong>: 建议配置前10只股票，单只占比不超过15%</li>
                    <li><strong>定期调整</strong>: 每月重新筛选，动态优化组合</li>
                </ol>
            </div>
            """
            
            # 直接返回HTML内容，添加样式包装
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>股票筛选报告</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background-color: #f5f5f5;
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1000px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
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
                        border-radius: 5px;
                    }}
                    .success {{
                        background-color: #d4edda;
                        padding: 15px;
                        border-left: 4px solid #28a745;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    .info {{
                        background-color: #d1ecf1;
                        padding: 15px;
                        border-left: 4px solid #17a2b8;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    ul, ol {{
                        padding-left: 25px;
                    }}
                    li {{
                        margin: 8px 0;
                    }}
                    .footer {{
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        text-align: center;
                        color: #666;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    {html_content}
                    <div class="footer">
                        <p>📧 此邮件由量化交易系统自动发送 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return styled_html
            
        except Exception as e:
            logger.error(f"创建HTML报告失败: {e}")
            return f"<html><body><h1>报告生成失败</h1><p>{str(e)}</p></body></html>"
    
    def send_screening_results(self, results: List[Dict], summary: Dict, 
                             subject: str = None, attach_json: bool = True) -> bool:
        """
        发送股票筛选结果邮件
        
        Args:
            results: 筛选结果列表
            summary: 筛选摘要信息
            subject: 邮件主题
            attach_json: 是否附加JSON文件
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if not all([self.sender_email, self.sender_password, self.receiver_email]):
                logger.error("邮件配置不完整，请设置环境变量: EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER")
                return False
            
            # 创建邮件
            msg = MIMEMultipart('alternative')
            
            # 设置邮件主题
            if not subject:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                best_stock = summary.get('best_stock', 'N/A')
                subject = f"🚀 股票筛选报告 | {timestamp} | 最佳: {best_stock}"
            
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            
            # 创建HTML内容
            html_content = self.create_screening_report_html(results, summary)
            
            # 添加HTML部分
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 如果需要，附加JSON文件
            if attach_json and results:
                import json
                json_data = {
                    'summary': summary,
                    'results': results,
                    'generated_at': datetime.now().isoformat()
                }
                
                json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
                attachment = MIMEBase('application', 'json')
                attachment.set_payload(json_content.encode('utf-8'))
                encoders.encode_base64(attachment)
                
                filename = f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(attachment)
            
            # 发送邮件
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            
            logger.info(f"✅ 筛选结果邮件发送成功: {self.receiver_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")
            return False
    
    def send_markdown_report(self, md_file_path: str, subject: str = None) -> bool:
        """
        发送Markdown报告邮件
        
        Args:
            md_file_path: Markdown文件路径
            subject: 邮件主题
            
        Returns:
            bool: 发送是否成功
        """
        try:
            if not os.path.exists(md_file_path):
                logger.error(f"Markdown文件不存在: {md_file_path}")
                return False
            
            # 读取Markdown内容
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # 转换为HTML
            html_content = self.convert_md_to_html(md_content)
            
            # 创建邮件
            msg = MIMEMultipart('alternative')
            
            # 设置邮件主题
            if not subject:
                filename = os.path.basename(md_file_path)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                subject = f"📊 {filename} | {timestamp}"
            
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            
            # 添加HTML内容
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 附加原始Markdown文件
            with open(md_file_path, 'rb') as f:
                attachment = MIMEBase('text', 'markdown')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                
                filename = os.path.basename(md_file_path)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(attachment)
            
            # 发送邮件
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            
            logger.info(f"✅ Markdown报告邮件发送成功: {self.receiver_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Markdown邮件发送失败: {e}")
            return False

    def test_email_config(self) -> bool:
        """测试邮件配置"""
        try:
            if not all([self.sender_email, self.sender_password, self.receiver_email]):
                print("❌ 邮件配置不完整")
                print("请设置以下环境变量:")
                print("  EMAIL_SENDER=your_email@gmail.com")
                print("  EMAIL_PASSWORD=your_app_password")
                print("  EMAIL_RECEIVER=receiver@email.com")
                return False
            
            # 测试连接
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
            
            print("✅ 邮件配置测试成功")
            print(f"发送方: {self.sender_email}")
            print(f"接收方: {self.receiver_email}")
            return True
            
        except Exception as e:
            print(f"❌ 邮件配置测试失败: {e}")
            return False 