import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import default_config
from monitor.trading_monitor import AlertSystem

def test_email():
    print("开始测试邮件发送...")
    
    # 创建警报系统
    alert_system = AlertSystem(default_config)
    
    # 测试邮件内容
    subject = "交易系统邮件测试"
    body = """
    <h2>这是一封测试邮件</h2>
    <p>如果您收到这封邮件，说明邮件系统配置正确。</p>
    <p>邮件配置信息:</p>
    <ul>
        <li>发件人: {}</li>
        <li>收件人: {}</li>
        <li>SMTP服务器: {}:{}</li>
    </ul>
    """.format(
        default_config.email.sender_email,
        default_config.email.receiver_emails,
        default_config.email.smtp_server,
        default_config.email.smtp_port
    )
    
    try:
        # 发送测试邮件
        alert_system.send_email(subject, body)
        print("测试邮件已发送，请检查邮箱")
    except Exception as e:
        print(f"发送邮件时出错: {str(e)}")

if __name__ == "__main__":
    test_email() 