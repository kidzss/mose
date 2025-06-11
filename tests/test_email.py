import logging
from monitor.trading_monitor import AlertSystem
from config.trading_config import default_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_email_sending():
    """测试邮件发送功能"""
    try:
        # 创建 AlertSystem 实例
        alert_system = AlertSystem(default_config)
        
        # 发送测试邮件
        alert_system.send_alert(
            stock="AAPL",
            alert_type="test_alert",
            message="这是一封测试邮件",
            price=150.0,
            indicators={
                "MA5": 149.5,
                "MA10": 148.8,
                "RSI": 65.2
            }
        )
        
        logger.info("测试邮件发送成功")
        
    except Exception as e:
        logger.error(f"测试邮件发送失败: {str(e)}")

if __name__ == "__main__":
    test_email_sending() 