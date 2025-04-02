import os
import json
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from strategy_optimizer.alerts.strategy_alert import StrategyOptimizedAlert
from strategy_optimizer.utils.config_loader import load_config
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.trading_config import default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_alerts.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_monitored_stocks():
    """返回指定的股票列表"""
    stock_list = [
        "GOOG", "NVDA", "AMD", "TSLA", "AAPL",
        "ASML", "MSFT", "AMZN", "META", "GOOGL"
    ]
    
    # 转换为所需的格式
    monitored_stocks = [
        {"code": code, "name": code}  # 使用代码作为名称
        for code in stock_list
    ]
    
    logger.info(f"使用指定的股票列表: {', '.join(stock_list)}")
    return monitored_stocks

def main():
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 配置文件路径（仅用于模型配置）
        config_path = os.path.join(current_dir, 'configs', 'optimizer_config.json')
        model_path = os.path.join(current_dir, 'outputs', 'model.pth')
        
        # 使用trading_config中的邮件配置
        email_config = {
            'smtp_server': default_config.email.smtp_server,
            'smtp_port': default_config.email.smtp_port,
            'sender_email': default_config.email.sender_email,
            'sender_password': default_config.email.sender_password,
            'receiver_email': default_config.email.receiver_emails[0]  # 使用第一个接收邮箱
        }
        
        # 从数据库获取监控的股票列表
        monitored_stocks = get_monitored_stocks()
        
        if not monitored_stocks:
            logger.warning("未从数据库获取到需要监控的股票列表")
            return
        
        # 初始化预警系统
        alert_system = StrategyOptimizedAlert(
            model_path=model_path,
            config_path=config_path,  # 仅用于加载模型配置
            email_config=email_config
        )
        
        logger.info(f"开始监控股票: {', '.join([stock['name'] + '(' + stock['code'] + ')' for stock in monitored_stocks])}")
        
        # 为每个股票生成预警
        for stock in monitored_stocks:
            try:
                stock_code = stock['code']
                stock_name = stock['name']
                logger.info(f"正在处理股票 {stock_name}({stock_code})")
                alert_system.generate_alert(stock_code)
            except Exception as e:
                logger.error(f"处理股票 {stock_name}({stock_code}) 时出错: {str(e)}")
                continue
        
        logger.info("预警检查完成")
        
    except Exception as e:
        logger.error(f"运行预警系统时出错: {str(e)}")

if __name__ == "__main__":
    main() 