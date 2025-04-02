import os
import time
import logging
import subprocess
from datetime import datetime
import pytz
from typing import Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alert_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_us_market_open() -> Tuple[bool, datetime]:
    """
    检查美股市场是否开市
    
    Returns:
        Tuple[bool, datetime]: (是否开市, 当前美东时间)
    """
    # 获取美东时间
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    
    # 检查是否为工作日
    if now_est.weekday() >= 5:  # 5是周六，6是周日
        return False, now_est
        
    # 检查是否在交易时间内（9:30 - 16:00）
    market_start = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_open = market_start <= now_est <= market_end
    return is_open, now_est

def run_alert_system():
    """运行警报系统"""
    try:
        logger.info("开始运行警报系统")
        
        # 运行警报脚本
        result = subprocess.run(
            ["python", "strategy_optimizer/run_alerts.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("警报系统运行成功")
        else:
            logger.error(f"警报系统运行失败: {result.stderr}")
            
    except Exception as e:
        logger.error(f"运行警报系统时出错: {str(e)}")

def main():
    """主函数"""
    logger.info("启动警报调度系统")
    
    while True:
        try:
            is_open, current_time = is_us_market_open()
            
            if is_open:
                logger.info("市场开市中，以5分钟间隔运行")
                run_alert_system()
                time.sleep(300)  # 5分钟
            else:
                logger.info("市场休市中，以2小时间隔运行")
                run_alert_system()
                time.sleep(7200)  # 2小时
                
        except Exception as e:
            logger.error(f"调度系统运行出错: {str(e)}")
            time.sleep(300)  # 发生错误时等待5分钟后继续

if __name__ == "__main__":
    main() 