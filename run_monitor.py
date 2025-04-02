#!/usr/bin/env python
"""
主运行脚本，用于启动实时监控系统
"""

import sys
import logging
from monitor.real_time_monitor import RealTimeMonitor
from monitor.data_fetcher import DataFetcher
from monitor.stock_monitor_manager import StockMonitorManager
from monitor.market_monitor import MarketMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("realtime_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'mose',
    'port': 3306
}


def main():
    try:
        # 初始化组件
        data_fetcher = DataFetcher(db_config=DB_CONFIG)
        stock_manager = StockMonitorManager(db_config=DB_CONFIG)
        market_monitor = MarketMonitor()

        # 创建监控器实例
        monitor = RealTimeMonitor(
            data_fetcher=data_fetcher,
            stock_manager=stock_manager,
            market_monitor=market_monitor
        )

        # 启动监控
        monitor.start_monitoring()

        # 保持程序运行
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到停止信号，正在停止监控...")
            monitor.stop_monitoring()

    except Exception as e:
        logger.error(f"监控系统运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
