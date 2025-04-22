import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import threading
import asyncio

from data.data_interface import RealTimeDataSource, YahooFinanceRealTimeSource
from strategy.strategy_base import Strategy
from monitor.notification_manager import NotificationManager
from monitor.strategy_monitor import StrategyMonitor
from config.config_manager import ConfigManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_time_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTimeMonitor")

class RealTimeMonitor:
    """实时监控类"""
    def __init__(self, config_path: str = 'config/strategy_config.json'):
        self.config = ConfigManager(config_path)
        self.data_source = YahooFinanceRealTimeSource()
        self.strategy_monitor = StrategyMonitor(config_path)
        self.monitored_stocks = self.config.get('monitored_stocks', [])
        self.monitor_interval = self.config.get('monitor_interval', 60)
        self.loop = asyncio.get_event_loop()
        self.logger = logging.getLogger(__name__)
        
    async def _update_real_time_data(self) -> Dict[str, pd.DataFrame]:
        """获取并更新实时数据"""
        try:
            data = await self.data_source.get_realtime_data(self.monitored_stocks)
            return data
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {str(e)}")
            return {}
            
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 获取实时数据
                data = await self._update_real_time_data()
                
                if not data:
                    self.logger.warning("No data received, skipping this iteration")
                    await asyncio.sleep(self.monitor_interval)
                    continue
                
                # 更新策略监控
                for symbol, df in data.items():
                    if not df.empty:
                        self.strategy_monitor.update_data(symbol, df)
                        signals = self.strategy_monitor.get_signals(symbol)
                        
                        # 处理信号
                        for signal in signals:
                            self.logger.info(f"Signal for {symbol}: {signal}")
                            # 这里可以添加信号处理逻辑
                
                # 等待下一次更新
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitor_interval)
                
    def start(self):
        """启动监控"""
        try:
            self.logger.info("Starting real-time monitor...")
            self.loop.run_until_complete(self._monitoring_loop())
        except KeyboardInterrupt:
            self.logger.info("Stopping real-time monitor...")
        finally:
            self.loop.close()
            
    def stop(self):
        """停止监控"""
        self.logger.info("Stopping real-time monitor...")
        self.loop.stop()

    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        return {
            "is_running": self.loop.is_running(),
            "last_check_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }