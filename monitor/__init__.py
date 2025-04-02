"""
Monitor module for real-time market data monitoring
"""

from .real_time_monitor import RealTimeMonitor
from .data_fetcher import DataFetcher
from .stock_monitor_manager import StockMonitorManager
from .market_monitor import MarketMonitor

__all__ = ['RealTimeMonitor', 'DataFetcher', 'StockMonitorManager', 'MarketMonitor']

__version__ = '0.1.0'


def realtime_monitor():
    return None