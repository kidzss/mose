"""
投资组合监控模块
"""

from .portfolio_monitor import PortfolioMonitor
from .report_generator import ReportGenerator
from .notification_manager import NotificationManager
from .market_monitor import MarketMonitor

__all__ = [
    'PortfolioMonitor',
    'ReportGenerator',
    'NotificationManager',
    'MarketMonitor'
]

__version__ = '0.1.0'


def realtime_monitor():
    return None