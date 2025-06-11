"""
投资组合监控模块
"""

from .portfolio_monitor import PortfolioMonitor
from .report_generator import ReportGenerator
from .notification_manager import NotificationManager
from .market_monitor import MarketMonitor
from .smart_daily_report import SmartDailyReportGenerator
from .smart_daily_email_sender import SmartDailyEmailSender

__all__ = [
    'PortfolioMonitor',
    'ReportGenerator',
    'NotificationManager',
    'MarketMonitor',
    'SmartDailyReportGenerator',
    'SmartDailyEmailSender'
]

__version__ = '0.1.0'


def realtime_monitor():
    return None