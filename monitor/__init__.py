"""
投资组合监控模块
"""

from .portfolio_monitor import PortfolioMonitor
from .report_generator import ReportGenerator

# 可选导入，如果模块不存在则跳过
try:
    from .notification_manager import NotificationManager
except ImportError:
    NotificationManager = None

try:
    from .market_monitor import MarketMonitor
except ImportError:
    MarketMonitor = None

# 已删除的模块，暂时注释掉
# from .smart_daily_report import SmartDailyReportGenerator
# from .smart_daily_email_sender import SmartDailyEmailSender

__all__ = [
    'PortfolioMonitor',
    'ReportGenerator',
]

# 添加可选模块
if NotificationManager:
    __all__.append('NotificationManager')
if MarketMonitor:
    __all__.append('MarketMonitor')

__version__ = '0.1.0'


def realtime_monitor():
    return None