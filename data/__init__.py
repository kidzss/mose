"""
Data package for the trading system.
"""

from .data_interface import DataInterface, DataSource, MySQLDataSource, YahooFinanceDataSource
from .data_validator import DataValidator
# from .data_updater import MarketDataUpdater

__all__ = [
    'DataInterface',
    'DataSource',
    'DataValidator',
    'MySQLDataSource',
    'YahooFinanceDataSource',
]

# 版本信息
__version__ = '1.0.0' 