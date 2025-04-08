"""
MOSE Data Module
===============

统一的数据访问模块，提供标准化的接口来访问不同来源的金融市场数据。

主要组件:
- DataInterface: 统一的数据访问接口
- DataValidator: 数据验证和处理工具
- DataSource: 数据源抽象基类
- MarketDataUpdater: 市场数据更新器

主要功能:
1. 数据获取：历史数据、实时数据、批量数据
2. 数据验证：完整性检查、异常值检测、连续性验证
3. 数据处理：缺失值处理、异常值处理、技术指标计算
4. 数据更新：自动更新、状态监控、质量检查
"""

from .data_interface import DataInterface, DataSource, MySQLDataSource, YahooFinanceDataSource
from .data_validator import DataValidator
from .data_updater import MarketDataUpdater

__all__ = [
    'DataInterface',
    'DataSource',
    'DataValidator',
    'MarketDataUpdater',
    'MySQLDataSource',
    'YahooFinanceDataSource',
]

# 版本信息
__version__ = '1.0.0' 