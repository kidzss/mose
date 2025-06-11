"""
股票交易策略和分析模块 - 简化版本
"""

# 版本信息
__version__ = '0.2.0'

# 基本导入，避免复杂依赖
try:
    from .strategy_base import Strategy
    from .strategy_factory import StrategyFactory
    from .combined_strategy import CombinedStrategy
    
    __all__ = [
        'Strategy',
        'StrategyFactory', 
        'CombinedStrategy'
    ]
except ImportError as e:
    print(f"Warning: Some strategy modules failed to import: {e}")
    __all__ = [] 