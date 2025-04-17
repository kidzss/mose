"""
策略工厂模块，用于创建和管理交易策略
"""

from typing import Dict, Type
from strategy.uss_gold_triangle_strategy import GoldTriangleStrategy
from strategy.uss_momentum_strategy import MomentumStrategy
from strategy.uss_niuniu_strategy import NiuniuStrategy
from strategy.uss_tdi_strategy import TDIStrategy
from strategy.uss_market_forecast_strategy import MarketForecastStrategy
from strategy.uss_cpgw_strategy import CPGWStrategy
from strategy.uss_volume_strategy import VolumeStrategy

class StrategyFactory:
    """策略工厂类"""
    
    def __init__(self):
        """初始化策略工厂"""
        self.strategy_classes = {
            'GoldTriangle': GoldTriangleStrategy,
            'Momentum': MomentumStrategy,
            'Niuniu': NiuniuStrategy,
            'TDI': TDIStrategy,
            'MarketForecast': MarketForecastStrategy,
            'CPGW': CPGWStrategy,
            'Volume': VolumeStrategy
        }
        
    def create_strategy(self, strategy_name: str, **kwargs):
        """
        创建指定的策略实例
        
        参数:
            strategy_name: 策略名称
            **kwargs: 传递给策略构造函数的参数
        """
        if strategy_name not in self.strategy_classes:
            raise ValueError(f"未知的策略名称: {strategy_name}")
            
        strategy_class = self.strategy_classes[strategy_name]
        # 创建一个新的kwargs字典，不包含strategy_name
        strategy_kwargs = {k: v for k, v in kwargs.items() if k != 'strategy_name'}
        return strategy_class(**strategy_kwargs)
        
    def create_all_strategies(self, **kwargs) -> Dict:
        """
        创建所有已注册的策略实例
        
        参数:
            **kwargs: 传递给所有策略构造函数的参数
            
        返回:
            策略名称到策略实例的映射字典
        """
        return {
            name: self.create_strategy(name, **kwargs)
            for name in self.strategy_classes
        }
        
    def register_strategy(self, name: str, strategy_class: Type):
        """
        注册新的策略类
        
        参数:
            name: 策略名称
            strategy_class: 策略类
        """
        self.strategy_classes[name] = strategy_class 