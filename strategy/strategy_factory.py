import importlib
import logging
from typing import Dict, List, Optional, Any, Type
import os
import inspect
import sys
from pathlib import Path

from .strategy_base import Strategy
from .tdi_strategy import TDIStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .custom_strategy import CustomStrategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .combined_strategy import CombinedStrategy
from .uss_market_forecast import USSMarketForecast
from .uss_gold_triangle_risk import USSGoldTriangleRisk
from .cpgw_strategy import CPGWStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyFactory")


class StrategyFactory:
    """策略工厂类，用于创建和管理交易策略"""

    def __init__(self):
        """初始化策略工厂"""
        self.strategies: Dict[str, Type[Strategy]] = {}
        self._register_builtin_strategies()

    def _register_builtin_strategies(self):
        """注册内置策略"""
        self.register_strategy("TDI", TDIStrategy)
        self.register_strategy("BollingerBands", BollingerBandsStrategy)
        self.register_strategy("Custom", CustomStrategy)
        self.register_strategy("NiuniuV3", NiuniuStrategyV3)
        self.register_strategy("Combined", CombinedStrategy)
        self.register_strategy("MarketForecast", USSMarketForecast)
        self.register_strategy("GoldTriangle", USSGoldTriangleRisk)
        self.register_strategy("CPGW", CPGWStrategy)
        logger.info(f"注册了 {len(self.strategies)} 个内置策略")

    def register_strategy(self, name: str, strategy_class: Type[Strategy]) -> None:
        """
        注册新的策略
        
        参数:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, Strategy):
            logger.error(f"策略类 {strategy_class.__name__} 必须继承自 Strategy 基类")
            return

        self.strategies[name] = strategy_class
        logger.info(f"成功注册策略: {name}")

    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> Strategy:
        """
        创建策略实例
        :param strategy_name: 策略名称
        :param kwargs: 策略参数
        :return: 策略实例
        """
        strategy_map = {
            'tdi': TDIStrategy,
            'bollinger_bands': BollingerBandsStrategy,
            'custom': CustomStrategy,
            'niuniu_v3': NiuniuStrategyV3,
            'combined': CombinedStrategy,
            'market_forecast': USSMarketForecast,
            'gold_triangle': USSGoldTriangleRisk,
            'cpgw': CPGWStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        return strategy_map[strategy_name](**kwargs)

    def create_all_strategies(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Strategy]:
        """
        创建所有已注册策略的实例
        
        参数:
            parameters: 策略参数字典，格式为 {策略名称: 策略参数}
            
        返回:
            策略实例字典
        """
        if parameters is None:
            parameters = {}

        strategies = {}
        for name in self.strategies:
            strategy = self.create_strategy(name, parameters.get(name))
            if strategy:
                strategies[name] = strategy

        return strategies

    def get_strategy_names(self) -> List[str]:
        """获取所有已注册的策略名称"""
        return list(self.strategies.keys())

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略信息
        
        参数:
            name: 策略名称
            
        返回:
            策略信息字典
        """
        strategy_class = self.strategies.get(name)
        if not strategy_class:
            return None

        return {
            'name': name,
            'description': strategy_class.__doc__,
            'parameters': strategy_class(None).get_strategy_info()
        }

    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略信息
        
        返回:
            策略信息字典，键为策略名称，值为策略信息
        """
        result = {}
        for name in self.strategies:
            result[name] = self.get_strategy_info(name)
        return result
