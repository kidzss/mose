import importlib
import logging
from typing import Dict, List, Optional, Any, Type
import os
import inspect
import sys
from pathlib import Path

from .strategy_base import Strategy
from .uss_gold_triangle_strategy import GoldTriangleStrategy
from .uss_momentum_strategy import MomentumStrategy
from .uss_tdi_strategy import TDIStrategy
from .uss_market_forecast_strategy import MarketForecastStrategy
from .uss_cpgw_strategy import CPGWStrategy
from .uss_volume_strategy import VolumeStrategy
from .uss_niuniu_strategy import NiuniuStrategy
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
        self.register_strategy("GoldTriangle", GoldTriangleStrategy)
        self.register_strategy("Momentum", MomentumStrategy)
        self.register_strategy("TDI", TDIStrategy)
        self.register_strategy("MarketForecast", MarketForecastStrategy)
        self.register_strategy("CPGW", CPGWStrategy)
        self.register_strategy("Volume", VolumeStrategy)
        self.register_strategy("Niuniu", NiuniuStrategy)
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

    def create_strategy(self, name: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Strategy]:
        """
        创建策略实例
        
        参数:
            name: 策略名称
            parameters: 策略参数
            
        返回:
            策略实例，如果策略不存在则返回None
        """
        strategy_class = self.strategies.get(name)
        if strategy_class is None:
            logger.warning(f"策略 {name} 不存在")
            return None

        try:
            strategy = strategy_class(parameters)
            logger.info(f"成功创建策略 {name} 实例")
            return strategy
        except Exception as e:
            logger.error(f"创建策略 {name} 实例时出错: {e}")
            return None

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
