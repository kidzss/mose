import importlib
import logging
from typing import Dict, List, Optional, Any, Type
import os
import inspect
import sys
from pathlib import Path

from .strategy_base import Strategy
from .tdi_strategy import TDIStrategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .combined_strategy import CombinedStrategy
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
        """注册内置核心策略"""
        self.register_strategy("TDI", TDIStrategy)
        self.register_strategy("NiuniuV3", NiuniuStrategyV3)
        self.register_strategy("CPGW", CPGWStrategy)
        self.register_strategy("Combined", CombinedStrategy)
        logger.info(f"✅ 注册了 {len(self.strategies)} 个核心策略")

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
            'niuniu_v3': NiuniuStrategyV3,
            'cpgw': CPGWStrategy,
            'combined': CombinedStrategy,
            # 兼容性别名
            'TDI': TDIStrategy,
            'NiuniuV3': NiuniuStrategyV3,
            'CPGW': CPGWStrategy,
            'Combined': CombinedStrategy
        }
        
        if strategy_name not in strategy_map:
            available_strategies = ', '.join(strategy_map.keys())
            raise ValueError(f"未知策略: {strategy_name}. 可用策略: {available_strategies}")
            
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
            try:
                strategy_class = self.strategies[name]
                strategy_params = parameters.get(name, {})
                strategy = strategy_class(**strategy_params)
                strategies[name] = strategy
                logger.info(f"✅ 创建策略实例: {name}")
            except Exception as e:
                logger.error(f"❌ 创建策略 {name} 实例失败: {e}")

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

        try:
            # 创建一个临时实例以获取信息
            temp_strategy = strategy_class()
            return {
                'name': name,
                'description': strategy_class.__doc__ or f"{name} 策略",
                'class_name': strategy_class.__name__,
                'module': strategy_class.__module__
            }
        except Exception as e:
            logger.error(f"获取策略 {name} 信息时出错: {e}")
            return None

    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略信息
        
        返回:
            策略信息字典，键为策略名称，值为策略信息
        """
        result = {}
        for name in self.strategies:
            info = self.get_strategy_info(name)
            if info:
                result[name] = info
        return result

    def get_core_strategies(self) -> List[str]:
        """获取核心策略列表"""
        return ['TDI', 'NiuniuV3', 'CPGW']

    def create_combined_strategy(self, weights: Optional[Dict[str, float]] = None) -> CombinedStrategy:
        """
        创建配置好的组合策略
        
        参数:
            weights: 策略权重，默认为 {'niuniu': 0.5, 'tdi': 0.3, 'cpgw': 0.2}
        
        返回:
            配置好的组合策略实例
        """
        default_weights = {
            'weight_niuniu': 0.50,
            'weight_tdi': 0.30, 
            'weight_cpgw': 0.20
        }
        
        if weights:
            # 转换权重格式
            for key, value in weights.items():
                if key == 'niuniu':
                    default_weights['weight_niuniu'] = value
                elif key == 'tdi':
                    default_weights['weight_tdi'] = value
                elif key == 'cpgw':
                    default_weights['weight_cpgw'] = value
        
        return CombinedStrategy(parameters=default_weights)
