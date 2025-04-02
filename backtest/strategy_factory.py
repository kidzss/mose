from typing import Dict, Type, Optional
import logging
from pathlib import Path
import importlib
import inspect

from .strategy_base import Strategy

logger = logging.getLogger(__name__)

class StrategyFactory:
    """策略工厂类，用于管理和创建交易策略"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[Strategy]] = {}
        self._load_strategies()
        
    def _load_strategies(self):
        """加载策略目录下的所有策略类"""
        try:
            # 获取策略目录
            strategy_dir = Path(__file__).parent.parent / 'strategy'
            
            # 遍历策略文件
            for strategy_file in strategy_dir.glob('*.py'):
                if strategy_file.name.startswith('__'):
                    continue
                    
                try:
                    # 导入模块
                    module_name = f"strategy.{strategy_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # 查找策略类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Strategy) and 
                            obj != Strategy):
                            self.strategies[name] = obj
                            logger.info(f"加载策略: {name}")
                            
                except Exception as e:
                    logger.error(f"加载策略文件 {strategy_file} 时出错: {e}")
                    
        except Exception as e:
            logger.error(f"加载策略时出错: {e}")
            
    def get_strategy(self, name: str) -> Optional[Type[Strategy]]:
        """获取指定名称的策略类"""
        return self.strategies.get(name)
        
    def create_strategy(self, name: str, **kwargs) -> Optional[Strategy]:
        """创建指定名称的策略实例"""
        strategy_class = self.get_strategy(name)
        if strategy_class:
            try:
                return strategy_class(**kwargs)
            except Exception as e:
                logger.error(f"创建策略 {name} 时出错: {e}")
        return None
        
    def create_all_strategies(self, **kwargs) -> Dict[str, Strategy]:
        """创建所有已注册的策略实例"""
        instances = {}
        for name, strategy_class in self.strategies.items():
            try:
                instances[name] = strategy_class(**kwargs)
                logger.info(f"创建策略实例: {name}")
            except Exception as e:
                logger.error(f"创建策略 {name} 时出错: {e}")
        return instances
        
    def list_strategies(self) -> Dict[str, Type[Strategy]]:
        """列出所有已注册的策略"""
        return self.strategies.copy() 