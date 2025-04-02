import os
import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Type, Optional, Any, Set, Tuple
import pandas as pd
import traceback

from .strategy_base import Strategy


class StrategyManager:
    """
    策略管理器
    
    负责:
    1. 自动发现和注册策略
    2. 实例化和管理策略对象
    3. 运行分析并汇总结果
    """
    
    def __init__(self, strategies_package: str = 'strategy'):
        """
        初始化策略管理器
        
        参数:
            strategies_package: 策略模块所在的包名
        """
        self.strategies_package = strategies_package
        self.strategy_classes: Dict[str, Type[Strategy]] = {}
        self.strategy_instances: Dict[str, Strategy] = {}
        
        # 设置日志
        self.logger = logging.getLogger("strategy_manager")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 自动发现策略类
        self.discover_strategies()
        
    def discover_strategies(self) -> None:
        """自动发现和注册所有策略类"""
        self.logger.info(f"开始发现策略类...")
        
        try:
            # 导入策略包
            strategies_module = importlib.import_module(self.strategies_package)
            strategies_path = os.path.dirname(strategies_module.__file__)
            
            # 遍历包中的所有模块
            for _, module_name, is_pkg in pkgutil.iter_modules([strategies_path]):
                if is_pkg or module_name in ['__init__', 'strategy_base', 'signal_interface', 'strategy_manager']:
                    continue  # 跳过包和特定模块
                
                # 导入模块
                module_path = f"{self.strategies_package}.{module_name}"
                self.logger.debug(f"发现模块: {module_path}")
                
                try:
                    module = importlib.import_module(module_path)
                    
                    # 查找模块中的策略类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Strategy) and 
                            obj is not Strategy and
                            not name.startswith('_')):
                            # 注册策略类
                            self.register_strategy(name, obj)
                            
                except Exception as e:
                    self.logger.error(f"导入模块 {module_path} 时出错: {e}")
                    traceback.print_exc()
            
            self.logger.info(f"发现 {len(self.strategy_classes)} 个策略类")
            
        except Exception as e:
            self.logger.error(f"发现策略类时出错: {e}")
            traceback.print_exc()
    
    def register_strategy(self, name: str, strategy_class: Type[Strategy]) -> None:
        """
        注册策略类
        
        参数:
            name: 策略类名称
            strategy_class: 策略类
        """
        self.logger.debug(f"注册策略类: {name}")
        self.strategy_classes[name] = strategy_class
    
    def create_strategy(self, strategy_name: str, parameters: Optional[Dict[str, Any]] = None) -> Strategy:
        """
        创建策略实例
        
        参数:
            strategy_name: 策略名称
            parameters: 策略参数
            
        返回:
            策略实例
        """
        if strategy_name not in self.strategy_classes:
            raise ValueError(f"未知策略: {strategy_name}")
        
        strategy_class = self.strategy_classes[strategy_name]
        
        try:
            # 创建策略实例
            strategy = strategy_class(parameters)
            # 保存实例
            self.strategy_instances[strategy.name] = strategy
            self.logger.info(f"创建策略实例: {strategy.name}")
            return strategy
        except Exception as e:
            self.logger.error(f"创建策略 {strategy_name} 实例时出错: {e}")
            traceback.print_exc()
            raise
    
    def get_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """
        获取策略实例
        
        参数:
            strategy_name: 策略名称
            
        返回:
            策略实例，如果不存在则返回None
        """
        return self.strategy_instances.get(strategy_name)
    
    def get_all_strategy_names(self) -> List[str]:
        """
        获取所有已注册的策略名称
        
        返回:
            策略名称列表
        """
        return list(self.strategy_classes.keys())
    
    def get_active_strategies(self) -> List[Strategy]:
        """
        获取所有活跃的策略实例
        
        返回:
            策略实例列表
        """
        return list(self.strategy_instances.values())
    
    def run_strategy_analysis(self, strategy_name: str, data: Dict[str, pd.DataFrame], 
                             market_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行单个策略的分析
        
        参数:
            strategy_name: 策略名称
            data: 股票数据字典
            market_state: 市场状态
            
        返回:
            分析结果
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"策略 {strategy_name} 未找到或未实例化")
            return {}
        
        try:
            self.logger.info(f"运行策略分析: {strategy_name}")
            results = strategy.analyze(data, market_state)
            self.logger.info(f"策略 {strategy_name} 分析完成，发现 {len(results)} 个信号")
            return results
        except Exception as e:
            self.logger.error(f"运行策略 {strategy_name} 分析时出错: {e}")
            traceback.print_exc()
            return {}
    
    def run_all_strategies(self, data: Dict[str, pd.DataFrame], 
                          market_state: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        运行所有活跃策略的分析
        
        参数:
            data: 股票数据字典
            market_state: 市场状态
            
        返回:
            分析结果字典，键为策略名称
        """
        all_results = {}
        
        for strategy_name, strategy in self.strategy_instances.items():
            try:
                results = strategy.analyze(data, market_state)
                all_results[strategy_name] = results
            except Exception as e:
                self.logger.error(f"运行策略 {strategy_name} 分析时出错: {e}")
                traceback.print_exc()
                all_results[strategy_name] = {}
        
        return all_results
    
    def optimize_strategy(self, strategy_name: str, data: pd.DataFrame, 
                         param_grid: Dict[str, List[Any]], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            strategy_name: 策略名称
            data: 历史价格数据
            param_grid: 参数网格
            metric: 优化指标
            
        返回:
            优化后的参数
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"策略 {strategy_name} 未找到或未实例化")
            return {}
        
        try:
            self.logger.info(f"开始优化策略 {strategy_name} 的参数")
            best_params = strategy.optimize_parameters(data, param_grid, metric)
            self.logger.info(f"策略 {strategy_name} 参数优化完成，最佳参数: {best_params}")
            return best_params
        except Exception as e:
            self.logger.error(f"优化策略 {strategy_name} 参数时出错: {e}")
            traceback.print_exc()
            return {}
    
    def generate_consolidated_signals(self, data: Dict[str, pd.DataFrame], 
                                     market_state: Optional[Dict[str, Any]] = None,
                                     strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        生成合并后的信号
        
        参数:
            data: 股票数据字典
            market_state: 市场状态
            strategies: 要运行的策略列表，如果为None则运行所有活跃策略
            
        返回:
            合并后的信号结果
        """
        # 如果没有指定策略，运行所有活跃策略
        if strategies is None:
            strategy_results = self.run_all_strategies(data, market_state)
        else:
            strategy_results = {}
            for strategy_name in strategies:
                strategy_results[strategy_name] = self.run_strategy_analysis(
                    strategy_name, data, market_state)
        
        # 合并结果
        consolidated_results = {}
        
        # 收集所有股票代码
        all_symbols = set()
        for results in strategy_results.values():
            all_symbols.update(results.keys())
        
        for symbol in all_symbols:
            # 收集该股票的所有信号
            symbol_signals = []
            
            for strategy_name, results in strategy_results.items():
                if symbol in results:
                    signal_data = results[symbol]
                    signal_data['strategy'] = strategy_name  # 添加策略来源
                    symbol_signals.append(signal_data)
            
            if symbol_signals:
                # 计算平均信号，并保留最强信号
                avg_signal = sum(s['signal'] for s in symbol_signals) / len(symbol_signals)
                
                # 找出信号最强的策略
                strongest_signal = max(symbol_signals, key=lambda s: abs(s['signal']))
                
                # 合并信号
                consolidated_results[symbol] = {
                    'signal': avg_signal,  # 平均信号值
                    'strongest_signal': strongest_signal['signal'],  # 最强信号值
                    'strongest_strategy': strongest_signal['strategy'],  # 最强信号策略
                    'total_strategies': len(symbol_signals),  # 总策略数
                    'market_regime': strongest_signal.get('market_regime', 'unknown'),  # 市场环境
                    'price': strongest_signal.get('price', 0),  # 价格
                    'timestamp': strongest_signal.get('timestamp', None),  # 时间戳
                    'strategies': {s['strategy']: s for s in symbol_signals}  # 各策略详情
                }
        
        return consolidated_results 