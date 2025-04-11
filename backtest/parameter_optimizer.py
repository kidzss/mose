import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from .strategy_evaluator import StrategyEvaluator

@dataclass
class OptimizationResult:
    parameters: Dict
    metrics: Dict
    score: float

class ParameterOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        param_ranges: Dict[str, List],
        strategy_func: Callable,
        metric_weights: Dict[str, float] = None
    ):
        """
        参数:
            data: 回测数据
            param_ranges: 参数范围字典，如 {'sma_period': [5,10,20], 'rsi_period': [14,21]}
            strategy_func: 策略函数，接受数据和参数，返回仓位序列
            metric_weights: 评估指标权重，如 {'sharpe_ratio': 0.4, 'max_drawdown': -0.3}
        """
        self.data = data
        self.param_ranges = param_ranges
        self.strategy_func = strategy_func
        self.metric_weights = metric_weights or {
            'sharpe_ratio': 0.4,
            'max_drawdown': -0.3,
            'win_rate': 0.2,
            'profit_factor': 0.1
        }
        self.evaluator = StrategyEvaluator(data)
        
    def _calculate_score(self, metrics: Dict) -> float:
        """计算策略得分"""
        score = 0
        for metric, weight in self.metric_weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
        return score
        
    def _evaluate_parameters(self, params: Dict) -> OptimizationResult:
        """评估单组参数"""
        try:
            # 运行策略获取仓位
            positions = self.strategy_func(self.data, params)
            
            # 评估策略
            strategy_metrics = self.evaluator.analyze_trades(positions)
            
            # 计算综合得分
            score = self._calculate_score(vars(strategy_metrics))
            
            return OptimizationResult(
                parameters=params,
                metrics=vars(strategy_metrics),
                score=score
            )
        except Exception as e:
            print(f"参数评估出错: {e}")
            return None
            
    def optimize(self, max_workers: int = None) -> List[OptimizationResult]:
        """
        优化参数
        
        参数:
            max_workers: 最大并行进程数，默认为CPU核心数
        """
        # 生成参数组合
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        param_combinations = list(product(*param_values))
        
        results = []
        
        # 并行评估参数
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for combo in param_combinations:
                params = dict(zip(param_names, combo))
                futures.append(
                    executor.submit(self._evaluate_parameters, params)
                )
                
            # 收集结果
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
                    
        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results
        
    def generate_optimization_report(self, results: List[OptimizationResult], top_n: int = 5) -> str:
        """生成优化报告"""
        if not results:
            return "没有有效的优化结果"
            
        report = f"""
参数优化报告
===========
评估指标权重:
"""
        for metric, weight in self.metric_weights.items():
            report += f"- {metric}: {weight}\n"
            
        report += f"\n前 {top_n} 组最优参数:\n"
        
        for i, result in enumerate(results[:top_n], 1):
            report += f"""
第 {i} 组:
- 参数: {result.parameters}
- 得分: {result.score:.4f}
- 性能指标:
  - 夏普比率: {result.metrics['sharpe_ratio']:.2f}
  - 最大回撤: {result.metrics['max_drawdown']:.2f}%
  - 胜率: {result.metrics['win_rate']:.2%}
  - 盈亏比: {result.metrics['profit_factor']:.2f}
  - 总收益率: {result.metrics['total_return']:.2f}%
  - 年化收益率: {result.metrics['annual_return']:.2f}%
"""
            
        return report 