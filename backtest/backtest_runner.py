import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .strategy_evaluator import StrategyEvaluator, StrategyMetrics
from .parameter_optimizer import ParameterOptimizer, OptimizationResult
from .market_analyzer import MarketAnalyzer, MarketState
from .risk_manager import RiskManager, RiskMetrics


@dataclass
class BacktestResult:
    strategy_metrics: StrategyMetrics
    risk_metrics: RiskMetrics
    market_state: MarketState
    optimization_result: Optional[OptimizationResult] = None


class BacktestRunner:
    def __init__(
            self,
            data: pd.DataFrame,
            strategy_func: Callable,
            benchmark_data: Optional[pd.DataFrame] = None,
            param_ranges: Optional[Dict] = None
    ):
        """
        初始化回测运行器
        
        参数:
            data: 回测数据
            strategy_func: 策略函数
            benchmark_data: 基准数据
            param_ranges: 参数优化范围
        """
        self.data = data
        self.strategy_func = strategy_func
        self.benchmark_data = benchmark_data
        self.param_ranges = param_ranges

        # 初始化各个组件
        self.evaluator = StrategyEvaluator(data)
        self.risk_manager = RiskManager(data, benchmark_data)
        self.market_analyzer = MarketAnalyzer(data)

        if param_ranges:
            self.optimizer = ParameterOptimizer(
                data=data,
                param_ranges=param_ranges,
                strategy_func=strategy_func
            )
        else:
            self.optimizer = None

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, parameters: Optional[Dict] = None) -> BacktestResult:
        """
        运行回测
        
        参数:
            parameters: 策略参数
        """
        try:
            # 运行策略获取仓位
            positions = self.strategy_func(self.data, parameters or {})

            # 评估策略
            strategy_metrics = self.evaluator.analyze_trades(positions)

            # 计算收益率
            returns = self.evaluator.calculate_returns(positions)

            # 评估风险
            risk_metrics = self.risk_manager.evaluate_risk(returns, positions)

            # 分析市场状态
            market_state = self.market_analyzer.analyze_market_state()

            # 如果需要优化参数
            optimization_result = None
            if self.optimizer and self.param_ranges:
                optimization_result = self.optimizer._evaluate_parameters(parameters or {})

            return BacktestResult(
                strategy_metrics=strategy_metrics,
                risk_metrics=risk_metrics,
                market_state=market_state,
                optimization_result=optimization_result
            )

        except Exception as e:
            self.logger.error(f"回测过程中出错: {str(e)}")
            raise

    def run_optimization(self) -> OptimizationResult:
        """运行参数优化"""
        try:
            if not self.optimizer:
                raise ValueError("未配置参数优化器")

            # 设置优化指标权重
            metric_weights = {
                'sharpe_ratio': 0.4,
                'max_drawdown': -0.3,
                'win_rate': 0.2,
                'profit_factor': 0.1
            }

            # 运行优化
            return self.optimizer._evaluate_parameters(self.param_ranges)

        except Exception as e:
            self.logger.error(f"参数优化过程中出错: {str(e)}")
            raise

    def generate_report(self, result: BacktestResult) -> str:
        """生成回测报告"""
        report = """
回测分析报告
===========

一、策略绩效分析
-------------
"""
        # 添加策略评估指标
        report += f"""
交易统计:
- 总交易次数: {result.strategy_metrics.total_trades}
- 胜率: {result.strategy_metrics.win_rate:.2%}
- 盈亏比: {result.strategy_metrics.profit_factor:.2f}
- 总收益率: {result.strategy_metrics.total_return:.2f}%
- 年化收益率: {result.strategy_metrics.annual_return:.2f}%
- 平均每笔收益: {result.strategy_metrics.avg_profit_per_trade:.2f}%
- 平均持仓天数: {result.strategy_metrics.avg_holding_days:.1f}天
"""

        report += """
二、风险分析
---------
"""
        # 添加风险分析指标
        report += f"""
风险指标:
- 最大回撤: {result.risk_metrics.max_drawdown:.2%}
- 波动率(年化): {result.risk_metrics.volatility:.2%}
- Beta系数: {result.risk_metrics.beta:.2f}
- 相关系数: {result.risk_metrics.correlation:.2f}
- 平均持仓: {result.risk_metrics.position_exposure:.2%}
"""

        report += """
三、市场分析
---------
"""
        # 添加市场分析指标
        report += f"""
市场状态:
- 主要趋势: {result.market_state.trend}
- 市场阶段: {result.market_state.market_regime}
- 波动率: {result.market_state.volatility:.2%}
- 动量: {result.market_state.momentum:.2%}
- 成交量趋势: {result.market_state.volume_trend}
- 支撑位: {result.market_state.support_level:.2f}
- 压力位: {result.market_state.resistance_level:.2f}
"""

        # 如果有优化结果，添加优化报告
        if result.optimization_result:
            report += """
四、参数优化结果
------------
"""
            report += f"""
最优参数组合:
{result.optimization_result.parameters}

优化指标:
- 综合得分: {result.optimization_result.score:.2f}
- Sharpe比率: {result.optimization_result.metrics['sharpe_ratio']:.2f}
- 最大回撤: {result.optimization_result.metrics['max_drawdown']:.2%}
- 胜率: {result.optimization_result.metrics['win_rate']:.2%}
- 盈亏比: {result.optimization_result.metrics['profit_factor']:.2f}
"""

        return report

    def plot_results(self, positions: pd.Series):
        """绘制回测结果图表"""
        try:
            import matplotlib.pyplot as plt

            # 计算策略收益
            returns = self.evaluator.calculate_returns(positions)
            cumulative_returns = (1 + returns).cumprod()

            # 计算基准收益
            if self.benchmark_data is not None:
                benchmark_returns = self.benchmark_data['Close'].pct_change()
                cumulative_benchmark = (1 + benchmark_returns).cumprod()

            # 创建图表
            plt.figure(figsize=(12, 8))

            # 绘制策略收益曲线
            plt.plot(cumulative_returns.index, cumulative_returns.values,
                     label='Strategy', color='blue')

            # 绘制基准收益曲线
            if self.benchmark_data is not None:
                plt.plot(cumulative_benchmark.index, cumulative_benchmark.values,
                         label='Benchmark', color='gray', alpha=0.5)

            # 添加图表元素
            plt.title('Backtest Results')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)

            # 显示图表
            plt.show()

        except ImportError:
            self.logger.warning("未安装matplotlib，无法绘制图表")
        except Exception as e:
            self.logger.error(f"绘制图表时出错: {str(e)}")


def run_parallel_backtest(
        data_dict: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        param_ranges: Optional[Dict] = None,
        n_workers: int = 4
) -> Dict[str, BacktestResult]:
    """
    并行运行多个回测
    
    参数:
        data_dict: 股票数据字典，格式为 {symbol: data_frame}
        strategy_func: 策略函数
        param_ranges: 参数优化范围
        n_workers: 并行进程数
    """

    def _run_single_backtest(args):
        symbol, data = args
        runner = BacktestRunner(
            data=data,
            strategy_func=strategy_func,
            param_ranges=param_ranges
        )
        return symbol, runner.run_backtest()

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for symbol, result in executor.map(_run_single_backtest, data_dict.items()):
            results[symbol] = result

    return results
