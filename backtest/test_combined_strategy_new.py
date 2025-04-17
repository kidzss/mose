import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from strategy.combined_strategy import CombinedStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.data_loader import DataLoader
from backtest.strategy_evaluator import StrategyEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backtest(symbol: str = 'NVDA', 
                start_date: str = '2023-01-01',
                end_date: str = '2024-04-01',
                initial_capital: float = 1_000_000):
    """
    运行回测
    
    Args:
        symbol: 交易标的
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
    """
    try:
        # 加载数据
        data_loader = DataLoader()
        df = data_loader.load_data(symbol, start_date, end_date)
        
        if df.empty:
            logger.error(f"无法加载 {symbol} 的数据")
            return None
        
        # 创建策略实例
        strategy = CombinedStrategy()
        
        # 创建回测引擎
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.001
        )
        
        # 运行回测
        results = engine.run_backtest(df, strategy)
        
        # 评估策略
        evaluator = StrategyEvaluator()
        metrics = evaluator.evaluate(results)
        
        # 打印结果
        logger.info(f"\n回测结果 ({symbol}):")
        logger.info(f"总收益率: {metrics['total_return']:.2%}")
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"胜率: {metrics['win_rate']:.2%}")
        logger.info(f"盈亏比: {metrics['profit_factor']:.2f}")
        
        # 绘制结果
        plot_results(results, symbol)
        
        return results
        
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        return None

def plot_results(results: pd.DataFrame, symbol: str):
    """绘制回测结果"""
    plt.figure(figsize=(15, 10))
    
    # 绘制净值曲线
    plt.subplot(2, 1, 1)
    plt.plot(results.index, results['equity'], label='Strategy')
    plt.plot(results.index, results['benchmark'], label='Benchmark')
    plt.title(f'{symbol} Strategy Performance')
    plt.legend()
    plt.grid(True)
    
    # 绘制回撤
    plt.subplot(2, 1, 2)
    plt.fill_between(results.index, results['drawdown'], 0, color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'backtest/results/{symbol}_backtest_results.png')
    plt.close()

if __name__ == '__main__':
    # 运行回测
    results = run_backtest()
    
    if results is not None:
        logger.info("回测完成")
    else:
        logger.error("回测失败") 