import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import logging
import sys
import os

# 添加项目路径到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mose.data_loader import CSVDataLoader
from mose.strategy import CustomStrategy
from tests.strategy_backtest import backtest

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_custom_strategy_parameters():
    """优化自定义策略参数"""
    # 加载数据
    data_loader = CSVDataLoader()
    df = data_loader.load_data("tests/test_data/BTCUSDT-1h-2023.csv")
    
    # 参数网格
    param_grid = {
        'bb_length': [15, 20, 25],
        'bb_std': [1.8, 2.0, 2.2],
        'short_ma': [5, 10, 15],
        'long_ma': [20, 30, 40],
        'rsi_length': [10, 14, 18],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 25, 30],
        'price_pct_trigger': [0.005, 0.01, 0.015]
    }
    
    # 获取所有参数组合
    keys = list(param_grid.keys())
    param_combinations = list(product(*[param_grid[key] for key in keys]))
    logger.info(f"将测试 {len(param_combinations)} 个参数组合")
    
    results = []
    
    # 测试每个参数组合
    for i, params in enumerate(param_combinations):
        if i % 20 == 0:
            logger.info(f"正在测试第 {i+1}/{len(param_combinations)} 个参数组合")
            
        param_dict = {keys[j]: params[j] for j in range(len(keys))}
        
        # 初始化策略
        strategy = CustomStrategy(**param_dict)
        
        # 回测
        df_with_signals = strategy.generate_signals(df.copy())
        backtest_result = backtest(df_with_signals)
        
        # 收集结果
        param_dict.update({
            'total_return': backtest_result['total_return'],
            'max_drawdown': backtest_result['max_drawdown'],
            'sharpe_ratio': backtest_result['sharpe_ratio'],
            'sortino_ratio': backtest_result['sortino_ratio'],
            'calmar_ratio': backtest_result['calmar_ratio'],
            'win_rate': backtest_result['win_rate'],
            'profit_factor': backtest_result['profit_factor'],
            'trade_count': backtest_result['trade_count']
        })
        
        results.append(param_dict)
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('calmar_ratio', ascending=False)
    
    # 保存结果
    os.makedirs('tests/output', exist_ok=True)
    results_df.to_csv('tests/output/custom_strategy_optimization.csv', index=False)
    
    # 输出最佳参数
    best_params = results_df.iloc[0].to_dict()
    logger.info("最佳参数组合:")
    for key in keys:
        logger.info(f"{key}: {best_params[key]}")
    
    logger.info(f"最佳性能指标:")
    logger.info(f"总收益率: {best_params['total_return']:.2%}")
    logger.info(f"最大回撤: {best_params['max_drawdown']:.2%}")
    logger.info(f"夏普比率: {best_params['sharpe_ratio']:.2f}")
    logger.info(f"卡玛比率: {best_params['calmar_ratio']:.2f}")
    logger.info(f"胜率: {best_params['win_rate']:.2%}")
    logger.info(f"盈亏比: {best_params['profit_factor']:.2f}")
    logger.info(f"交易次数: {int(best_params['trade_count'])}")
    
    # 可视化参数影响
    visualize_parameter_impact(results_df, keys[:4])  # 只可视化前4个参数
    
    return best_params

def visualize_parameter_impact(results_df, params):
    """可视化参数对性能的影响"""
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(params):
        plt.subplot(2, 2, i+1)
        grouped = results_df.groupby(param)['calmar_ratio'].mean()
        grouped.plot(kind='bar')
        plt.title(f'Average Calmar Ratio by {param}')
        plt.xlabel(param)
        plt.ylabel('Avg Calmar Ratio')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tests/output/parameter_impact.png')
    logger.info("参数影响分析图表已保存到 tests/output/parameter_impact.png")

if __name__ == "__main__":
    optimize_custom_strategy_parameters() 