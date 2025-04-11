import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from strategy.custom_tdi_strategy import CustomTDIStrategy
from strategy_optimizer.parameter_optimizer import TDIParameterOptimizer
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(symbol, start_date, end_date):
    """加载历史数据"""
    # 这里应该实现实际的数据加载逻辑
    # 为了示例，我们使用随机生成的数据
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    np.random.seed(42)
    prices = 100 * (1 + np.random.normal(0, 0.02, n).cumsum())
    volumes = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载数据
    logger.info("加载 AAPL 的历史数据...")
    data = load_data('AAPL', '2024-01-01', '2025-03-25')
    
    # 创建策略实例
    strategy = CustomTDIStrategy()
    
    # 创建优化器
    optimizer = TDIParameterOptimizer(strategy, data)
    
    # 开始参数优化
    logger.info("开始参数优化...")
    best_params = optimizer.optimize_parameters()
    
    # 设置最佳参数
    for param, value in best_params.items():
        setattr(strategy, param, value)
    
    # 计算策略表现
    returns = strategy.backtest(data)
    sharpe_ratio = optimizer._calculate_sharpe_ratio(returns)
    sortino_ratio = optimizer._calculate_sortino_ratio(returns)
    total_return = optimizer._calculate_total_return(returns)
    
    # 记录结果
    logger.info("\n参数优化完成！")
    logger.info("最佳参数：")
    for param, value in best_params.items():
        logger.info(f"{param}: {value:.4f}")
    
    logger.info("\n优化后的策略表现：")
    logger.info(f"夏普比率: {sharpe_ratio:.4f}")
    logger.info(f"索提诺比率: {sortino_ratio:.4f}")
    logger.info(f"总收益率: {total_return*100:.2f}%")
    
    # 保存结果
    results = {
        'best_parameters': {k: float(v) for k, v in best_params.items()},
        'performance': {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'total_return': float(total_return)
        }
    }
    
    with open('optimization_results_AAPL.json', 'w') as f:
        json.dump(results, f, indent=4)
    
if __name__ == '__main__':
    main() 