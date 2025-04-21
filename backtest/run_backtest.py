import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from strategy.strategy_base import Strategy
from strategy.uss_gold_triangle_risk import GoldTriangleStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.niuniu_strategy_v3 import NiuniuStrategy
from strategy.tdi_strategy import TDIStrategy
from strategy.uss_market_forecast import MarketForecastStrategy
from strategy.cpgw_strategy import CPGWStrategy
from strategy.bollinger_bands_strategy import BollingerBandsStrategy
from backtest.strategy_backtest import StrategyBacktest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    加载回测配置
    
    Returns:
        Dict[str, Any]: 回测配置
    """
    return {
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 1000000,
        'commission_rate': 0.0003,
        'slippage': 0.001,
        'risk_free_rate': 0.02
    }

def run_backtest() -> Dict[str, Any]:
    """
    运行回测
    
    Returns:
        Dict[str, Any]: 回测结果
    """
    try:
        # 加载配置
        config = load_config()
        
        # 加载数据
        data_path = Path('data/stock_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # 初始化回测
        backtest = StrategyBacktest(data, config)
        
        # 添加策略
        strategies = [
            GoldTriangleStrategy(),
            MomentumStrategy(),
            NiuniuStrategy(),
            TDIStrategy(),
            MarketForecastStrategy(),
            CPGWStrategy(),
            BollingerBandsStrategy()
        ]
        
        for strategy in strategies:
            backtest.add_strategy(strategy)
            
        # 运行回测
        results = backtest.run_backtest()
        
        # 生成报告
        report = backtest.generate_report()
        logger.info(report)
        
        return results
        
    except Exception as e:
        logger.error(f"运行回测时出错: {e}")
        return {}

if __name__ == '__main__':
    results = run_backtest()
    if results:
        logger.info("回测完成")
        logger.info(f"总收益率: {results['total_return']:.2%}")
        logger.info(f"年化收益率: {results['annual_return']:.2%}")
        logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    else:
        logger.error("回测失败")
