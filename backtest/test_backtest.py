import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import os
from strategy.cpgw_strategy import CPGWStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.backtest_logger import BacktestLogger

# 添加项目根目录到路径，确保能导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取市场数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        市场数据
    """
    # 这里应该从数据源获取数据
    # 为了测试，我们生成一些模拟数据
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    
    # 生成随机价格
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    
    # 生成OHLC数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.random.normal(0.01, 0.01, n)),
        'low': prices * (1 + np.random.normal(-0.01, 0.01, n)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    return data

def run_backtest(params: dict, market_data: pd.DataFrame = None) -> dict:
    """
    运行单次回测
    
    Args:
        params: 策略参数
        market_data: 市场数据
        
    Returns:
        回测结果
    """
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-04-11')
    initial_capital = 1_000_000
    
    # 如果没有传入市场数据，则获取
    if market_data is None:
        market_data = get_market_data('SPY', start_date, end_date)
        if market_data.empty:
            return None
    
    # 创建策略实例
    strategy = CPGWStrategy(
        lookback_period=params['lookback_period'],
        overbought=params['overbought'],
        oversold=params['oversold'],
        fast_ma=params['fast_ma'],
        slow_ma=params['slow_ma'],
        use_market_regime=True
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.001,
        position_size=params['position_size'],
        max_position=params['max_position'],
        cooldown_period=params['cooldown_period'],
        stop_loss=params['stop_loss'],
        take_profit=params['take_profit']
    )
    
    # 运行回测
    results = engine.run_backtest(market_data, strategy)
    
    # 保存结果
    engine.save_results('backtest/results')
    
    return results

def main():
    """主函数"""
    # 设置策略参数
    params = {
        'lookback_period': 14,
        'overbought': 70,
        'oversold': 30,
        'fast_ma': 5,
        'slow_ma': 20,
        'position_size': 0.1,
        'max_position': 0.5,
        'cooldown_period': 5,
        'stop_loss': 0.1,
        'take_profit': 0.2
    }
    
    # 运行回测
    results = run_backtest(params)
    
    if results:
        # 打印回测结果
        print("\n回测结果:")
        print(f"开始日期: {results['start_date']}")
        print(f"结束日期: {results['end_date']}")
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']*100:.2f}%")
        print(f"年化收益率: {results['annual_return']*100:.2f}%")
        print(f"波动率: {results['volatility']*100:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {results['sortino_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
        print(f"总交易次数: {results['total_trades']}")
        print(f"盈利交易次数: {results['winning_trades']}")
        print(f"胜率: {results['win_rate']*100:.2f}%")
        print(f"平均持仓天数: {results['avg_holding_days']:.1f}")
    else:
        print("回测失败")

if __name__ == "__main__":
    main() 