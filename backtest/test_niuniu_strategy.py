import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from strategy.niuniu_strategy_v3 import NiuniuStrategyV3
from backtest.backtest_engine import BacktestEngine

def get_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取市场数据"""
    # 这里使用模拟数据，实际应用中应该从数据源获取
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.normal(100, 1, len(dates))
    data['high'] = data['open'] + np.random.normal(0.5, 0.2, len(dates))
    data['low'] = data['open'] - np.random.normal(0.5, 0.2, len(dates))
    data['close'] = (data['high'] + data['low']) / 2
    data['volume'] = np.random.randint(1000, 10000, len(dates))
    return data

def run_backtest(params, market_data=None):
    """运行单次回测"""
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-04-11')
    initial_capital = 1_000_000
    
    # 如果没有传入市场数据，则获取
    if market_data is None:
        market_data = get_market_data('SPY', start_date, end_date)
        if market_data.empty:
            return None
    
    # 创建策略实例
    strategy = NiuniuStrategyV3(
        parameters={
            'fast_period': params['fast_period'],
            'slow_period': params['slow_period'],
            'rsi_period': params['rsi_period'],
            'rsi_oversold': params['rsi_oversold'],
            'rsi_overbought': params['rsi_overbought'],
            'adx_period': params['adx_period'],
            'adx_threshold': params['adx_threshold'],
            'stop_loss': params['stop_loss'],
            'take_profit': params['take_profit'],
            'trailing_stop': params['trailing_stop'],
            'max_position_size': params['max_position_size'],
            'min_position_size': params['min_position_size'],
            'min_hold_days': params['min_hold_days'],
            'max_trades_per_day': params['max_trades_per_day']
        }
    )
    
    # 使用遗传算法优化参数
    optimized_params = strategy.optimize_parameters(
        market_data,
        param_grid={
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40],
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [20, 25, 30],
            'rsi_overbought': [70, 75, 80],
            'adx_period': [10, 14, 20],
            'adx_threshold': [15, 20, 25],
            'stop_loss': [0.05, 0.08, 0.1],
            'take_profit': [0.15, 0.2, 0.25],
            'trailing_stop': [0.03, 0.05, 0.08],
            'max_position_size': [0.6, 0.8, 1.0],
            'min_position_size': [0.2, 0.4, 0.6],
            'min_hold_days': [2, 3, 5],
            'max_trades_per_day': [2, 3, 5]
        },
        metric='sharpe_ratio'
    )
    
    # 更新策略参数
    strategy.update_parameters(optimized_params)
    
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
    
    return results

if __name__ == '__main__':
    # 设置默认参数
    params = {
        'fast_period': 10,
        'slow_period': 30,
        'rsi_period': 14,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'adx_period': 14,
        'adx_threshold': 20,
        'stop_loss': 0.08,
        'take_profit': 0.25,
        'trailing_stop': 0.05,
        'max_position_size': 0.8,
        'min_position_size': 0.4,
        'min_hold_days': 3,
        'max_trades_per_day': 3,
        'position_size': 0.1,
        'max_position': 0.5,
        'cooldown_period': 5
    }
    
    # 运行回测
    results = run_backtest(params)
    
    if results:
        print("回测结果:")
        print(f"总收益率: {results['total_return']:.2%}")
        print(f"年化收益率: {results['annualized_return']:.2%}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"胜率: {results['win_rate']:.2%}")
        print(f"盈亏比: {results['profit_loss_ratio']:.2f}")
    else:
        print("回测失败，请检查数据或参数设置") 