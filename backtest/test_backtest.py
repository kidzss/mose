from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import json
import os
import yfinance as yf

from strategy.cpgw_strategy import CPGWStrategy
from backtest.backtest_engine import BacktestEngine

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def save_best_results(results: Dict[str, Any], params: Dict[str, Any], filename: str = 'backtest/best_results.json'):
    """保存最佳回测结果"""
    # 将结果中的时间戳转换为字符串
    results_copy = {
        'initial_capital': float(results['initial_capital']),
        'final_capital': float(results['final_capital']),
        'total_return': float(results['total_return']),
        'annual_return': float(results['annual_return']),
        'sharpe_ratio': float(results['sharpe_ratio']),
        'max_drawdown': float(results['max_drawdown']),
        'win_rate': float(results['win_rate']),
        'total_trades': int(results['total_trades'])
    }
    
    # 添加策略参数
    strategy_params = {
        'lookback_period': 14,
        'overbought': 70,
        'oversold': 30,
        'fast_ma': 5,
        'slow_ma': 20
    }
    
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results_copy,
        'params': {**params, **strategy_params}  # 合并交易参数和策略参数
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_best_results(filename: str = 'backtest/best_results.json') -> tuple[Dict[str, Any], Dict[str, Any]]:
    """加载最佳回测结果"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['results'], data['params']
    return None, None

def compare_results(new_results: Dict[str, Any], best_results: Dict[str, Any]) -> bool:
    """比较回测结果，返回新结果是否更好"""
    if best_results is None:
        return True
    
    # 使用年化收益率作为主要比较指标
    new_annual_return = new_results['annual_return']
    best_annual_return = best_results['annual_return']
    
    # 如果年化收益率相近（差异小于1%），则比较夏普比率
    if abs(new_annual_return - best_annual_return) < 0.01:
        return new_results['sharpe_ratio'] > best_results['sharpe_ratio']
    
    return new_annual_return > best_annual_return

def get_market_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """获取真实市场数据"""
    try:
        # 使用yfinance获取数据
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # 确保数据不为空
        if data.empty:
            print("获取的数据为空")
            return pd.DataFrame()
        
        # 处理多级索引列名
        if isinstance(data.columns, pd.MultiIndex):
            # 如果是多级索引，只保留第一级
            data.columns = data.columns.get_level_values(0)
        
        # 打印基本信息
        print(f"\n获取的真实数据基本信息:")
        print(f"日期范围: {data.index.min()} 到 {data.index.max()}")
        print(f"总记录数: {len(data)}")
        print(f"列名: {data.columns.tolist()}")
        
        # 添加其他必要列
        data['Dividends'] = 0
        data['Stock Splits'] = 0
        data['Capital Gains'] = 0
        
        return data
        
    except Exception as e:
        print(f"获取市场数据时出错: {e}")
        return pd.DataFrame()

def plot_results(results):
    """
    绘制回测结果图表
    """
    plt.figure(figsize=(15, 10))
    
    # 如果存在权益曲线数据则绘制
    if 'equity_curve' in results:
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(results['equity_curve'], label='权益曲线')
        ax1.set_title('回测结果')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('权益')
        ax1.legend()
        
        # 如果存在回撤曲线数据则绘制
        if 'drawdown_curve' in results:
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(results['drawdown_curve'], label='回撤曲线', color='red')
            ax2.set_title('回撤情况')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('回撤率')
            ax2.legend()
    
    plt.tight_layout()
    plt.show()

def optimize_parameters():
    """参数优化函数"""
    # 定义参数搜索空间
    param_grid = {
        'position_size': [0.1, 0.15, 0.2],
        'max_position': [50, 75, 100],
        'cooldown_period': [5, 7, 10],
        'stop_loss': [0.05, 0.08, 0.1],
        'take_profit': [0.1, 0.15, 0.2],
        'lookback_period': [10, 14, 20],
        'overbought': [65, 70, 75],
        'oversold': [25, 30, 35],
        'fast_ma': [3, 5, 8],
        'slow_ma': [15, 20, 25]
    }
    
    best_results = None
    best_params = None
    
    # 遍历参数组合
    for position_size in param_grid['position_size']:
        for max_position in param_grid['max_position']:
            for cooldown_period in param_grid['cooldown_period']:
                for stop_loss in param_grid['stop_loss']:
                    for take_profit in param_grid['take_profit']:
                        for lookback_period in param_grid['lookback_period']:
                            for overbought in param_grid['overbought']:
                                for oversold in param_grid['oversold']:
                                    for fast_ma in param_grid['fast_ma']:
                                        for slow_ma in param_grid['slow_ma']:
                                            params = {
                                                'position_size': position_size,
                                                'max_position': max_position,
                                                'cooldown_period': cooldown_period,
                                                'stop_loss': stop_loss,
                                                'take_profit': take_profit,
                                                'lookback_period': lookback_period,
                                                'overbought': overbought,
                                                'oversold': oversold,
                                                'fast_ma': fast_ma,
                                                'slow_ma': slow_ma
                                            }
                                            
                                            # 运行回测
                                            results = run_backtest(params)
                                            
                                            # 更新最佳结果
                                            if best_results is None or compare_results(results, best_results):
                                                best_results = results
                                                best_params = params
                                                print(f"\n找到新的最佳参数组合:")
                                                print(f"年化收益率: {results['annual_return']:.2%}")
                                                print(f"夏普比率: {results['sharpe_ratio']:.2f}")
                                                print(f"最大回撤: {results['max_drawdown']:.2%}")
                                                print(f"胜率: {results['win_rate']:.2%}")
                                                print(f"参数: {params}")
    
    return best_params, best_results

def run_backtest(params):
    """运行单次回测"""
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-04-11')
    initial_capital = 1_000_000
    
    # 获取市场数据
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
    results = engine.run_backtest(market_data, strategy, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    return results

def genetic_optimization(initial_params, population_size=10, generations=5, mutation_rate=0.1):
    """遗传算法优化参数"""
    def mutate_param(param, param_range):
        """参数变异"""
        if isinstance(param, float):
            return max(min(param * (1 + np.random.normal(0, mutation_rate)), param_range[1]), param_range[0])
        else:
            return np.random.randint(param_range[0], param_range[1] + 1)
    
    def crossover(parent1, parent2):
        """参数交叉"""
        child = {}
        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    # 定义参数范围
    param_ranges = {
        'position_size': [0.1, 0.3],
        'max_position': [50, 150],
        'cooldown_period': [3, 15],
        'stop_loss': [0.03, 0.15],
        'take_profit': [0.05, 0.25],
        'lookback_period': [5, 30],
        'overbought': [60, 80],
        'oversold': [20, 40],
        'fast_ma': [2, 10],
        'slow_ma': [10, 30]
    }
    
    # 初始化种群
    population = [initial_params]
    for _ in range(population_size - 1):
        individual = {}
        for key, value in initial_params.items():
            individual[key] = mutate_param(value, param_ranges[key])
        population.append(individual)
    
    best_params = initial_params
    best_results = run_backtest(initial_params)
    
    print("\n开始遗传算法优化...")
    for generation in range(generations):
        print(f"\n第 {generation + 1} 代:")
        
        # 评估种群
        results = []
        for params in population:
            result = run_backtest(params)
            if result:
                results.append((params, result))
        
        # 按年化收益率排序
        results.sort(key=lambda x: x[1]['annual_return'], reverse=True)
        
        # 更新最佳结果
        if results and compare_results(results[0][1], best_results):
            best_params = results[0][0]
            best_results = results[0][1]
            print(f"\n发现新的最佳参数组合:")
            print(f"年化收益率: {best_results['annual_return']:.2%}")
            print(f"夏普比率: {best_results['sharpe_ratio']:.2f}")
            print(f"最大回撤: {best_results['max_drawdown']:.2%}")
            print(f"胜率: {best_results['win_rate']:.2%}")
            print(f"参数: {best_params}")
        
        # 选择精英个体
        elite_size = max(2, population_size // 4)
        elite = [params for params, _ in results[:elite_size]]
        
        # 生成新一代
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = np.random.choice(elite)
            parent2 = np.random.choice(elite)
            child = crossover(parent1, parent2)
            
            # 变异
            for key in child:
                if np.random.random() < mutation_rate:
                    child[key] = mutate_param(child[key], param_ranges[key])
            
            new_population.append(child)
        
        population = new_population
    
    return best_params, best_results

def main():
    # 设置回测参数
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-04-11')
    initial_capital = 1_000_000
    
    # 使用当前参数作为初始参数
    current_params = {
        'position_size': 0.15,
        'max_position': 75,
        'cooldown_period': 7,
        'stop_loss': 0.08,
        'take_profit': 0.15,
        'lookback_period': 14,
        'overbought': 70,
        'oversold': 30,
        'fast_ma': 5,
        'slow_ma': 20
    }
    
    print("\n===== 使用当前参数 =====")
    for key, value in current_params.items():
        print(f"{key}: {value}")
    
    # 获取市场数据
    market_data = get_market_data('SPY', start_date, end_date)
    if market_data.empty:
        print("无法获取市场数据，退出回测")
        return
    
    # 运行初始回测
    initial_results = run_backtest(current_params)
    print("\n===== 初始回测结果 =====")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"最终资金: {initial_results['final_capital']:,.2f}")
    print(f"总收益率: {initial_results['total_return']:.2%}")
    print(f"年化收益率: {initial_results['annual_return']:.2%}")
    print(f"夏普比率: {initial_results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {initial_results['max_drawdown']:.2%}")
    print(f"胜率: {initial_results['win_rate']:.2%}")
    print(f"总交易次数: {initial_results['total_trades']}")
    
    # 进行遗传算法优化
    best_params, best_results = genetic_optimization(current_params)
    
    print("\n===== 优化后的最优参数组合 =====")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    print("\n===== 优化后的最优结果 =====")
    print(f"初始资金: {best_results['initial_capital']:,.2f}")
    print(f"最终资金: {best_results['final_capital']:,.2f}")
    print(f"总收益率: {best_results['total_return']:.2%}")
    print(f"年化收益率: {best_results['annual_return']:.2%}")
    print(f"夏普比率: {best_results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {best_results['max_drawdown']:.2%}")
    print(f"胜率: {best_results['win_rate']:.2%}")
    print(f"总交易次数: {best_results['total_trades']}")
    
    # 保存优化后的最佳结果
    save_best_results(best_results, best_params)
    
    # 绘制优化后的结果图表
    plot_results(best_results)

if __name__ == "__main__":
    main() 