from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import json
import os
import pymysql
from sqlalchemy import text

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
    """从数据库获取市场数据"""
    try:
        from data.data_updater import DatabaseManager
        from config.trading_config import default_config
        
        # 数据库配置
        db_config = {
            "host": default_config.database.host,
            "port": default_config.database.port,
            "user": default_config.database.user,
            "password": default_config.database.password,
            "database": default_config.database.database
        }
        
        # 创建数据库管理器
        db_manager = DatabaseManager(db_config)
        
        # 构建SQL查询
        query = text("""
        SELECT Date, Open, High, Low, Close, Volume, AdjClose, Dividends, StockSplits
        FROM stock_code_time 
        WHERE Code = :symbol 
        AND Date BETWEEN :start_date AND :end_date
        ORDER BY Date
        """)
        
        # 执行查询
        with db_manager.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "symbol": symbol,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d')
                }
            )
        
        # 确保数据不为空
        if df.empty:
            print("获取的数据为空")
            return pd.DataFrame()
        
        # 设置日期为索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 打印基本信息
        print(f"\n获取的数据基本信息:")
        print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
        print(f"总记录数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        
        return df
        
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

def get_historical_best_params() -> Dict[str, Any]:
    """获取历史最优参数"""
    return {
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

def verify_parameters(params: Dict[str, Any]) -> bool:
    """验证参数是否与历史最优一致"""
    historical_params = get_historical_best_params()
    for key, value in historical_params.items():
        if params[key] != value:
            print(f"参数 {key} 不一致: 当前值 {params[key]}, 历史最优值 {value}")
            return False
    return True

def reset_to_historical_best() -> Dict[str, Any]:
    """重置到历史最优参数"""
    print("\n正在重置到历史最优参数...")
    params = get_historical_best_params()
    print("参数已重置为历史最优值:")
    for key, value in params.items():
        print(f"{key}: {value}")
    return params

def check_data_consistency(market_data: pd.DataFrame, historical_data: pd.DataFrame) -> bool:
    """检查数据一致性"""
    if market_data.empty or historical_data.empty:
        print("数据为空，无法比较")
        return False
    
    # 检查日期范围
    market_start = market_data.index.min()
    market_end = market_data.index.max()
    historical_start = historical_data.index.min()
    historical_end = historical_data.index.max()
    
    if market_start != historical_start or market_end != historical_end:
        print(f"日期范围不一致:")
        print(f"当前数据: {market_start} 到 {market_end}")
        print(f"历史数据: {historical_start} 到 {historical_end}")
        return False
    
    # 检查数据点数量
    if len(market_data) != len(historical_data):
        print(f"数据点数量不一致:")
        print(f"当前数据: {len(market_data)} 个点")
        print(f"历史数据: {len(historical_data)} 个点")
        return False
    
    # 检查关键价格数据
    for col in ['Open', 'High', 'Low', 'Close']:
        if not np.allclose(market_data[col], historical_data[col], rtol=1e-5):
            print(f"{col} 价格数据不一致")
            return False
    
    return True

def load_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """加载历史最优数据"""
    try:
        # 从历史最优结果文件加载数据
        with open('backtest/historical_data.json', 'r') as f:
            historical_data = json.load(f)
        
        # 转换为DataFrame
        df = pd.DataFrame(historical_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 筛选日期范围
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df
        
    except Exception as e:
        print(f"加载历史数据时出错: {e}")
        return pd.DataFrame()

def main():
    # 使用历史最优参数
    params = {
        'position_size': 0.25,  # 增加仓位大小
        'max_position': 150,    # 增加最大持仓
        'cooldown_period': 2,   # 减少冷却期
        'stop_loss': 0.12,      # 放宽止损
        'take_profit': 0.25,    # 放宽止盈
        'lookback_period': 10,  # 减少回看期
        'overbought': 75,       # 提高超买阈值
        'oversold': 25,         # 降低超卖阈值
        'fast_ma': 3,           # 减少快速均线周期
        'slow_ma': 15           # 减少慢速均线周期
    }
    
    print("\n===== 使用历史最优参数 =====")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # 获取市场数据
    market_data = get_market_data('SPY', pd.Timestamp('2024-01-01'), pd.Timestamp('2025-04-11'))
    if market_data.empty:
        print("无法获取市场数据，退出回测")
        return
    
    # 运行回测
    results = run_backtest(params, market_data)
    
    # 输出回测结果
    print("\n===== 回测结果 =====")
    print(f"初始资金: {results['initial_capital']:,.2f}")
    print(f"最终资金: {results['final_capital']:,.2f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    
    # 绘制结果图表
    plot_results(results)

if __name__ == "__main__":
    main() 