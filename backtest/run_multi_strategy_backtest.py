import pandas as pd
import numpy as np
import datetime as dt
import logging
import argparse
import sys
import os
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.multi_strategy_backtest import MultiStrategyBacktest
from monitor.data_manager import DataManager
from monitor.stock_monitor_manager import StockMonitorManager
from config.trading_config import TradingConfig, default_config
from strategy.strategy_factory import StrategyFactory
from backtest.strategy_evaluator import StrategyEvaluator
from backtest.strategy_scorer import StrategyScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_multi_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_multi_strategy_backtest")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多策略回测系统')
    
    parser.add_argument('--start_date', type=str, default=None,
                        help='回测开始日期，格式：YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None,
                        help='回测结束日期，格式：YYYY-MM-DD')
    parser.add_argument('--strategies', type=str, default=None,
                        help='策略名称，用逗号分隔，例如：GoldTriangleStrategy,MomentumStrategy')
    parser.add_argument('--parallel', action='store_true',
                        help='是否并行运行回测')
    parser.add_argument('--market_regime', type=str, default='normal',
                        choices=['normal', 'volatile', 'trending', 'range'],
                        help='市场环境')
    parser.add_argument('--list_strategies', action='store_true',
                        help='列出所有可用策略')
    parser.add_argument('--strategy_info', type=str, default=None,
                        help='显示指定策略的详细信息')
    parser.add_argument('--top_n', type=int, default=3,
                        help='为每个股票选择前N个策略')
    
    return parser.parse_args()

def list_strategies():
    """列出所有可用策略"""
    factory = StrategyFactory()
    strategies = factory.get_all_strategies_info()
    
    print("\n可用策略列表:")
    print("=" * 50)
    
    for name, info in strategies.items():
        print(f"\n策略名称: {name}")
        print(f"版本: {info.get('version', '1.0.0')}")
        print(f"描述: {info.get('description', '无描述')}")
        print("-" * 50)
        
    return strategies

def show_strategy_info(strategy_name):
    """显示指定策略的详细信息"""
    factory = StrategyFactory()
    info = factory.get_strategy_info(strategy_name)
    
    if not info:
        print(f"策略 {strategy_name} 不存在")
        return
        
    print("\n策略详细信息:")
    print("=" * 50)
    print(f"策略名称: {strategy_name}")
    print(f"版本: {info.get('version', '1.0.0')}")
    print(f"描述: {info.get('description', '无描述')}")
    print("\n参数列表:")
    
    for param_name, param_value in info.get('parameters', {}).items():
        print(f"  - {param_name}: {param_value}")
        
    print("=" * 50)

# 创建样本数据函数
def create_sample_data(symbol: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    """
    创建样本数据用于测试
    
    参数:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        样本数据DataFrame
    """
    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 初始价格和波动率
    initial_price = 100.0
    volatility = 0.02
    
    # 生成价格序列
    np.random.seed(hash(symbol) % 10000)  # 使用股票代码作为随机种子，确保每个股票有不同但可重复的数据
    
    # 生成随机价格变动
    returns = np.random.normal(0.0005, volatility, size=len(date_range))
    price_series = initial_price * (1 + returns).cumprod()
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'Open': price_series * (1 - 0.005 * np.random.random(len(date_range))),
        'High': price_series * (1 + 0.01 * np.random.random(len(date_range))),
        'Low': price_series * (1 - 0.01 * np.random.random(len(date_range))),
        'Close': price_series,
        'Volume': np.random.randint(100000, 10000000, size=len(date_range))
    }, index=date_range)
    
    # 确保High >= Open >= Close >= Low的关系
    for i in range(len(data)):
        high = max(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['High'])
        low = min(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['Low'])
        data.iloc[i, data.columns.get_loc('High')] = high
        data.iloc[i, data.columns.get_loc('Low')] = low
    
    return data

# 修改DataManager类的get_latest_data方法
def get_sample_data(data_manager, symbol, start_date=None, end_date=None):
    """
    获取样本数据，如果数据库中没有数据则创建样本数据
    
    参数:
        data_manager: DataManager实例
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        股票数据DataFrame
    """
    # 尝试从数据管理器获取数据
    data = data_manager.get_latest_data(symbol)
    
    # 如果没有数据，创建样本数据
    if data is None or data.empty:
        logger.info(f"数据库中没有找到 {symbol} 的数据，创建样本数据用于测试")
        data = create_sample_data(symbol, start_date, end_date)
    
    return data

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 如果只是列出策略，则执行后退出
    if args.list_strategies:
        list_strategies()
        return
        
    # 如果只是显示策略信息，则执行后退出
    if args.strategy_info:
        show_strategy_info(args.strategy_info)
        return
    
    # 解析日期
    start_date = dt.datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = dt.datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # 解析策略列表
    strategies = args.strategies.split(',') if args.strategies else None
    
    logger.info("初始化多策略回测系统")
    logger.info(f"回测期间: {start_date} 至 {end_date}")
    logger.info(f"策略列表: {strategies}")
    logger.info(f"市场环境: {args.market_regime}")
    logger.info(f"并行运行: {args.parallel}")
    
    try:
        # 加载配置
        config = default_config
        
        # 将DatabaseConfig对象转换为字典
        db_config = {
            'host': config.database.host,
            'port': config.database.port,
            'user': config.database.user,
            'password': config.database.password,
            'database': config.database.database
        }
        
        # 初始化数据管理器
        data_manager = DataManager(db_config)
        
        # 初始化股票管理器
        stock_manager = StockMonitorManager(db_config)
        
        # 初始化多策略回测系统
        backtest = MultiStrategyBacktest(
            data_manager=data_manager,
            stock_manager=stock_manager,
            start_date=start_date,
            end_date=end_date,
            strategy_names=strategies,
            market_regime=args.market_regime
        )
        
        # 使用顺序执行模式进行回测，避免数据库连接的pickle错误
        logger.info("使用顺序执行模式进行回测，避免数据库连接的pickle错误")
        results = backtest.run_backtest(parallel=False)
        
        # 生成报告
        report = backtest.generate_report()
        
        # 打印报告
        print("\n" + report)
        
        # 为每个股票找出最佳策略
        stocks = stock_manager.get_monitored_stocks()
        if not stocks.empty:
            print("\n每个股票的最佳策略:")
            print("=" * 50)
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                best_strategy = backtest.get_best_strategy_for_stock(symbol)
                if best_strategy:
                    print(f"股票 {symbol}: 最佳策略 = {best_strategy}")
                    
            # 为每个股票计算策略分配权重
            print("\n每个股票的策略分配权重:")
            print("=" * 50)
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                weights = backtest.get_strategy_allocation(symbol, top_n=args.top_n)
                if weights:
                    print(f"\n股票 {symbol} 的策略分配:")
                    for strategy, weight in weights.items():
                        print(f"  - {strategy}: {weight:.2%}")
        
    except Exception as e:
        logger.error(f"运行多策略回测时出错: {e}")
        raise

if __name__ == "__main__":
    main() 