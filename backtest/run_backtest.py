import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import DataLoader
from backtest.backtest_runner import BacktestRunner, run_parallel_backtest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",  # root用户没有密码
    "database": "mose",
    "table_name": "stock_time_code"  # 使用正确的表名
}


def cpgw_strategy(data: pd.DataFrame, params: Dict) -> pd.Series:
    """CPGW策略实现"""
    # 计算技术指标
    sma_short = data['Close'].rolling(window=int(params.get('sma_short', 5))).mean()
    sma_long = data['Close'].rolling(window=int(params.get('sma_long', 20))).mean()

    # 生成交易信号
    positions = pd.Series(0, index=data.index)
    positions.loc[sma_short > sma_long] = 1  # 金叉做多
    positions.loc[sma_short < sma_long] = 0  # 死叉平仓

    return positions


def run_single_backtest(
        symbol: str,
        start_date: datetime,
        end_date: datetime = None,
        strategy_func=cpgw_strategy,
        param_ranges: Dict = None,
        benchmark_symbol: str = 'SPY'  # 使用SPY作为默认基准
) -> None:
    """运行单个股票的回测"""
    try:
        # 初始化数据加载器
        loader = DataLoader(DB_CONFIG)

        # 准备数据
        stock_data = loader.load_stock_data(symbol, start_date, end_date)
        benchmark_data = loader.load_benchmark_data(benchmark_symbol, start_date, end_date)  # 加载基准数据

        if stock_data.empty:
            logger.error(f"无法获取股票 {symbol} 的数据")
            return

        # 打印数据结构
        print("\n数据结构:")
        print(stock_data.head())
        print("\n数据列:")
        print(stock_data.columns)

        # 设置默认参数范围（如果未提供）
        if param_ranges is None:
            param_ranges = {
                'sma_short': [5, 10, 15],
                'sma_long': [20, 30, 40]
            }

        # 初始化回测运行器
        runner = BacktestRunner(
            data=stock_data,
            strategy_func=strategy_func,
            benchmark_data=benchmark_data,  # 传入基准数据
            param_ranges=param_ranges
        )

        # 运行回测
        result = runner.run_backtest()

        # 生成报告
        report = runner.generate_report(result)
        print(report)

        # 绘制结果
        positions = strategy_func(stock_data, result.optimization_result.parameters)
        runner.plot_results(positions)

    except Exception as e:
        logger.error(f"运行回测时出错: {str(e)}")
        raise


def run_portfolio_backtest(
        symbols: List[str],
        start_date: datetime,
        end_date: datetime = None,
        strategy_func=cpgw_strategy,
        param_ranges: Dict = None,
        n_workers: int = 4
) -> None:
    """运行投资组合回测"""
    try:
        # 初始化数据加载器
        loader = DataLoader(DB_CONFIG)

        # 加载多个股票的数据
        data_dict = loader.load_multiple_stocks(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        if not data_dict:
            logger.error("无法获取任何股票数据")
            return

        # 运行并行回测
        results = run_parallel_backtest(
            data_dict=data_dict,
            strategy_func=strategy_func,
            param_ranges=param_ranges,
            n_workers=n_workers
        )

        # 生成汇总报告
        print("\n投资组合回测报告")
        print("================\n")

        for symbol, result in results.items():
            print(f"\n{symbol} 回测结果:")
            print("-" * 20)
            print(f"总收益率: {result.strategy_metrics.total_return:.2f}%")
            print(f"年化收益率: {result.strategy_metrics.annual_return:.2f}%")
            print(f"夏普比率: {result.strategy_metrics.sharpe_ratio:.2f}")
            print(f"最大回撤: {result.strategy_metrics.max_drawdown:.2f}%")
            print(f"胜率: {result.strategy_metrics.win_rate:.2%}")

    except Exception as e:
        logger.error(f"运行投资组合回测时出错: {str(e)}")


def main():
    """主函数"""
    # 设置回测参数
    symbol = "AAPL"  # 示例股票
    start_date = datetime(2022, 1, 1)
    end_date = datetime.now()

    # 运行单个股票回测
    print(f"\n运行 {symbol} 的回测...")
    run_single_backtest(symbol, start_date, end_date)

    # 运行投资组合回测
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]  # 示例投资组合
    print("\n运行投资组合回测...")
    run_portfolio_backtest(symbols, start_date, end_date)


if __name__ == "__main__":
    main()
