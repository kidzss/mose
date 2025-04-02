import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import datetime as dt
import logging
from pathlib import Path
import importlib
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pickle

from .strategy_scorer import StrategyScorer, StrategyScore
from .strategy_evaluator import StrategyEvaluator
from monitor.data_manager import DataManager
from monitor.stock_monitor_manager import StockMonitorManager
from strategy.strategy_base import Strategy
from strategy.strategy_factory import StrategyFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiStrategyBacktest")


# 定义一个独立函数用于并行处理
def _process_single_stock(stock_info, strategy_params, market_regime, db_config, start_date, end_date):
    """
    处理单个股票的回测，用于并行处理
    
    参数:
        stock_info: 股票信息 (symbol, name)
        strategy_params: 策略参数字典 {策略名称: 策略参数}
        market_regime: 市场环境
        db_config: 数据库配置
        start_date: 回测开始日期
        end_date: 回测结束日期
        
    返回:
        (股票代码, 回测结果字典)
    """
    symbol = stock_info['symbol']

    try:
        # 在进程内创建数据管理器
        data_manager = DataManager(db_config)

        # 获取股票数据
        data = data_manager.get_historical_data(symbol, start_date, end_date)

        if data is None or data.empty:
            logger.warning(f"无法获取股票 {symbol} 的数据")
            return symbol, {}

        # 创建策略工厂
        factory = StrategyFactory()

        # 创建评分器
        scorer = StrategyScorer(market_regime=market_regime)

        # 运行每个策略
        results = {}
        for strategy_name, params in strategy_params.items():
            try:
                # 创建策略实例
                strategy = factory.create_strategy(strategy_name, params)

                if not strategy:
                    logger.warning(f"无法创建策略 {strategy_name}")
                    continue

                # 生成信号
                signals = strategy.generate_signals(data.copy())

                # 评估策略
                evaluator = StrategyEvaluator(data)
                metrics = evaluator.analyze_trades(signals['signal'])

                # 添加策略名称和股票代码到指标中
                metrics['strategy_name'] = strategy_name
                metrics['symbol'] = symbol

                # 计算得分
                score = scorer.calculate_score(metrics)

                # 保存结果
                results[strategy_name] = score

            except Exception as e:
                logger.error(f"运行策略 {strategy_name} 时出错: {e}")

        return symbol, results

    except Exception as e:
        logger.error(f"处理股票 {symbol} 时出错: {e}")
        return symbol, {}


class MultiStrategyBacktest:
    """多策略回测系统"""

    def __init__(
            self,
            data_manager: DataManager,
            stock_manager: StockMonitorManager,
            start_date: Optional[dt.datetime] = None,
            end_date: Optional[dt.datetime] = None,
            strategy_names: Optional[List[str]] = None,
            market_regime: str = 'normal'
    ):
        """
        初始化多策略回测系统
        
        参数:
            data_manager: 数据管理器
            stock_manager: 股票管理器
            start_date: 回测开始日期
            end_date: 回测结束日期
            strategy_names: 策略名称列表，如果为None则使用所有可用策略
            market_regime: 市场环境
        """
        self.data_manager = data_manager
        self.stock_manager = stock_manager
        self.start_date = start_date or dt.datetime.now() - dt.timedelta(days=365)
        self.end_date = end_date or dt.datetime.now()
        self.market_regime = market_regime

        # 初始化策略评分器
        self.scorer = StrategyScorer(market_regime=market_regime)
        self.strategy_results: Dict[str, List[StrategyScore]] = {}

        # 初始化策略工厂
        self.strategy_factory = StrategyFactory()

        # 加载策略
        if strategy_names:
            self.strategies = {}
            for name in strategy_names:
                strategy = self.strategy_factory.create_strategy(name)
                if strategy:
                    self.strategies[name] = strategy
        else:
            # 加载所有策略
            self.strategies = self.strategy_factory.create_all_strategies()

        logger.info(f"MultiStrategyBacktest初始化完成，加载了{len(self.strategies)}个策略")
        logger.info(f"市场环境: {market_regime}")
        logger.info(f"回测期间: {self.start_date} 至 {self.end_date}")

    def run_backtest(self, parallel: bool = True) -> Dict[str, List[StrategyScore]]:
        """
        运行多策略回测
        
        参数:
            parallel: 是否并行运行回测
            
        返回:
            回测结果
        """
        try:
            # 获取监控的股票列表
            stocks = self.stock_manager.get_monitored_stocks()
            if stocks.empty:
                logger.warning("没有可回测的股票")
                return {}

            logger.info(f"开始回测 {len(stocks)} 只股票，使用 {len(self.strategies)} 个策略")

            # 对每个股票进行回测
            if parallel:
                self._run_parallel_backtest(stocks)
            else:
                self._run_sequential_backtest(stocks)

            logger.info("回测完成")
            return self.strategy_results

        except Exception as e:
            logger.error(f"运行回测时出错: {e}")
            return {}

    def _run_parallel_backtest(self, stocks: pd.DataFrame) -> None:
        """
        并行运行回测
        
        参数:
            stocks: 股票列表
        """
        with ProcessPoolExecutor() as executor:
            futures = []

            # 准备策略参数字典，而不是策略实例
            strategy_params = {}
            for name, strategy in self.strategies.items():
                strategy_params[name] = strategy.get_strategy_info().get('parameters', {})

            # 将数据库配置转换为字典
            db_config_dict = {
                'host': self.data_manager.db_config['host'],
                'port': self.data_manager.db_config['port'],
                'user': self.data_manager.db_config['user'],
                'password': self.data_manager.db_config['password'],
                'database': self.data_manager.db_config['database']
            }

            # 提交任务
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                futures.append(
                    executor.submit(
                        _process_single_stock,
                        {'symbol': symbol},
                        strategy_params,
                        self.market_regime,
                        db_config_dict,
                        self.start_date,
                        self.end_date
                    )
                )

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    symbol, stock_results = future.result()
                    if stock_results:
                        for strategy_name, score in stock_results.items():
                            if strategy_name not in self.strategy_results:
                                self.strategy_results[strategy_name] = []
                            self.strategy_results[strategy_name].append((symbol, score))
                except Exception as e:
                    logger.error(f"处理回测结果时出错: {e}")

    def _run_sequential_backtest(self, stocks: pd.DataFrame) -> None:
        """
        顺序运行回测
        
        参数:
            stocks: 股票列表
        """
        for _, stock in stocks.iterrows():
            symbol = stock['symbol']
            symbol, stock_results = self._backtest_single_stock(symbol)

            if stock_results:
                for strategy_name, score in stock_results.items():
                    if strategy_name not in self.strategy_results:
                        self.strategy_results[strategy_name] = []
                    self.strategy_results[strategy_name].append(score)

    def _backtest_single_stock(self, symbol: str) -> Tuple[str, Dict[str, StrategyScore]]:
        """
        对单个股票进行回测
        
        参数:
            symbol: 股票代码
            
        返回:
            (股票代码, 回测结果字典)
        """
        try:
            # 获取股票数据
            data = self.data_manager.get_stock_data(symbol, self.start_date, self.end_date)

            if data is None or len(data) == 0:
                logger.warning(f"无法获取股票 {symbol} 的数据")
                return symbol, {}

            # 确保数据包含所有必要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'Adj Close' and 'AdjClose' in data.columns:
                        data['Adj Close'] = data['AdjClose']
                    else:
                        data[col] = data['Close']

            # 确保索引是日期类型
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # 运行每个策略
            results = {}
            for strategy_name, strategy in self.strategies.items():
                try:
                    # 首先计算技术指标
                    data_with_indicators = strategy.calculate_indicators(data.copy())

                    if data_with_indicators is None or data_with_indicators.empty:
                        logger.warning(f"策略 {strategy_name} 计算指标后数据为空")
                        continue

                    # 生成信号
                    signals = strategy.generate_signals(data_with_indicators)

                    if signals is None or 'signal' not in signals.columns:
                        logger.warning(f"策略 {strategy_name} 未能生成有效信号")
                        continue

                    # 评估策略
                    evaluator = StrategyEvaluator(data)
                    metrics = evaluator.analyze_trades(signals['signal'])

                    # 添加策略名称和股票代码
                    metrics_dict = vars(metrics)
                    metrics_dict['strategy_name'] = strategy_name
                    metrics_dict['symbol'] = symbol

                    # 计算策略得分
                    score = self.scorer.calculate_score(metrics_dict)
                    results[strategy_name] = score

                    logger.info(f"股票 {symbol} 使用策略 {strategy_name} 的回测得分: {score.total_score:.2f}")

                except Exception as e:
                    logger.error(f"运行策略 {strategy_name} 于股票 {symbol} 时出错: {e}")

            return symbol, results

        except Exception as e:
            logger.error(f"回测股票 {symbol} 时出错: {e}")
            return symbol, {}

    # def _create_sample_data(self, symbol: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    #     """
    #     创建样本数据用于测试
    #
    #     参数:
    #         symbol: 股票代码
    #         start_date: 开始日期
    #         end_date: 结束日期
    #
    #     返回:
    #         样本数据DataFrame
    #     """
    #     # 生成日期范围
    #     date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    #
    #     # 初始价格和波动率
    #     initial_price = 100.0
    #     volatility = 0.02
    #
    #     # 生成价格序列
    #     np.random.seed(hash(symbol) % 10000)  # 使用股票代码作为随机种子，确保每个股票有不同但可重复的数据
    #
    #     # 生成随机价格变动
    #     returns = np.random.normal(0.0005, volatility, size=len(date_range))
    #     price_series = initial_price * (1 + returns).cumprod()
    #
    #     # 创建OHLCV数据
    #     data = pd.DataFrame({
    #         'Open': price_series * (1 - 0.005 * np.random.random(len(date_range))),
    #         'High': price_series * (1 + 0.01 * np.random.random(len(date_range))),
    #         'Low': price_series * (1 - 0.01 * np.random.random(len(date_range))),
    #         'Close': price_series,
    #         'Volume': np.random.randint(100000, 10000000, size=len(date_range)),
    #         'Adj Close': price_series  # 添加Adj Close列，这里简单地使用Close的值
    #     }, index=date_range)
    #
    #     # 确保High >= Open >= Close >= Low的关系
    #     for i in range(len(data)):
    #         high = max(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['High'])
    #         low = min(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['Low'])
    #         data.iloc[i, data.columns.get_loc('High')] = high
    #         data.iloc[i, data.columns.get_loc('Low')] = low
    #
    #     # 计算移动平均线
    #     for period in [5, 10, 20, 100]:
    #         data[f'SMA_{period}'] = data['Adj Close'].rolling(window=period).mean()
    #
    #     return data

    def generate_report(self) -> str:
        """
        生成回测报告
        
        返回:
            回测报告文本
        """
        try:
            if not self.strategy_results:
                return "没有回测结果可供生成报告"

            report = "多策略回测报告\n"
            report += "=" * 50 + "\n\n"
            report += f"回测期间: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}\n"
            report += f"市场环境: {self.market_regime}\n"
            report += f"策略数量: {len(self.strategy_results)}\n"
            report += f"股票数量: {len(next(iter(self.strategy_results.values()))) if self.strategy_results else 0}\n\n"

            # 计算每个策略的平均得分
            strategy_avg_scores = {}
            for strategy_name, scores in self.strategy_results.items():
                avg_score = StrategyScore(
                    strategy_name=strategy_name,
                    symbol="ALL",
                    total_score=np.mean([s.total_score for s in scores]),
                    profit_score=np.mean([s.profit_score for s in scores]),
                    adaptability_score=np.mean([s.adaptability_score for s in scores]),
                    robustness_score=np.mean([s.robustness_score for s in scores]),
                    details={
                        'total_return': np.mean([s.details.get('total_return', 0) for s in scores]),
                        'win_rate': np.mean([s.details.get('win_rate', 0) for s in scores]),
                        'profit_factor': np.mean([s.details.get('profit_factor', 0) for s in scores]),
                        'max_drawdown': np.mean([s.details.get('max_drawdown', 0) for s in scores]),
                        'sharpe_ratio': np.mean([s.details.get('sharpe_ratio', 0) for s in scores]),
                        'avg_holding_days': np.mean([s.details.get('avg_holding_days', 0) for s in scores])
                    }
                )
                strategy_avg_scores[strategy_name] = avg_score

            # 生成评分报告
            report += "策略平均表现:\n"
            report += self.scorer.generate_score_report(list(strategy_avg_scores.values()))

            # 为每个股票找出最佳策略
            report += "\n每个股票的最佳策略:\n"
            report += "-" * 50 + "\n"

            # 按股票分组
            stock_results = {}
            for strategy_name, scores in self.strategy_results.items():
                for score in scores:
                    symbol = score.symbol
                    if symbol not in stock_results:
                        stock_results[symbol] = []
                    stock_results[symbol].append(score)

            # 找出每个股票的最佳策略
            for symbol, scores in stock_results.items():
                best_score = max(scores, key=lambda x: x.total_score)
                report += f"股票 {symbol}: 最佳策略 = {best_score.strategy_name}, 得分 = {best_score.total_score:.2f}\n"

            # 保存报告
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)

            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"multi_strategy_backtest_{timestamp}.txt"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"回测报告已保存到: {report_file}")
            return report

        except Exception as e:
            logger.error(f"生成回测报告时出错: {e}")
            return str(e)

    def get_best_strategy_for_stock(self, symbol: str) -> Optional[str]:
        """
        获取股票的最佳策略
        
        参数:
            symbol: 股票代码
            
        返回:
            最佳策略名称
        """
        try:
            # 收集所有策略对该股票的得分
            scores = []
            for strategy_name, strategy_scores in self.strategy_results.items():
                for score in strategy_scores:
                    if score.symbol == symbol:
                        scores.append(score)

            if not scores:
                return None

            # 找出得分最高的策略
            best_score = max(scores, key=lambda x: x.total_score)
            return best_score.strategy_name

        except Exception as e:
            logger.error(f"获取股票 {symbol} 的最佳策略时出错: {e}")
            return None

    def get_strategy_allocation(self, symbol: str, top_n: int = 3) -> Dict[str, float]:
        """
        获取股票的策略分配权重
        
        参数:
            symbol: 股票代码
            top_n: 使用前N个策略
            
        返回:
            策略分配权重字典
        """
        try:
            # 收集所有策略对该股票的得分
            scores = []
            for strategy_name, strategy_scores in self.strategy_results.items():
                for score in strategy_scores:
                    if score.symbol == symbol:
                        scores.append(score)

            if not scores:
                return {}

            # 按得分排序
            scores.sort(key=lambda x: x.total_score, reverse=True)

            # 取前N个策略
            top_scores = scores[:top_n]

            # 计算总得分
            total_score = sum(score.total_score for score in top_scores)

            # 计算权重
            weights = {}
            for score in top_scores:
                weights[score.strategy_name] = score.total_score / total_score

            return weights

        except Exception as e:
            logger.error(f"获取股票 {symbol} 的策略分配权重时出错: {e}")
            return {}
