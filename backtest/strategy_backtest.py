import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
import json
import os

from monitor.data_fetcher import DataFetcher
from monitor.stock_monitor_manager import StockMonitorManager
from monitor.notification_manager import NotificationManager
from strategy.uss_gold_triangle_strategy import GoldTriangleStrategy
from strategy.uss_momentum_strategy import MomentumStrategy
from strategy.uss_niuniu_strategy import NiuniuStrategy
from strategy.uss_tdi_strategy import TDIStrategy
from strategy.uss_market_forecast_strategy import MarketForecastStrategy
from strategy.uss_cpgw_strategy import CPGWStrategy
from strategy.uss_volume_strategy import VolumeStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrategyBacktest")


class MonitorStrategy:
    """监控策略基类"""

    def __init__(self, name):
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.indicator_columns = []  # 子类定义需要的技术指标
        self.name = name

    def get_required_columns(self) -> List[str]:
        """获取策略所需的列"""
        return self.required_columns + self.indicator_columns

    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否包含所需的列"""
        return all(col in df.columns for col in self.get_required_columns())

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备策略所需的数据"""
        if not self.validate_data(df):
            raise ValueError("数据缺少必需的列")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = self.prepare_data(df)
        df = self.calculate_indicators(df)
        return self._generate_signals_impl(df)

    def _generate_signals_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """具体的信号生成逻辑，由子类实现"""
        raise NotImplementedError


class VolatilityBreakoutStrategy(MonitorStrategy):
    """波动率突破策略"""

    def __init__(
            self,
            volatility_window: int = 20,
            std_threshold: float = 2.0,
            volume_threshold: float = 2.0
    ):
        super().__init__('VolatilityBreakout')
        self.volatility_window = volatility_window
        self.std_threshold = std_threshold
        self.volume_threshold = volume_threshold
        self.indicator_columns = ['volatility', 'ma', 'std', 'upper_band', 'lower_band', 'volume_ma']

    def _generate_signals_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成波动率突破信号
        
        参数:
            data: 股票数据，包含OHLCV
            
        返回:
            带有信号的DataFrame
        """
        try:
            # 计算日收益率
            data['returns'] = data['close'].pct_change()

            # 计算波动率
            data['volatility'] = data['returns'].rolling(window=self.volatility_window).std()

            # 计算移动平均和标准差
            data['ma'] = data['close'].rolling(window=self.volatility_window).mean()
            data['std'] = data['close'].rolling(window=self.volatility_window).std()

            # 计算布林带
            data['upper_band'] = data['ma'] + self.std_threshold * data['std']
            data['lower_band'] = data['ma'] - self.std_threshold * data['std']

            # 计算成交量移动平均
            data['volume_ma'] = data['volume'].rolling(window=self.volatility_window).mean()

            # 生成信号
            data['signal'] = 0

            # 突破上轨且成交量放大
            upper_breakout = (data['close'] > data['upper_band']) & \
                             (data['volume'] > data['volume_ma'] * self.volume_threshold)

            # 突破下轨且成交量放大
            lower_breakout = (data['close'] < data['lower_band']) & \
                             (data['volume'] > data['volume_ma'] * self.volume_threshold)

            data.loc[upper_breakout, 'signal'] = 1  # 看多信号
            data.loc[lower_breakout, 'signal'] = -1  # 看空信号

            return data
        except Exception as e:
            logger.error(f"生成信号时出错: {e}")
            return data


class PriceBreakoutStrategy(MonitorStrategy):
    """价格突破策略"""

    def __init__(
            self,
            breakout_window: int = 20,
            volume_ratio: float = 2.0,
            price_change_threshold: float = 0.02
    ):
        super().__init__('PriceBreakout')
        self.breakout_window = breakout_window
        self.volume_ratio = volume_ratio
        self.price_change_threshold = price_change_threshold
        self.indicator_columns = ['high_window', 'low_window', 'price_change', 'volume_ma', 'volume_ratio']

    def _generate_signals_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成价格突破信号
        
        参数:
            data: 股票数据，包含OHLCV
            
        返回:
            带有信号的DataFrame
        """
        try:
            # 计算最高价和最低价的移动窗口
            data['high_window'] = data['high'].rolling(window=self.breakout_window).max()
            data['low_window'] = data['low'].rolling(window=self.breakout_window).min()

            # 计算价格变化百分比
            data['price_change'] = data['close'].pct_change()

            # 计算成交量比率
            data['volume_ma'] = data['volume'].rolling(window=self.breakout_window).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']

            # 生成信号
            data['signal'] = 0

            # 向上突破
            upward_breakout = (data['close'] > data['high_window'].shift(1)) & \
                              (data['price_change'] > self.price_change_threshold) & \
                              (data['volume_ratio'] > self.volume_ratio)

            # 向下突破
            downward_breakout = (data['close'] < data['low_window'].shift(1)) & \
                                (data['price_change'] < -self.price_change_threshold) & \
                                (data['volume_ratio'] > self.volume_ratio)

            data.loc[upward_breakout, 'signal'] = 1
            data.loc[downward_breakout, 'signal'] = -1

            return data
        except Exception as e:
            logger.error(f"生成信号时出错: {e}")
            return data


class StrategyBacktest:
    """策略回测类"""

    def __init__(
            self,
            data_manager,  # 使用新的数据管理器
            stock_manager,
            strategies=None
    ):
        """
        初始化回测系统
        
        参数:
            data_manager: 数据管理器
            stock_manager: 股票管理器
            strategies: 策略列表
        """
        self.data_manager = data_manager
        self.stock_manager = stock_manager
        
        # 如果没有提供策略列表，使用所有内置策略
        self.strategies = strategies or [
            GoldTriangleStrategy(),
            MomentumStrategy(),
            NiuniuStrategy(),
            TDIStrategy(),
            MarketForecastStrategy(),
            CPGWStrategy(),
            VolumeStrategy()
        ]

        # 回测结果
        self.results = {}

        logger.info(f"StrategyBacktest初始化完成，加载了 {len(self.strategies)} 个策略")

    def run_backtest(self) -> Dict:
        """运行回测"""
        try:
            # 获取监控的股票列表
            stocks = self.stock_manager.get_monitored_stocks()
            if stocks.empty:
                logger.warning("没有可回测的股票")
                return {}

            logger.info(f"开始回测，股票数量: {len(stocks)}, 策略数量: {len(self.strategies)}")

            # 对每个股票进行回测
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                try:
                    # 获取最新数据（包含历史数据和实时数据）
                    data = self.data_manager.get_latest_data(symbol)

                    if data is None or data.empty:
                        logger.warning(f"无法获取股票 {symbol} 的数据")
                        continue

                    # 运行每个策略
                    stock_results = {}
                    strategy_weights = {}
                    total_weight = 0
                    composite_signal = 0

                    for strategy in self.strategies:
                        try:
                            # 生成信号
                            signals = strategy.generate_signals(data.copy())

                            # 评估策略
                            evaluation = self._evaluate_strategy(signals, strategy.name)
                            stock_results[strategy.name] = evaluation

                            # 计算策略权重
                            weight = self._calculate_strategy_weight(evaluation)
                            strategy_weights[strategy.name] = weight
                            total_weight += weight

                            # 累加加权信号
                            if 'signal' in signals.columns:
                                last_signal = signals['signal'].iloc[-1]
                                composite_signal += last_signal * weight

                        except Exception as e:
                            logger.error(f"运行策略 {strategy.name} 于股票 {symbol} 时出错: {e}")

                    # 归一化策略权重
                    if total_weight > 0:
                        for strategy_name in strategy_weights:
                            strategy_weights[strategy_name] /= total_weight
                        composite_signal /= total_weight

                    # 保存结果
                    self.results[symbol] = {
                        'strategy_results': stock_results,
                        'strategy_weights': strategy_weights,
                        'composite_signal': composite_signal
                    }

                    # 记录信号
                    if abs(composite_signal) > 0.7:
                        signal_type = "强" + ("买入" if composite_signal > 0 else "卖出")
                        logger.info(f"{symbol} 产生{signal_type}信号: {composite_signal:.2f}")
                    elif abs(composite_signal) > 0.3:
                        signal_type = "中等" + ("买入" if composite_signal > 0 else "卖出")
                        logger.info(f"{symbol} 产生{signal_type}信号: {composite_signal:.2f}")

                except Exception as e:
                    logger.error(f"回测股票 {symbol} 时出错: {e}")

            logger.info("回测完成")
            return self.results

        except Exception as e:
            logger.error(f"运行回测时出错: {e}")
            return {}

    def _evaluate_strategy(self, data: pd.DataFrame, strategy_name: str) -> Dict:
        """评估策略表现"""
        try:
            # 确保数据中有必要的列
            required_columns = ['close', 'signal']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("数据缺少必要的列")

            # 计算策略收益
            data['strategy_returns'] = data['signal'].shift(1) * data['close'].pct_change()

            # 计算累积收益
            data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

            # 计算评估指标
            total_return = data['cumulative_returns'].iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(data)) - 1

            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = data['strategy_returns'] - risk_free_rate / 252
            epsilon = 1e-8
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() + epsilon

            # 计算最大回撤
            rolling_max = data['cumulative_returns'].expanding().max()
            drawdowns = data['cumulative_returns'] / rolling_max - 1
            max_drawdown = drawdowns.min()

            # 计算信号统计
            total_signals = len(data[data['signal'] != 0])
            long_signals = len(data[data['signal'] == 1])
            short_signals = len(data[data['signal'] == -1])

            return {
                "strategy_name": strategy_name,
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_signals": total_signals,
                "long_signals": long_signals,
                "short_signals": short_signals,
                "win_rate": self._calculate_win_rate(data)
            }

        except Exception as e:
            logger.error(f"评估策略 {strategy_name} 时出错: {e}")
            return {}

    def _calculate_win_rate(self, data: pd.DataFrame) -> float:
        """计算胜率"""
        try:
            # 获取有信号的交易
            trades = data[data['signal'] != 0]
            if len(trades) == 0:
                return 0.0

            # 计算每笔交易的收益
            winning_trades = len(trades[trades['strategy_returns'] > 0])
            return winning_trades / len(trades)

        except Exception as e:
            logger.error(f"计算胜率时出错: {e}")
            return 0.0

    def _calculate_strategy_weight(self, evaluation: Dict) -> float:
        """
        计算策略权重
        
        基于以下指标计算权重：
        1. Sharpe比率 (40%)
        2. 胜率 (30%)
        3. 最大回撤 (20%)
        4. 年化收益率 (10%)
        """
        try:
            sharpe_weight = 0.4 * max(0, min(1, evaluation.get('sharpe_ratio', 0) / 2))
            win_rate_weight = 0.3 * evaluation.get('win_rate', 0)
            drawdown_weight = 0.2 * (1 + min(0, evaluation.get('max_drawdown', 0)))
            return_weight = 0.1 * max(0, min(1, evaluation.get('annual_return', 0)))

            return sharpe_weight + win_rate_weight + drawdown_weight + return_weight

        except Exception as e:
            logger.error(f"计算策略权重时出错: {e}")
            return 0.0

    def generate_report(self) -> str:
        """生成回测报告"""
        try:
            if not self.results:
                logger.warning("没有回测结果可供生成报告")
                return ""

            # 创建报告文件名
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path("reports") / f"backtest_report_{timestamp}.json"

            # 计算每个策略的平均表现
            strategy_metrics = {}
            for strategy in self.strategies:
                strategy_name = strategy.name
                metrics = {
                    "total_return": [],
                    "annual_return": [],
                    "sharpe_ratio": [],
                    "max_drawdown": [],
                    "win_rate": []
                }

                # 收集所有股票的策略表现
                for stock_results in self.results.values():
                    if strategy_name in stock_results['strategy_results']:
                        result = stock_results['strategy_results'][strategy_name]
                        metrics["total_return"].append(result.get("total_return", 0))
                        metrics["annual_return"].append(result.get("annual_return", 0))
                        metrics["sharpe_ratio"].append(result.get("sharpe_ratio", 0))
                        metrics["max_drawdown"].append(result.get("max_drawdown", 0))
                        metrics["win_rate"].append(result.get("win_rate", 0))

                # 计算平均值
                strategy_metrics[strategy_name] = {
                    "avg_total_return": np.mean(metrics["total_return"]),
                    "avg_annual_return": np.mean(metrics["annual_return"]),
                    "avg_sharpe_ratio": np.mean(metrics["sharpe_ratio"]),
                    "avg_max_drawdown": np.mean(metrics["max_drawdown"]),
                    "avg_win_rate": np.mean(metrics["win_rate"])
                }

            # 生成最优参数建议
            optimal_thresholds = {
                "price_change": 0.02,  # 默认值
                "volume_ratio": 2.0,
                "volatility_ratio": 1.5,
                "signal_confirm": 2
            }

            # 根据回测结果调整参数
            best_strategy = max(
                strategy_metrics.items(),
                key=lambda x: x[1]["avg_sharpe_ratio"]
            )[0]

            if best_strategy == "VolatilityBreakout":
                optimal_thresholds["volatility_ratio"] = 2.0
            elif best_strategy == "PriceBreakout":
                optimal_thresholds["price_change"] = 0.03

            # 创建完整报告
            report = {
                "backtest_period": {
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "strategy_performance": strategy_metrics,
                "stock_results": self.results,
                "optimal_thresholds": optimal_thresholds,
                "best_strategy": best_strategy
            }

            # 保存报告
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)

            logger.info(f"回测报告已保存到: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"生成回测报告时出错: {e}")
            return ""

    def _analyze_market_condition(self) -> Dict:
        """分析市场状况"""
        try:
            # 统计所有股票的信号
            buy_signals = 0
            sell_signals = 0
            total_volatility = 0

            for stock_results in self.results.values():
                for strategy_result in stock_results['strategy_results'].values():
                    if 'signal' in strategy_result:
                        if strategy_result['signal'] > 0:
                            buy_signals += 1
                        elif strategy_result['signal'] < 0:
                            sell_signals += 1
                    if 'volatility' in strategy_result:
                        total_volatility += strategy_result['volatility']

            total_stocks = len(self.results)
            if total_stocks > 0:
                avg_volatility = total_volatility / total_stocks

                # 判断市场状况
                if buy_signals > sell_signals * 2:
                    market_condition = "强势上涨"
                    risk_level = "低"
                elif buy_signals > sell_signals:
                    market_condition = "温和上涨"
                    risk_level = "中"
                elif sell_signals > buy_signals * 2:
                    market_condition = "强势下跌"
                    risk_level = "高"
                elif sell_signals > buy_signals:
                    market_condition = "温和下跌"
                    risk_level = "中高"
                else:
                    market_condition = "盘整"
                    risk_level = "中"

                return {
                    "market_condition": market_condition,
                    "risk_level": risk_level,
                    "opportunity_sectors": self._identify_opportunity_sectors(),
                    "avg_volatility": avg_volatility
                }

            return {
                "market_condition": "未知",
                "risk_level": "未知",
                "opportunity_sectors": [],
                "avg_volatility": 0
            }

        except Exception as e:
            logger.error(f"分析市场状况时出错: {e}")
            return {
                "market_condition": "分析失败",
                "risk_level": "未知",
                "opportunity_sectors": [],
                "avg_volatility": 0
            }

    def _identify_opportunity_sectors(self) -> List[str]:
        """识别机会板块"""
        try:
            # 按板块统计表现
            sector_performance = {}

            for symbol, results in self.results.items():
                # 获取股票所属板块
                stock_info = self.stock_manager.get_stock_info(symbol)
                sector = stock_info.get('sector', 'Unknown')

                # 计算该股票的平均收益
                total_return = 0
                strategy_count = 0
                for strategy_name, strategy_result in results['strategy_results'].items():
                    if 'total_return' in strategy_result:
                        total_return += strategy_result['total_return']
                        strategy_count += 1

                if strategy_count > 0:
                    avg_return = total_return / strategy_count
                    if sector not in sector_performance:
                        sector_performance[sector] = []
                    sector_performance[sector].append(avg_return)

            # 计算每个板块的平均表现
            sector_avg_performance = {
                sector: sum(returns) / len(returns)
                for sector, returns in sector_performance.items()
            }

            # 选择表现最好的前3个板块
            top_sectors = sorted(
                sector_avg_performance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            return [sector for sector, _ in top_sectors]

        except Exception as e:
            logger.error(f"识别机会板块时出错: {e}")
            return []


class StockMonitor:
    def __init__(self, stock_list):
        self.stocks = stock_list
        self.volatility_strategy = VolatilityBreakoutStrategy()
        self.price_strategy = PriceBreakoutStrategy()

    def monitor_individual_stocks(self):
        for stock in self.stocks:
            # 分析个股异常
            volatility_signal = self.volatility_strategy.generate_signals(stock)
            price_signal = self.price_strategy.generate_signals(stock)

            if self.is_significant_event(volatility_signal, price_signal):
                self.generate_alert(stock)


class MarketMonitor:
    def __init__(self):
        self.market_indices = ['SPY', 'QQQ', 'IWM']  # 主要市场指数
        self.sector_etfs = ['XLF', 'XLK', 'XLE']  # 主要行业ETF

    def analyze_market_condition(self):
        # 分析市场整体状况
        market_volatility = self.calculate_market_volatility()
        sector_rotation = self.analyze_sector_rotation()
        market_breadth = self.calculate_market_breadth()

        return {
            'market_condition': 'bull/bear/neutral',
            'risk_level': 'high/medium/low',
            'opportunity_sectors': ['tech', 'finance']
        }
