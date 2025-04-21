import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Dict, Optional, Union, Tuple, Any
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
from strategy.strategy_base import Strategy

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


class StrategyBacktest(Strategy):
    """
    策略回测类，用于回测交易策略
    
    Args:
        data: 历史数据
        config: 回测配置
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        super().__init__("StrategyBacktest", config)
        self.data = data
        self.config = config
        self.strategies = []
        self.returns = pd.Series()
        
    def add_strategy(self, strategy: Strategy) -> None:
        """
        添加策略
        
        Args:
            strategy: 交易策略
        """
        if strategy not in self.strategies:
            self.strategies.append(strategy)
            logger.info(f"Added strategy: {strategy.name}")
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 历史数据
            
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        # 计算移动平均线
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # 计算相对强弱指标
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma_20'] + 2 * df['std_20']
        df['lower_band'] = df['ma_20'] - 2 * df['std_20']
        
        return df
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含技术指标的数据
            
        Returns:
            pd.DataFrame: 包含交易信号的数据
        """
        # 初始化信号列
        df['signal'] = 0
        
        # 生成移动平均线交叉信号
        df.loc[df['ma_5'] > df['ma_20'], 'signal'] = 1
        df.loc[df['ma_5'] < df['ma_20'], 'signal'] = -1
        
        # RSI 超买超卖信号
        df.loc[df['rsi'] > 70, 'signal'] = -1
        df.loc[df['rsi'] < 30, 'signal'] = 1
        
        # 布林带突破信号
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1
        
        return df
        
    def run_backtest(self) -> Dict[str, Any]:
        """
        运行回测
        
        Returns:
            Dict[str, Any]: 回测结果
        """
        results = {}
        
        # 计算技术指标
        self.data = self.calculate_indicators(self.data)
        
        # 生成交易信号
        self.data = self.generate_signals(self.data)
        
        # 计算收益率
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        self.data['returns'] = self.data['position'] * self.data['close'].pct_change()
        self.returns = self.data['returns']
        
        # 计算回测指标
        results['total_return'] = self.calculate_total_return()
        results['annual_return'] = self.calculate_annual_return()
        results['sharpe_ratio'] = self.calculate_sharpe_ratio()
        results['max_drawdown'] = self.calculate_max_drawdown()
        
        return results
        
    def calculate_total_return(self) -> float:
        """
        计算总收益率
        
        Returns:
            float: 总收益率
        """
        return (1 + self.returns).prod() - 1
        
    def calculate_annual_return(self) -> float:
        """
        计算年化收益率
        
        Returns:
            float: 年化收益率
        """
        days = (self.data.index[-1] - self.data.index[0]).days
        return (1 + self.calculate_total_return()) ** (365 / days) - 1
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        Args:
            risk_free_rate: 无风险利率
            
        Returns:
            float: 夏普比率
        """
        excess_returns = self.returns - risk_free_rate / 252
        if self.returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / self.returns.std()
        return 0
        
    def calculate_max_drawdown(self) -> float:
        """
        计算最大回撤
        
        Returns:
            float: 最大回撤
        """
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()

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
