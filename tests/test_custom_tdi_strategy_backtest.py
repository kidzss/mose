import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy.custom_tdi_strategy import CustomTDIStrategy
from data.data_interface import DataInterface
from config.data_config import default_data_config
import logging

class TestCustomTDIStrategyBacktest(unittest.TestCase):
    def setUp(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据接口
        self.data_interface = DataInterface(
            default_source='mysql',
            config={'mysql': default_data_config.get_mysql_dict()}
        )
        
        # 初始化策略
        self.strategy = CustomTDIStrategy()
        
        # 设置回测参数
        self.start_date = datetime(2022, 1, 1)  # 延长回测期间
        self.end_date = datetime(2024, 1, 1)
        self.initial_capital = 100000.0
        self.position_size = 0.1  # 每次交易使用10%的资金
        self.transaction_cost = 0.001  # 0.1%交易成本
        
    def test_backtest_performance(self):
        """测试策略回测表现"""
        self.logger.info("开始回测...")
        
        # 获取回测数据
        df = self.data_interface.get_historical_data(
            symbol='GOOGL',
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if df.empty:
            self.logger.warning("未获取到数据，请检查数据源配置")
            return
            
        # 打印数据格式信息
        self.logger.info(f"数据列名: {df.columns.tolist()}")
        self.logger.info(f"数据前5行:\n{df.head()}")
        
        # 生成信号
        final_signals = self.strategy.generate_signals(df)
        
        # 计算策略收益
        strategy_returns = pd.Series(0.0, index=df.index)
        position = 0
        
        for i in range(1, len(df)):
            if final_signals.iloc[i] == 1 and position <= 0:  # 买入信号
                position = self.position_size
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position - self.transaction_cost
            elif final_signals.iloc[i] == -1 and position >= 0:  # 卖出信号
                position = -self.position_size
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position - self.transaction_cost
            else:  # 持仓不变
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position
                
        # 计算基准收益
        benchmark_returns = df['close'].pct_change()
        
        # 计算年化收益率
        annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = strategy_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # 计算最大回撤
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 计算交易统计
        trades = strategy_returns[strategy_returns != 0]
        total_trades = len(trades)
        winning_trades = len(trades[trades > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均盈亏
        avg_win = trades[trades > 0].mean() if len(trades[trades > 0]) > 0 else 0
        avg_loss = trades[trades < 0].mean() if len(trades[trades < 0]) > 0 else 0
        
        # 计算盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 计算总交易成本
        total_cost = total_trades * self.transaction_cost
        
        # 输出回测结果
        self.logger.info(f"回测期间: {self.start_date.date()} 到 {self.end_date.date()}")
        self.logger.info(f"策略年化收益率: {annual_return:.2%}")
        self.logger.info(f"基准年化收益率: {benchmark_annual_return:.2%}")
        self.logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        self.logger.info(f"最大回撤: {max_drawdown:.2%}")
        self.logger.info(f"总交易次数: {total_trades}")
        self.logger.info(f"胜率: {win_rate:.2%}")
        self.logger.info(f"平均盈利: {avg_win:.2%}")
        self.logger.info(f"平均亏损: {avg_loss:.2%}")
        self.logger.info(f"盈亏比: {profit_factor:.2f}")
        self.logger.info(f"总交易成本: {total_cost:.2%}")
        
        # 输出月度收益分析
        monthly_returns = strategy_returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        self.logger.info("\n月度收益分析:")
        self.logger.info(monthly_returns.describe())
        
        # 验证性能指标
        self.assertGreater(annual_return, 0.0, "策略收益率应该为正")
        self.assertGreater(sharpe_ratio, 0.0, "夏普比率应该为正")
        self.assertGreater(win_rate, 0.4, "胜率应该大于40%")
        self.assertGreater(max_drawdown, -0.2, "最大回撤应该大于-20%")
        
    def test_risk_metrics(self):
        """测试风险指标"""
        self.logger.info("计算风险指标...")
        
        # 获取回测数据
        df = self.data_interface.get_historical_data(
            symbol='GOOGL',
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if df.empty:
            self.logger.warning("未获取到数据，请检查数据源配置")
            return
            
        # 打印数据格式信息
        self.logger.info(f"数据列名: {df.columns.tolist()}")
        self.logger.info(f"数据前5行:\n{df.head()}")
        
        # 生成信号
        final_signals = self.strategy.generate_signals(df)
        
        # 计算策略收益
        strategy_returns = pd.Series(0.0, index=df.index)
        position = 0
        
        for i in range(1, len(df)):
            if final_signals.iloc[i] == 1 and position <= 0:  # 买入信号
                position = self.position_size
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position - self.transaction_cost
            elif final_signals.iloc[i] == -1 and position >= 0:  # 卖出信号
                position = -self.position_size
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position - self.transaction_cost
            else:  # 持仓不变
                strategy_returns.iloc[i] = df['close'].pct_change().iloc[i] * position
                
        # 计算年化波动率
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        
        # 计算95% VaR
        var_95 = np.percentile(strategy_returns, 5)
        
        # 计算最大连续亏损天数
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in strategy_returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
                
        # 计算下行波动率
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        # 计算最大回撤持续时间
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        underwater = cumulative_returns < rolling_max
        underwater_periods = underwater.astype(int).groupby(underwater.astype(int).diff().ne(0).cumsum()).sum()
        max_drawdown_duration = underwater_periods.max()
        
        # 输出风险指标
        self.logger.info(f"年化波动率: {annual_volatility:.2%}")
        self.logger.info(f"95% VaR: {var_95:.2%}")
        self.logger.info(f"最大连续亏损天数: {max_consecutive_losses}")
        self.logger.info(f"下行波动率: {downside_volatility:.2%}")
        self.logger.info(f"最大回撤持续时间: {max_drawdown_duration}天")
        
        # 验证风险指标
        self.assertLess(annual_volatility, 0.15, "年化波动率应该小于15%")
        self.assertGreater(var_95, -0.02, "95% VaR应该大于-2%")
        self.assertLess(max_consecutive_losses, 10, "最大连续亏损天数应该小于10天")
        self.assertLess(max_drawdown_duration, 120, "最大回撤持续时间应该小于120天") 