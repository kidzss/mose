import pandas as pd
import numpy as np
from typing import Dict, Any

class StrategyEvaluator:
    """策略评估器，用于计算策略的各项表现指标"""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        初始化评估器
        
        Args:
            results_df: 包含回测结果的DataFrame，必须包含'return'列
        """
        self.results = results_df
        
    def evaluate(self) -> Dict[str, float]:
        """
        评估策略表现，计算各项指标
        
        Returns:
            包含各项指标的字典
        """
        returns = self.results['return'].fillna(0)
        
        # 计算累积收益
        cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
        
        # 计算年化收益
        n_years = len(returns) / 252  # 假设一年252个交易日
        annual_return = (1 + cumulative_return) ** (1/n_years) - 1
        
        # 计算波动率
        daily_std = returns.std()
        annual_volatility = daily_std * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # 计算最大回撤
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 计算胜率
        win_rate = (returns > 0).mean()
        
        # 计算盈亏比
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
        
        # 计算信息比率
        benchmark_return = 0  # 这里可以替换为实际的基准收益
        tracking_error = (returns - benchmark_return).std() * np.sqrt(252)
        information_ratio = (annual_return - benchmark_return) / tracking_error if tracking_error != 0 else 0
        
        return {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'information_ratio': information_ratio
        }
        
    def plot_equity_curve(self) -> None:
        """绘制权益曲线"""
        import matplotlib.pyplot as plt
        
        returns = self.results['return'].fillna(0)
        equity_curve = (1 + returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title('Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.show()
        
    def plot_drawdown(self) -> None:
        """绘制回撤曲线"""
        import matplotlib.pyplot as plt
        
        returns = self.results['return'].fillna(0)
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns.index, drawdowns.values)
        plt.title('Strategy Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()
        
    def generate_report(self) -> str:
        """
        生成策略评估报告
        
        Returns:
            包含评估结果的字符串报告
        """
        metrics = self.evaluate()
        
        report = "策略评估报告\n"
        report += "=" * 50 + "\n\n"
        
        report += "收益指标:\n"
        report += f"累积收益率: {metrics['cumulative_return']:.2%}\n"
        report += f"年化收益率: {metrics['annual_return']:.2%}\n"
        report += f"年化波动率: {metrics['annual_volatility']:.2%}\n"
        report += f"夏普比率: {metrics['sharpe_ratio']:.2f}\n"
        report += f"信息比率: {metrics['information_ratio']:.2f}\n\n"
        
        report += "风险指标:\n"
        report += f"最大回撤: {metrics['max_drawdown']:.2%}\n"
        report += f"胜率: {metrics['win_rate']:.2%}\n"
        report += f"盈亏比: {metrics['profit_loss_ratio']:.2f}\n"
        
        return report 