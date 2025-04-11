import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import (
    Strategy,
    GoldenCrossStrategy,
    BollingerBandsStrategy,
    MACDStrategy,
    RSIStrategy,
    CustomStrategy
)

class StrategyBacktest:
    """策略回测类"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 initial_capital: float = 100000.0,
                 position_size: float = 0.1,  # 每次交易的资金比例
                 commission: float = 0.001,  # 佣金比例
                 slippage: float = 0.001,  # 滑点比例
                 risk_free_rate: float = 0.03):  # 无风险利率
        """
        初始化回测环境
        
        参数:
            data: OHLCV数据，日期索引
            initial_capital: 初始资金
            position_size: 每次交易使用的资金比例
            commission: 佣金比例
            slippage: 滑点比例
            risk_free_rate: 年化无风险利率
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # 确保数据具有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        # 确保索引是日期类型
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是日期类型")
    
    def run_backtest(self, strategy: Strategy) -> pd.DataFrame:
        """
        使用给定策略运行回测
        
        参数:
            strategy: 交易策略实例
            
        返回:
            回测结果DataFrame，包含交易记录和绩效指标
        """
        print(f"运行 {strategy.name} 策略的回测...")
        
        # 获取策略信号
        signal_data = strategy.generate_signals(self.data)
        
        # 初始化回测结果
        result = signal_data.copy()
        result['position'] = 0  # 持仓状态
        result['entry_price'] = 0.0  # 入场价格
        result['exit_price'] = 0.0  # 出场价格
        result['shares'] = 0.0  # 持有股数
        result['trade_profit'] = 0.0  # 单次交易盈亏
        result['trade_return'] = 0.0  # 单次交易收益率
        result['equity'] = self.initial_capital  # 账户价值
        result['cash'] = self.initial_capital  # 现金
        result['drawdown'] = 0.0  # 回撤
        
        # 记录当前的持仓状态
        current_position = 0  # 0: 空仓, 1: 持多, -1: 持空
        entry_price = 0.0
        entry_date = None
        current_shares = 0.0
        current_cash = self.initial_capital
        current_equity = self.initial_capital
        max_equity = self.initial_capital
        
        # 遍历每个交易日
        for i in range(1, len(result)):
            date = result.index[i]
            signal = result['signal'].iloc[i]
            
            # 如果有买入信号且当前无持仓
            if signal == 1 and current_position == 0:
                # 计算买入价格（考虑滑点）
                entry_price = result['open'].iloc[i] * (1 + self.slippage)
                entry_date = date
                
                # 计算可以买入的股数
                trade_value = current_cash * self.position_size
                current_shares = trade_value / entry_price
                
                # 考虑佣金
                commission_cost = trade_value * self.commission
                current_cash -= (trade_value + commission_cost)
                
                # 更新持仓状态
                current_position = 1
                
                # 记录到结果中
                result['position'].iloc[i] = current_position
                result['entry_price'].iloc[i] = entry_price
                result['shares'].iloc[i] = current_shares
            
            # 如果有卖出信号且当前持有多头仓位
            elif signal == -1 and current_position == 1:
                # 计算卖出价格（考虑滑点）
                exit_price = result['open'].iloc[i] * (1 - self.slippage)
                
                # 计算交易盈亏
                trade_value = current_shares * exit_price
                commission_cost = trade_value * self.commission
                trade_profit = trade_value - (current_shares * entry_price) - commission_cost
                trade_return = trade_profit / (current_shares * entry_price)
                
                # 更新现金
                current_cash += trade_value - commission_cost
                
                # 记录到结果中
                result['exit_price'].iloc[i] = exit_price
                result['trade_profit'].iloc[i] = trade_profit
                result['trade_return'].iloc[i] = trade_return
                
                # 重置持仓状态
                current_position = 0
                current_shares = 0
                entry_price = 0
                
                # 更新持仓状态
                result['position'].iloc[i] = current_position
            
            # 计算当前的账户价值
            if current_position == 1:
                # 如果持有仓位，考虑未实现的收益
                current_equity = current_cash + (current_shares * result['close'].iloc[i])
            else:
                # 如果空仓，价值等于现金
                current_equity = current_cash
            
            # 更新最大账户价值和回撤
            max_equity = max(max_equity, current_equity)
            current_drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
            
            # 记录到结果中
            result['equity'].iloc[i] = current_equity
            result['cash'].iloc[i] = current_cash
            result['drawdown'].iloc[i] = current_drawdown
        
        # 计算每日回报率
        result['daily_return'] = result['equity'].pct_change()
        
        # 计算累积回报率
        result['cumulative_return'] = (1 + result['daily_return']).cumprod() - 1
        
        # 计算绩效指标
        metrics = self.calculate_performance_metrics(result)
        
        # 添加绩效摘要
        result.attrs['metrics'] = metrics
        
        return result
    
    def calculate_performance_metrics(self, result: pd.DataFrame) -> Dict[str, float]:
        """
        计算绩效指标
        
        参数:
            result: 包含回测结果的DataFrame
            
        返回:
            包含各种绩效指标的字典
        """
        # 过滤出包含交易的日期
        trades = result[result['trade_profit'] != 0]
        
        # 总收益率
        total_return = (result['equity'].iloc[-1] / self.initial_capital) - 1
        
        # 年化收益率
        days = (result.index[-1] - result.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 最大回撤
        max_drawdown = result['drawdown'].max()
        
        # 交易次数
        num_trades = len(trades)
        
        # 胜率
        if num_trades > 0:
            win_rate = len(trades[trades['trade_profit'] > 0]) / num_trades
        else:
            win_rate = 0
        
        # 盈亏比
        if len(trades[trades['trade_profit'] < 0]) > 0 and len(trades[trades['trade_profit'] > 0]) > 0:
            avg_win = trades[trades['trade_profit'] > 0]['trade_profit'].mean()
            avg_loss = abs(trades[trades['trade_profit'] < 0]['trade_profit'].mean())
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_loss_ratio = 0
        
        # 夏普比率
        daily_returns = result['daily_return'].dropna()
        if len(daily_returns) > 0:
            # 将年化无风险利率转换为日度
            daily_risk_free = (1 + self.risk_free_rate) ** (1 / 252) - 1
            excess_returns = daily_returns - daily_risk_free
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 索提诺比率
        if len(daily_returns) > 0:
            # 只考虑负收益的波动率
            downside_returns = daily_returns[daily_returns < daily_risk_free] - daily_risk_free
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = np.sqrt(252) * (daily_returns.mean() - daily_risk_free) / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        # 卡玛比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # 返回所有指标
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def plot_backtest_results(self, result: pd.DataFrame, strategy_name: str, output_dir: str = 'tests/output'):
        """
        绘制回测结果
        
        参数:
            result: 包含回测结果的DataFrame
            strategy_name: 策略名称
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取性能指标
        metrics = result.attrs['metrics']
        
        # 1. 绘制权益曲线
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(result.index, result['equity'], label='账户价值')
        
        # 标记买入和卖出点
        entries = result[result['entry_price'] > 0]
        exits = result[result['exit_price'] > 0]
        
        plt.scatter(entries.index, entries['equity'], marker='^', color='g', s=100, label='买入')
        plt.scatter(exits.index, exits['equity'], marker='v', color='r', s=100, label='卖出')
        
        plt.title(f'{strategy_name} - 账户价值曲线')
        plt.grid(True)
        plt.legend()
        
        # 2. 绘制回撤
        plt.subplot(2, 1, 2)
        plt.fill_between(result.index, -result['drawdown'] * 100, 0, color='red', alpha=0.3)
        plt.title('回撤百分比')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/{strategy_name}_equity_curve.png')
        plt.close()
        
        # 3. 绘制价格和交易点
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        plt.plot(result.index, result['close'], label='收盘价')
        
        # 标记买入和卖出点
        plt.scatter(entries.index, result.loc[entries.index, 'close'], marker='^', color='g', s=100, label='买入')
        plt.scatter(exits.index, result.loc[exits.index, 'close'], marker='v', color='r', s=100, label='卖出')
        
        plt.title(f'{strategy_name} - 价格和交易点')
        plt.grid(True)
        plt.legend()
        
        # 4. 绘制累积收益率
        plt.subplot(3, 1, 2)
        plt.plot(result.index, result['cumulative_return'] * 100, label='累积收益率(%)')
        plt.title('累积收益率')
        plt.grid(True)
        plt.legend()
        
        # 5. 绘制每日收益率
        plt.subplot(3, 1, 3)
        plt.bar(result.index, result['daily_return'] * 100, label='每日收益率(%)', alpha=0.5)
        plt.title('每日收益率')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/{strategy_name}_trading_performance.png')
        plt.close()
        
        # 打印性能指标
        print(f"\n{strategy_name} 性能指标:")
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annualized_return']:.2%}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"交易次数: {metrics['num_trades']}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {metrics['sortino_ratio']:.2f}")
        print(f"卡玛比率: {metrics['calmar_ratio']:.2f}")


def generate_sample_data(days=500):
    """
    生成模拟测试数据
    
    参数:
        days: 天数
        
    返回:
        DataFrame，包含OHLCV数据
    """
    # 创建模拟的价格数据
    np.random.seed(42)  # 固定随机种子以获得可重复的结果
    
    # 创建日期范围
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # 生成随机价格
    close_prices = np.random.randn(days).cumsum() + 100  # 起始价格100
    
    # 为了模拟趋势和波动性，我们添加一些模式
    # 上升趋势
    close_prices[50:150] += np.linspace(0, 20, 100)
    # 下降趋势
    close_prices[200:300] -= np.linspace(0, 15, 100)
    # 盘整期
    close_prices[350:450] += np.sin(np.linspace(0, 10 * np.pi, 100)) * 5
    
    # 生成开高低价格
    high_prices = close_prices + np.random.rand(days) * 3
    low_prices = close_prices - np.random.rand(days) * 3
    open_prices = low_prices + np.random.rand(days) * (high_prices - low_prices)
    
    # 生成成交量数据
    volume = np.random.randint(1000, 10000, days)
    # 在价格大幅波动的区域增加成交量
    volume[50:70] *= 2  # 上升趋势开始时成交量放大
    volume[145:155] *= 3  # 趋势转换点成交量放大
    volume[195:205] *= 2  # 下降趋势开始时成交量放大
    volume[295:305] *= 3  # 下降趋势结束时成交量放大
    
    # 创建OHLCV DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # 设置日期为索引
    df.set_index('date', inplace=True)
    
    return df


def compare_strategies(strategies: List[Strategy], data: pd.DataFrame, output_dir: str = 'tests/output'):
    """
    比较多个策略的性能
    
    参数:
        strategies: 策略列表
        data: 回测数据
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化回测环境
    backtest = StrategyBacktest(data)
    
    # 运行每个策略的回测并存储结果
    results = {}
    for strategy in strategies:
        result = backtest.run_backtest(strategy)
        results[strategy.name] = result
        backtest.plot_backtest_results(result, strategy.name, output_dir)
    
    # 比较各策略的累积收益率
    plt.figure(figsize=(14, 8))
    for name, result in results.items():
        plt.plot(result.index, result['cumulative_return'] * 100, label=f"{name} ({result.attrs['metrics']['total_return']:.2%})")
    
    plt.title('策略累积收益率比较')
    plt.xlabel('日期')
    plt.ylabel('累积收益率(%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/strategy_comparison.png')
    plt.close()
    
    # 比较各策略的性能指标
    metrics_summary = pd.DataFrame({
        name: {
            '总收益率': f"{result.attrs['metrics']['total_return']:.2%}",
            '年化收益率': f"{result.attrs['metrics']['annualized_return']:.2%}",
            '最大回撤': f"{result.attrs['metrics']['max_drawdown']:.2%}",
            '交易次数': result.attrs['metrics']['num_trades'],
            '胜率': f"{result.attrs['metrics']['win_rate']:.2%}",
            '盈亏比': f"{result.attrs['metrics']['profit_loss_ratio']:.2f}",
            '夏普比率': f"{result.attrs['metrics']['sharpe_ratio']:.2f}",
            '索提诺比率': f"{result.attrs['metrics']['sortino_ratio']:.2f}",
            '卡玛比率': f"{result.attrs['metrics']['calmar_ratio']:.2f}"
        } for name, result in results.items()
    })
    
    # 保存性能指标对比到CSV文件
    metrics_summary.to_csv(f'{output_dir}/strategy_metrics_comparison.csv')
    
    print("\n策略性能对比:")
    print(metrics_summary)
    
    # 返回比较的结果
    return results, metrics_summary


def run_all_strategy_backtests():
    """运行所有策略的回测"""
    # 生成样本数据
    data = generate_sample_data(days=500)
    
    # 创建策略实例
    strategies = [
        GoldenCrossStrategy(),
        BollingerBandsStrategy(),
        MACDStrategy(),
        RSIStrategy(),
        CustomStrategy()
    ]
    
    # 比较策略性能
    results, metrics_summary = compare_strategies(strategies, data)


if __name__ == '__main__':
    run_all_strategy_backtests() 