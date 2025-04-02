from strategy_optimizer.data_processors.data_processor import DataProcessor
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def calculate_returns(df: pd.DataFrame, signals: np.ndarray, stop_loss: np.ndarray, take_profit: np.ndarray) -> Tuple[List[Dict], pd.Series]:
    """计算每笔交易的收益和累计收益曲线"""
    position = 0  # 当前持仓状态
    entry_price = 0  # 入场价格
    entry_date = None  # 入场时间
    trades = []  # 交易记录
    daily_returns = pd.Series(0.0, index=df.index)  # 修改为float类型
    
    for i in range(len(signals)):
        current_date = df.index[i]
        current_price = df['close'].iloc[i]
        
        # 如果持有多头仓位
        if position == 1:
            # 计算当日收益率
            if i > 0:  # 确保有前一天的数据
                daily_returns.iloc[i] = (current_price - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            
            # 检查是否触发止损或止盈
            if current_price <= stop_loss[i-1] or current_price >= take_profit[i-1] or signals[i] == -1:
                # 记录交易
                trade = {
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'return': (current_price - entry_price) / entry_price * 100,  # 百分比收益
                    'holding_days': (current_date - entry_date).days,
                    'exit_reason': 'stop_loss' if current_price <= stop_loss[i-1] else 
                                 'take_profit' if current_price >= take_profit[i-1] else 
                                 'signal'
                }
                trades.append(trade)
                position = 0  # 平仓
        
        # 如果收到买入信号且当前没有持仓
        if signals[i] == 1 and position == 0:
            position = 1
            entry_price = current_price
            entry_date = current_date
    
    return trades, daily_returns

def calculate_performance_metrics(trades: List[Dict], daily_returns: pd.Series) -> Dict:
    """计算策略绩效指标"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    # 计算基础指标
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['return'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_return = np.mean([t['return'] for t in trades])
    avg_holding_days = np.mean([t['holding_days'] for t in trades])
    
    # 计算累计收益曲线
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # 计算最大回撤
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100  # 转换为百分比
    
    # 计算夏普比率 (假设无风险利率为0.02)
    risk_free_rate = 0.02
    excess_returns = daily_returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    # 计算每种平仓原因的统计
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'avg_return': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['avg_return'] += trade['return']
    
    for reason in exit_reasons:
        exit_reasons[reason]['avg_return'] /= exit_reasons[reason]['count']
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate * 100,  # 转换为百分比
        'avg_return': avg_return,
        'avg_holding_days': avg_holding_days,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_return': (cumulative_returns.iloc[-1] - 1) * 100,  # 转换为百分比
        'exit_reasons': exit_reasons
    }

def plot_results(df: pd.DataFrame, trades: List[Dict], daily_returns: pd.Series):
    """绘制回测结果图表"""
    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 绘制价格和交易点
    ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
    
    # 标记交易点
    for trade in trades:
        # 买入点
        ax1.scatter(trade['entry_date'], trade['entry_price'], 
                   color='g', marker='^', s=100, label='Buy' if trade == trades[0] else "")
        # 卖出点
        ax1.scatter(trade['exit_date'], trade['exit_price'],
                   color='r', marker='v', s=100, label='Sell' if trade == trades[0] else "")
    
    ax1.set_title('Price Movement and Trading Points')
    ax1.legend()
    
    # 绘制累计收益曲线
    cumulative_returns = (1 + daily_returns).cumprod()
    ax2.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    ax2.set_title('Cumulative Returns')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()

def test_strategy():
    # 初始化数据处理器
    dp = DataProcessor()
    
    # 获取测试数据 - 扩展到1年
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-03-31'
    
    print("1. 获取股票数据...")
    df = dp.get_stock_data(symbol, start_date, end_date)
    
    print("\n2. 基本数据统计:")
    print(f"数据点数量: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"可用特征: {df.columns.tolist()}")
    
    # 修改止损设置为25%
    df['trailing_stop'] = df['close'] * 0.75  # 25%的固定止损
    df['profit_target'] = df['close'] * 1.25  # 对称设置25%的止盈
    
    print("\n3. 生成交易信号...")
    signals = dp._calculate_momentum_signal(df)
    
    print("\n4. 信号分布:")
    unique_signals, counts = np.unique(signals, return_counts=True)
    for signal, count in zip(unique_signals, counts):
        print(f"信号 {signal}: {count} 次")
    
    print("\n5. 交易详情:")
    trades, daily_returns = calculate_returns(df, signals, df['trailing_stop'].values, df['profit_target'].values)
    
    for trade in trades:
        print(f"\n入场时间: {trade['entry_date'].strftime('%Y-%m-%d')}")
        print(f"入场价格: {trade['entry_price']:.2f}")
        print(f"出场时间: {trade['exit_date'].strftime('%Y-%m-%d')}")
        print(f"出场价格: {trade['exit_price']:.2f}")
        print(f"收益率: {trade['return']:.2f}%")
        print(f"持仓天数: {trade['holding_days']}")
        print(f"出场原因: {trade['exit_reason']}")
        print("---")
    
    print("\n6. 策略绩效:")
    metrics = calculate_performance_metrics(trades, daily_returns)
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate']:.2f}%")
    print(f"平均收益: {metrics['avg_return']:.2f}%")
    print(f"平均持仓天数: {metrics['avg_holding_days']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"总收益: {metrics['total_return']:.2f}%")
    
    print("\n出场原因统计:")
    for reason, stats in metrics['exit_reasons'].items():
        print(f"{reason}: {stats['count']} 次, 平均收益 {stats['avg_return']:.2f}%")
    
    # 绘制结果
    plot_results(df, trades, daily_returns)
    print("\n回测结果图表已保存为 'backtest_results.png'")

if __name__ == "__main__":
    test_strategy() 