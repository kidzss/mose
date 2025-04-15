import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy.intraday_momentum_strategy import IntradayMomentumStrategy
from strategy.breakout_strategy import BreakoutStrategy
from backtest.strategy_evaluator import StrategyEvaluator
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(symbol, start_date, end_date):
    """
    加载5分钟数据
    """
    try:
        # 使用Yahoo Finance API获取数据
        import yfinance as yf
        
        # 确保时间范围在60天以内
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if (end - start).days > 60:
            start = end - timedelta(days=60)
            logger.warning(f"调整开始日期到: {start.strftime('%Y-%m-%d')}")
        
        # 下载数据
        data = yf.download(symbol, start=start, end=end, interval='5m')
        
        # 确保数据完整性
        if data.empty:
            logger.error("未获取到数据")
            return None
            
        # 重命名列
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return data
        
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        return None

def run_backtest(strategy, data):
    """
    运行回测
    """
    try:
        if data is None or data.empty:
            logger.error("数据为空，无法进行回测")
            return None
            
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 计算指标
        df = strategy.calculate_indicators(df)
        
        # 生成信号
        signals = strategy.generate_signals(df)
        df['signal'] = signals
        
        # 计算收益
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = signals.shift(1) * df['returns']
        
        # 计算累积收益
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        return df
        
    except Exception as e:
        logger.error(f"回测时出错: {str(e)}")
        return None

def plot_results(data, strategy_name):
    """
    绘制回测结果
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # 绘制价格和信号
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['close'], label='Price')
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal')
        plt.title(f'{strategy_name} - Price and Signals')
        plt.legend()
        
        # 绘制累积收益
        plt.subplot(2, 1, 2)
        plt.plot(data.index, data['cumulative_returns'], label='Buy and Hold')
        plt.plot(data.index, data['strategy_cumulative_returns'], label='Strategy')
        plt.title('Cumulative Returns')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'backtest_results_{strategy_name}.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制结果时出错: {str(e)}")

def calculate_metrics(data):
    """
    计算策略性能指标
    """
    try:
        # 计算年化收益率
        total_days = (data.index[-1] - data.index[0]).days
        annual_return = (data['strategy_cumulative_returns'].iloc[-1] ** (365/total_days) - 1) * 100
        
        # 计算夏普比率
        daily_returns = data['strategy_returns'].fillna(0)
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # 计算最大回撤
        cum_returns = data['strategy_cumulative_returns']
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # 计算胜率
        winning_trades = (data['strategy_returns'] > 0).sum()
        total_trades = (data['signal'] != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        metrics = {
            'Annual Return (%)': round(annual_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Total Trades': total_trades
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算指标时出错: {str(e)}")
        return None

def main():
    # 设置参数
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 获取最近30天的数据
    
    # 加载数据
    data = load_data(symbol, start_date, end_date)
    if data is None:
        logger.error("无法加载数据")
        return
        
    # 初始化策略
    momentum_strategy = IntradayMomentumStrategy()
    breakout_strategy = BreakoutStrategy()
    
    # 运行回测
    momentum_results = run_backtest(momentum_strategy, data.copy())
    breakout_results = run_backtest(breakout_strategy, data.copy())
    
    if momentum_results is not None:
        # 计算性能指标
        momentum_metrics = calculate_metrics(momentum_results)
        logger.info(f"日内动量策略性能指标:")
        for metric, value in momentum_metrics.items():
            logger.info(f"{metric}: {value}")
        
        # 绘制结果
        plot_results(momentum_results, 'Intraday_Momentum')
    
    if breakout_results is not None:
        # 计算性能指标
        breakout_metrics = calculate_metrics(breakout_results)
        logger.info(f"突破策略性能指标:")
        for metric, value in breakout_metrics.items():
            logger.info(f"{metric}: {value}")
        
        # 绘制结果
        plot_results(breakout_results, 'Breakout')

if __name__ == "__main__":
    main() 