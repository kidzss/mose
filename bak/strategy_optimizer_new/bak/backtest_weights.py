import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import logging
from datetime import datetime, timedelta

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.strategy_factory import StrategyFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_stock_data(symbol, start_date, end_date):
    """加载股票数据"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'))
        
        # 确保列名小写
        df.columns = [col.lower() for col in df.columns]
        
        if df.empty:
            logger.warning(f"股票 {symbol} 数据为空")
            return None
            
        return df
    except Exception as e:
        logger.error(f"加载股票 {symbol} 数据时出错: {e}")
        return None

def calculate_strategy_signals(df, strategy_factory):
    """计算各个策略的信号"""
    strategies = strategy_factory.create_all_strategies()
    signals = {}
    
    for name, strategy in strategies.items():
        try:
            strategy_df = df.copy()
            signal = strategy.generate_signals(strategy_df)
            
            # 确保信号是一维数组
            if signal is not None:
                if isinstance(signal, pd.DataFrame):
                    if 'signal' in signal.columns:
                        signal = signal['signal'].values
                    else:
                        signal = signal.iloc[:, 0].values
                elif isinstance(signal, pd.Series):
                    signal = signal.values
                    
                # 确保信号是浮点数类型并处理无效值
                signal = np.nan_to_num(signal.astype(float), nan=0.0)
                
                if len(signal) > 0:
                    signals[name] = signal
                    logger.info(f"策略 {name} 生成了 {len(signal)} 个信号")
                else:
                    logger.warning(f"策略 {name} 生成了空信号")
            else:
                logger.warning(f"策略 {name} 没有生成有效信号")
        except Exception as e:
            logger.error(f"策略 {name} 生成信号时出错: {str(e)}")
            continue
            
    return signals

def backtest_weighted_strategy(df, signals, weights, initial_capital=10000.0):
    """使用给定的权重回测综合策略"""
    # 确保我们有足够的数据
    if df.empty or not signals:
        return None
        
    # 提取所有策略的权重
    strategy_weights = {}
    for strategy_name in signals.keys():
        if strategy_name in weights:
            strategy_weights[strategy_name] = weights[strategy_name]
        else:
            logger.warning(f"策略 {strategy_name} 没有提供权重，将使用0")
            strategy_weights[strategy_name] = 0.0
            
    # 创建结果DataFrame
    results = pd.DataFrame(index=df.index)
    results['open'] = df['open']
    results['high'] = df['high']
    results['low'] = df['low']
    results['close'] = df['close']
    
    # 计算综合信号
    combined_signal = np.zeros(len(df))
    
    # 添加每个策略的信号
    for strategy_name, signal in signals.items():
        if len(signal) != len(df):
            logger.warning(f"策略 {strategy_name} 的信号长度与数据不匹配，将被跳过")
            continue
            
        weight = strategy_weights.get(strategy_name, 0.0)
        combined_signal += signal * weight
        
        # 保存单个策略的信号
        results[f'signal_{strategy_name}'] = signal
    
    # 归一化组合信号到 -1 到 1 的范围
    max_abs_signal = max(1.0, np.max(np.abs(combined_signal)))
    combined_signal = combined_signal / max_abs_signal
    
    # 保存组合信号
    results['combined_signal'] = combined_signal
    
    # 计算头寸
    results['position'] = combined_signal
    
    # 计算每日回报
    results['returns'] = df['close'].pct_change()
    
    # 计算策略回报
    results['strategy_returns'] = results['position'].shift(1) * results['returns']
    
    # 计算累积回报
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
    
    # 计算资金曲线
    results['equity'] = initial_capital * results['strategy_cumulative_returns']
    
    return results

def calculate_performance_metrics(results):
    """计算回测性能指标"""
    if results is None or results.empty:
        return {}
        
    # 获取策略回报
    strategy_returns = results['strategy_returns'].dropna()
    
    if len(strategy_returns) == 0:
        return {}
        
    # 计算年化回报
    annual_return = strategy_returns.mean() * 252
    
    # 计算波动率
    volatility = strategy_returns.std() * np.sqrt(252)
    
    # 计算夏普比率 (假设无风险利率为0)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 计算最大回撤
    cumulative = results['strategy_cumulative_returns']
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # 计算胜率
    win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
    
    # 返回性能指标
    return {
        'Total Return (%)': (cumulative.iloc[-1] - 1) * 100,
        'Annual Return (%)': annual_return * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Win Rate (%)': win_rate * 100
    }

def plot_backtest_results(results, metrics, title, output_file=None):
    """绘制回测结果"""
    if results is None or results.empty:
        logger.error("没有回测结果可供绘制")
        return
        
    plt.figure(figsize=(12, 8))
    
    # 绘制价格和策略表现
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(results.index, results['close'], 'b-', label='价格')
    ax1.set_ylabel('价格', color='b')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(results.index, results['strategy_cumulative_returns'], 'r-', label='策略收益')
    ax2.set_ylabel('累积收益', color='r')
    ax2.legend(loc='upper right')
    
    # 绘制仓位
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(results.index, results['position'], 'g-', label='仓位')
    ax3.set_ylabel('仓位')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper left')
    
    # 设置标题
    plt.suptitle(title, fontsize=16)
    
    # 添加性能指标
    metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
    ax1.text(0.01, 0.05, metrics_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300)
        logger.info(f"回测图表已保存至 {output_file}")
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='回测策略权重组合')
    parser.add_argument('--weights', type=str, required=True, help='策略权重CSV文件路径')
    parser.add_argument('--symbol', type=str, default='AAPL', help='要回测的股票代码')
    parser.add_argument('--days', type=int, default=365, help='回测天数')
    parser.add_argument('--output', type=str, default='backtest_results.png', help='输出图表文件路径')
    args = parser.parse_args()
    
    try:
        # 加载策略权重
        weights_df = pd.read_csv(args.weights)
        
        # 获取特定股票的权重
        symbol_weights = weights_df[weights_df['Symbol'] == args.symbol]
        if symbol_weights.empty:
            logger.error(f"在权重文件中找不到股票 {args.symbol} 的数据")
            return 1
            
        # 提取权重 (排除_raw后缀的列和Symbol列)
        weight_cols = [col for col in symbol_weights.columns 
                      if not col.endswith('_raw') and col != 'Symbol']
        weights = {col: symbol_weights[col].values[0] for col in weight_cols}
        
        logger.info(f"使用以下权重进行回测: {weights}")
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # 加载股票数据
        df = load_stock_data(args.symbol, start_date, end_date)
        if df is None or df.empty:
            logger.error(f"无法获取股票 {args.symbol} 的数据")
            return 1
            
        logger.info(f"加载了 {len(df)} 条股票数据")
        
        # 初始化策略工厂
        factory = StrategyFactory()
        
        # 计算各个策略的信号
        signals = calculate_strategy_signals(df, factory)
        
        # 进行回测
        results = backtest_weighted_strategy(df, signals, weights)
        
        # 计算性能指标
        metrics = calculate_performance_metrics(results)
        logger.info("回测性能指标:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.2f}")
            
        # 绘制回测结果
        plot_backtest_results(
            results, 
            metrics, 
            f"{args.symbol} 策略优化权重回测 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})",
            args.output
        )
        
        # 保存回测数据
        csv_output = args.output.replace('.png', '.csv')
        results.to_csv(csv_output)
        logger.info(f"回测数据已保存至 {csv_output}")
        
        # 为比较目的，使用均等权重进行回测
        equal_weights = {k: 1.0/len(signals) for k in signals.keys()}
        logger.info(f"使用均等权重进行对比回测: {equal_weights}")
        
        equal_results = backtest_weighted_strategy(df, signals, equal_weights)
        equal_metrics = calculate_performance_metrics(equal_results)
        
        logger.info("均等权重回测性能指标:")
        for k, v in equal_metrics.items():
            logger.info(f"{k}: {v:.2f}")
            
        # 绘制均等权重回测结果
        plot_backtest_results(
            equal_results, 
            equal_metrics, 
            f"{args.symbol} 均等权重策略回测 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})",
            args.output.replace('.png', '_equal.png')
        )
        
        # 绘制两种策略的比较图
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['strategy_cumulative_returns'], 'b-', 
                linewidth=2, label='优化权重')
        plt.plot(equal_results.index, equal_results['strategy_cumulative_returns'], 'r--', 
                linewidth=2, label='均等权重')
        plt.title(f"{args.symbol} 优化权重 vs 均等权重", fontsize=16)
        plt.xlabel('日期')
        plt.ylabel('累积收益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加性能对比
        compare_text = "优化权重 vs 均等权重\n"
        for k in metrics.keys():
            compare_text += f"{k}: {metrics[k]:.2f} vs {equal_metrics[k]:.2f}\n"
        plt.figtext(0.01, 0.01, compare_text, fontsize=9, 
                  bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        comparison_output = args.output.replace('.png', '_comparison.png')
        plt.savefig(comparison_output, dpi=300)
        logger.info(f"比较图表已保存至 {comparison_output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 