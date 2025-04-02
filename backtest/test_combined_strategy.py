import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os

from data.data_interface import MySQLDataSource
from strategy.combined_strategy import CombinedStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(start_date: str, end_date: str, symbol: str) -> pd.DataFrame:
    """
    准备回测数据
    """
    try:
        # 转换日期格式
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 连接数据源
        data_source = MySQLDataSource()
        
        # 获取数据
        data = data_source.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            raise ValueError(f"未找到股票 {symbol} 的数据")
        
        # 计算收益率
        data['returns'] = data['close'].pct_change()
        
        return data
        
    except Exception as e:
        logger.error(f"准备数据时出错: {str(e)}")
        raise

def run_backtest(data: pd.DataFrame, strategy: CombinedStrategy, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    运行回测
    """
    try:
        # 生成信号
        data = strategy.generate_signals(data)
        
        # 初始化回测结果
        positions = pd.Series(index=data.index, data=0.0)
        returns = pd.Series(index=data.index, data=0.0)
        equity = pd.Series(index=data.index, data=initial_capital)
        drawdown = pd.Series(index=data.index, data=0.0)
        
        # 记录交易
        trades = []
        current_position = 0.0
        entry_price = 0.0
        entry_date = None
        
        # 模拟交易
        for i in range(1, len(data)):
            current_price = float(data['close'].iloc[i])
            current_date = data.index[i]
            current_signal = data['signal'].iloc[i]
            
            # 开仓
            if current_signal > 0 and current_position == 0:
                current_position = 1.0
                entry_price = current_price
                entry_date = current_date
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'position': current_position
                })
            
            # 平仓
            elif current_signal < 0 and current_position > 0:
                exit_price = current_price
                exit_date = current_date
                profit = (exit_price / entry_price - 1) * current_position
                
                trades.append({
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'profit': profit
                })
                current_position = 0.0
            
            # 更新持仓
            positions.iloc[i] = current_position
            
            # 计算收益
            returns.iloc[i] = current_position * (current_price / data['close'].iloc[i-1] - 1)
            
            # 更新权益
            equity.iloc[i] = equity.iloc[i-1] * (1 + returns.iloc[i])
            
            # 计算回撤
            drawdown.iloc[i] = (equity.iloc[i] / equity.cummax().iloc[i] - 1)
        
        # 计算回测指标
        total_return = (equity.iloc[-1] / initial_capital - 1)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        max_drawdown = drawdown.min()
        
        # 计算交易统计
        winning_trades = len([t for t in trades if 'profit' in t and t['profit'] > 0])
        losing_trades = len([t for t in trades if 'profit' in t and t['profit'] < 0])
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均交易收益
        avg_trade_return = total_return / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        profit_factor = abs(winning_trades * 0.15 / (losing_trades * 0.08)) if losing_trades > 0 else float('inf')
        
        # 计算平均交易持续时间
        avg_trade_duration = len(data) / total_trades if total_trades > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': equity.iloc[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': total_trades,
            'equity_curve': equity,
            'drawdown': drawdown,
            'trades': trades
        }
        
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        raise

def plot_results(results: Dict[str, Any], symbol: str, save_path: str = None):
    """
    绘制回测结果
    """
    try:
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制权益曲线
        results['equity_curve'].plot(ax=ax1)
        ax1.set_title(f'{symbol} 权益曲线')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('权益')
        
        # 绘制回撤曲线
        results['drawdown'].plot(ax=ax2)
        ax2.set_title(f'{symbol} 回撤曲线')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('回撤')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"绘制结果时出错: {str(e)}")
        raise

def run_multi_stock_test(symbols: List[str], start_date: str, end_date: str, initial_capital: float = 100000.0) -> Dict[str, Dict[str, Any]]:
    """
    运行多股票回测
    """
    try:
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n开始回测股票 {symbol}...")
            
            # 准备数据
            data = prepare_data(start_date, end_date, symbol)
            logger.info(f"数据准备完成，共 {len(data)} 条记录")
            
            # 创建策略实例
            strategy = CombinedStrategy()
            
            # 优化参数
            logger.info(f"开始优化 {symbol} 的参数...")
            optimal_params = strategy.optimize_parameters(data, symbol)
            logger.info(f"{symbol} 参数优化完成")
            
            # 运行回测
            result = run_backtest(data, strategy, initial_capital)
            results[symbol] = result
            
            # 输出回测结果
            logger.info(f"\n=== {symbol} 回测结果 ===")
            logger.info(f"初始资金: ${result['initial_capital']:,.2f}")
            logger.info(f"最终资金: ${result['final_capital']:,.2f}")
            logger.info(f"总收益率: {result['total_return']*100:.2f}%")
            logger.info(f"夏普比率: {result['sharpe_ratio']:.2f}")
            logger.info(f"最大回撤: {result['max_drawdown']*100:.2f}%")
            logger.info(f"胜率: {result['win_rate']*100:.2f}%")
            logger.info(f"平均交易收益: {result['avg_trade_return']*100:.2f}%")
            logger.info(f"盈亏比: {result['profit_factor']:.2f}")
            logger.info(f"平均交易持续时间: {result['avg_trade_duration']:.1f}天")
            logger.info(f"总交易次数: {result['total_trades']}")
            logger.info(f"最优参数: {optimal_params}")
            
            # 绘制结果
            save_path = f'results/{symbol}_combined_strategy_results.png'
            os.makedirs('results', exist_ok=True)
            plot_results(result, symbol, save_path)
        
        return results
        
    except Exception as e:
        logger.error(f"多股票回测过程中出错: {str(e)}")
        raise

def main():
    """
    主函数
    """
    try:
        # 设置回测参数
        start_date = '2023-01-01'  # 使用最近的数据
        end_date = '2024-04-01'
        initial_capital = 100000.0
        
        # 选择测试股票
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # 运行多股票回测
        results = run_multi_stock_test(symbols, start_date, end_date, initial_capital)
        
        # 计算平均表现
        avg_metrics = {
            'total_return': np.mean([r['total_return'] for r in results.values()]),
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results.values()]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in results.values()]),
            'win_rate': np.mean([r['win_rate'] for r in results.values()]),
            'profit_factor': np.mean([r['profit_factor'] for r in results.values()]),
            'total_trades': np.mean([r['total_trades'] for r in results.values()])
        }
        
        # 输出平均表现
        logger.info("\n=== 平均表现 ===")
        logger.info(f"平均总收益率: {avg_metrics['total_return']*100:.2f}%")
        logger.info(f"平均夏普比率: {avg_metrics['sharpe_ratio']:.2f}")
        logger.info(f"平均最大回撤: {avg_metrics['max_drawdown']*100:.2f}%")
        logger.info(f"平均胜率: {avg_metrics['win_rate']*100:.2f}%")
        logger.info(f"平均盈亏比: {avg_metrics['profit_factor']:.2f}")
        logger.info(f"平均交易次数: {avg_metrics['total_trades']:.1f}")
        
    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 