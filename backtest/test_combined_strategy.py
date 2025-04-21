"""
组合策略回测测试
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from strategy.combined_strategy import CombinedStrategy
from backtest.backtest_engine import BacktestEngine
from backtest.data_loader import DataLoader
from config.data_config import default_data_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_backtest(data: pd.DataFrame, strategy: CombinedStrategy, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    运行回测
    """
    try:
        # 准备额外的字段
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        
        # 生成信号
        data_with_signals = strategy.generate_signals(data)
        
        # 初始化回测结果
        positions = pd.Series(index=data.index, data=0.0)
        returns = pd.Series(index=data.index, data=0.0)
        equity = pd.Series(index=data.index, data=initial_capital)
        
        # 模拟交易
        for i in range(1, len(data)):
            # 获取信号
            signal = data_with_signals['signal'].iloc[i]
            
            # 更新持仓
            positions.iloc[i] = signal
            
            # 计算收益
            price_return = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
            returns.iloc[i] = positions.iloc[i-1] * price_return
            
            # 更新权益
            equity.iloc[i] = equity.iloc[i-1] * (1 + returns.iloc[i])
        
        # 计算回撤
        drawdown = (equity / equity.cummax() - 1)
        
        # 计算回测指标
        total_return = (equity.iloc[-1] / initial_capital - 1)
        annual_return = ((1 + total_return) ** (252 / len(data)) - 1) if len(data) > 0 else 0
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        max_drawdown = drawdown.min()
        
        # 计算交易统计
        total_trades = (positions.diff() != 0).sum()
        
        return {
            'initial_capital': initial_capital,
            'final_capital': equity.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'equity_curve': equity,
            'drawdown': drawdown,
            'positions': positions,
            'returns': returns
        }
        
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        raise

def plot_results(results: Dict[str, Any], save_path: str = None):
    """
    绘制回测结果
    """
    try:
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 绘制权益曲线
        axes[0].plot(results['equity_curve'], label='Equity')
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制回撤
        axes[1].fill_between(results['drawdown'].index, results['drawdown'], 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True)
        
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

def run_test():
    """
    运行回测测试
    """
    try:
        # 获取数据库配置
        db_config = default_data_config.get_mysql_dict()
        
        # 初始化数据加载器
        data_loader = DataLoader(db_config)
        
        # 设置回测参数
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # 使用持有的股票作为测试标的
        start_date = datetime(2024, 1, 1)  # 使用2024年的历史数据
        end_date = datetime(2025, 4, 16)   # 到现在
        initial_capital = 100000.0
        
        # 加载数据
        all_data = {}
        for symbol in symbols:
            data = data_loader.load_stock_data(symbol, start_date, end_date)
            if not data.empty:
                logger.info(f"成功加载 {symbol} 的数据")
                logger.info(f"数据列名: {data.columns.tolist()}")  # 打印列名
                all_data[symbol] = data
            else:
                logger.warning(f"无法加载 {symbol} 的数据")
        
        if not all_data:
            logger.error("没有成功加载任何股票数据")
            return
            
        # 创建策略实例
        strategy = CombinedStrategy()
        
        # 对每只股票运行回测
        all_results = {}
        for symbol, data in all_data.items():
            logger.info(f"开始回测 {symbol}")
            results = run_backtest(data, strategy, initial_capital/len(all_data))  # 平均分配初始资金
            all_results[symbol] = results
            
            # 打印单只股票的回测结果
            logger.info(f"{symbol} 回测结果:")
            logger.info(f"初始资金: ${results['initial_capital']:,.2f}")
            logger.info(f"最终资金: ${results['final_capital']:,.2f}")
            logger.info(f"总收益率: {results['total_return']*100:.2f}%")
            logger.info(f"年化收益率: {results['annual_return']*100:.2f}%")
            logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
            logger.info(f"最大回撤: {results['max_drawdown']*100:.2f}%")
            logger.info(f"总交易次数: {results['total_trades']}")
            
            # 绘制单只股票的结果
            plot_results(results, f'backtest_results_{symbol}.png')
        
        # 计算组合整体表现
        total_initial = initial_capital
        total_final = sum(results['final_capital'] for results in all_results.values())
        total_return = (total_final / total_initial - 1)
        
        # 计算组合年化收益率
        days = (end_date - start_date).days
        annual_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0
        
        # 打印组合整体表现
        logger.info("\n组合整体表现:")
        logger.info(f"初始总资金: ${total_initial:,.2f}")
        logger.info(f"最终总资金: ${total_final:,.2f}")
        logger.info(f"总收益率: {total_return*100:.2f}%")
        logger.info(f"年化收益率: {annual_return*100:.2f}%")
        
    except Exception as e:
        logger.error(f"运行测试时出错: {str(e)}")
        raise

if __name__ == '__main__':
    run_test() 