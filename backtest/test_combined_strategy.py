import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# 添加项目根目录到路径，确保能导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.combined_strategy import CombinedStrategy
from backtest.backtest_engine import BacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(days: int = 365) -> pd.DataFrame:
    """
    生成测试数据
    """
    try:
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 模拟价格数据
        np.random.seed(42)  # 固定随机种子以便结果可复现
        
        # 生成价格序列（包含趋势、周期性和随机性）
        trend = np.linspace(0, 0.5, len(date_range))  # 上升趋势
        cycle = 0.1 * np.sin(np.linspace(0, 15, len(date_range)))  # 周期性波动
        noise = 0.05 * np.random.randn(len(date_range))  # 随机噪声
        
        # 基础价格序列
        price_series = 100 * (1 + trend + cycle + noise)
        
        # 计算OHLCV数据
        closes = price_series
        opens = closes * (1 + 0.01 * np.random.randn(len(date_range)))
        highs = np.maximum(opens, closes) * (1 + 0.02 * np.abs(np.random.randn(len(date_range))))
        lows = np.minimum(opens, closes) * (1 - 0.02 * np.abs(np.random.randn(len(date_range))))
        volumes = 1000000 * (1 + 0.5 * np.random.rand(len(date_range)))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=date_range)
        
        # 计算收益率
        df['returns'] = df['close'].pct_change()
        
        return df
        
    except Exception as e:
        logger.error(f"生成测试数据时出错: {str(e)}")
        raise

def run_backtest(data: pd.DataFrame, strategy: CombinedStrategy, initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    运行回测
    """
    try:
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
        axes[0].plot(results['equity_curve'].index, results['equity_curve'].values)
        axes[0].set_title('权益曲线')
        axes[0].set_ylabel('资金')
        axes[0].grid(True)
        
        # 绘制回撤曲线
        axes[1].fill_between(results['drawdown'].index, 0, results['drawdown'].values * 100, color='red', alpha=0.5)
        axes[1].set_title('回撤曲线')
        axes[1].set_ylabel('回撤 (%)')
        axes[1].set_xlabel('日期')
        axes[1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"绘制结果时出错: {str(e)}")
        raise

def test_combined_strategy():
    """测试组合策略"""
    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', end='2024-04-01', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(100, 5, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    })
    
    # 创建策略实例
    strategy = CombinedStrategy()
    
    # 计算指标
    data = strategy.calculate_indicators(data)
    
    # 生成信号
    data = strategy.generate_signals(data)
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.001,
        slippage_rate=0.001
    )
    
    # 运行回测
    results = engine.run_backtest(data, strategy)
    
    # 打印结果
    print("\n回测结果:")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annualized_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"胜率: {results['win_rate']:.2%}")
    
    # 打印市场情绪分析结果
    print("\n市场情绪分析:")
    print(f"当前市场情绪: {data['market_sentiment'].iloc[-1]}")
    print(f"VIX指数: {data['vix'].iloc[-1]:.2f}")
    print(f"PCR比率: {data['pcr'].iloc[-1]:.2f}")
    
    # 打印策略权重
    print("\n策略权重:")
    weights = strategy._adjust_weights(
        market_regime='bullish',
        volatility_regime='low',
        sentiment='bullish'
    )
    for name, weight in weights.items():
        print(f"{name}: {weight:.2%}")

def main():
    """
    主函数
    """
    try:
        # 生成测试数据
        logger.info("生成测试数据...")
        data = generate_test_data(days=365)  # 生成一年的测试数据
        logger.info(f"测试数据生成完成，共 {len(data)} 条记录")
        
        # 创建策略实例
        strategy = CombinedStrategy()
        
        # 运行回测
        logger.info("开始回测...")
        results = run_backtest(data, strategy)
        
        # 输出回测结果
        logger.info("\n=== 回测结果 ===")
        logger.info(f"初始资金: ${results['initial_capital']:,.2f}")
        logger.info(f"最终资金: ${results['final_capital']:,.2f}")
        logger.info(f"总收益率: {results['total_return']*100:.2f}%")
        logger.info(f"年化收益率: {results['annual_return']*100:.2f}%")
        logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
        logger.info(f"最大回撤: {results['max_drawdown']*100:.2f}%")
        logger.info(f"总交易次数: {results['total_trades']}")
        
        # 绘制结果
        plot_results(results, 'combined_strategy_results.png')
        logger.info("结果已保存到 combined_strategy_results.png")
        
    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    test_combined_strategy() 