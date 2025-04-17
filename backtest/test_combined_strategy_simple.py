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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(days: int = 365) -> pd.DataFrame:
    """
    生成测试数据，包括预先计算的技术指标
    """
    try:
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+100)  # 额外添加100天数据，用于计算初始的移动平均线
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
        
        # 预先计算技术指标
        # 计算常用的移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # 计算MACD
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # 计算ADX
        # +DM, -DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff.abs()), 0)
        minus_dm = low_diff.abs().where((low_diff > 0) & (low_diff > high_diff), 0)
        
        # TR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算14日平均
        smoothed_plus_dm = plus_dm.rolling(window=14).sum()
        smoothed_minus_dm = minus_dm.rolling(window=14).sum()
        smoothed_tr = tr.rolling(window=14).sum()
        
        # +DI, -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
        # ADX
        df['ADX'] = dx.rolling(window=14).mean()
        
        # 计算波动率
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # 添加牛牛策略需要的指标
        df['fast_ma'] = df['close'].rolling(window=5).mean()
        df['middle_ma'] = df['close'].rolling(window=10).mean()
        df['slow_ma'] = df['close'].rolling(window=20).mean()
        
        # 计算市场预测策略需要的指标
        med_len = 31
        near_len = 3
        
        df['lowest_low_med'] = df['low'].rolling(window=med_len).min()
        df['highest_high_med'] = df['high'].rolling(window=med_len).max()
        df['fastK_I'] = (df['close'] - df['lowest_low_med']) / (df['highest_high_med'] - df['lowest_low_med']).replace(0, np.nan) * 100

        df['lowest_low_near'] = df['low'].rolling(window=near_len).min()
        df['highest_high_near'] = df['high'].rolling(window=near_len).max()
        df['fastK_N'] = (df['close'] - df['lowest_low_near']) / (df['highest_high_near'] - df['lowest_low_near']).replace(0, np.nan) * 100

        min1 = df['low'].rolling(window=4).min()
        max1 = df['high'].rolling(window=4).max()
        df['momentum'] = ((df['close'] - min1) / (max1 - min1).replace(0, np.nan)) * 100

        # 计算牛熊聚类
        df['bull_cluster'] = ((df['momentum'] <= 20) & (df['fastK_I'] <= 20) & (df['fastK_N'] <= 20)).astype(int)
        df['bear_cluster'] = ((df['momentum'] >= 80) & (df['fastK_I'] >= 80) & (df['fastK_N'] >= 80)).astype(int)
        
        # 添加趋势强度指标
        df['trend_strength'] = 0.0  # 默认值
        # 简单的趋势强度计算：基于移动平均线的相对位置
        for i in range(len(df)):
            if i >= 20:
                # 当所有均线都向上排列时，趋势强度为正
                if df['SMA_5'].iloc[i] > df['SMA_10'].iloc[i] > df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i]:
                    df.loc[df.index[i], 'trend_strength'] = 0.8
                # 当所有均线都向下排列时，趋势强度为负
                elif df['SMA_5'].iloc[i] < df['SMA_10'].iloc[i] < df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]:
                    df.loc[df.index[i], 'trend_strength'] = -0.8
                # 其他情况，根据短期均线和长期均线的差距计算趋势强度
                else:
                    strength = (df['SMA_5'].iloc[i] - df['SMA_50'].iloc[i]) / df['SMA_50'].iloc[i]
                    df.loc[df.index[i], 'trend_strength'] = np.clip(strength * 5, -1, 1)
        
        # 删除前100天的数据（这些数据只用于计算初始的技术指标）
        df = df.iloc[100:]
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        return df
        
    except Exception as e:
        logger.error(f"生成测试数据时出错: {str(e)}")
        raise

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
    main() 