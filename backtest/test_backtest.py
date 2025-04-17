import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sys
import os

# 添加项目根目录到路径，确保能导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入策略和回测引擎
from strategy.cpgw_strategy import CPGWStrategy
# 使用相对导入
from backtest_engine import BacktestEngine

def get_market_data(symbol, start_date, end_date):
    """从本地数据库获取市场数据"""
    # 连接数据库
    engine = create_engine('mysql+pymysql://root@localhost/mose')
    
    # 查询数据
    query = f"""
    SELECT 
        Date as date,
        Open as open,
        High as high, 
        Low as low,
        Close as close,
        Volume as volume
    FROM stock_time_code
    WHERE Code = '{symbol}'
    AND Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Date ASC
    """
    
    print(f"正在获取数据: {symbol}, 从 {start_date} 到 {end_date}")
    df = pd.read_sql_query(query, engine)
    print(f"获取到 {len(df)} 条数据")
    
    if df.empty:
        raise ValueError(f"未找到股票 {symbol} 的数据")
    
    # 设置日期为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 打印数据样例
    print("\n数据样例:")
    print(df.head())
    
    return df

def run_backtest(params, market_data=None):
    """运行单次回测"""
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2025-04-11')
    initial_capital = 1_000_000
    
    # 如果没有传入市场数据，则获取
    if market_data is None:
        market_data = get_market_data('SPY', start_date, end_date)
        if market_data.empty:
            return None
    
    # 创建策略实例
    strategy = CPGWStrategy(
        lookback_period=params['lookback_period'],
        overbought=params['overbought'],
        oversold=params['oversold'],
        fast_ma=params['fast_ma'],
        slow_ma=params['slow_ma'],
        use_market_regime=True  # 启用市场环境分析
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.001,
        position_size=params['position_size'],
        max_position=params['max_position'],
        cooldown_period=params['cooldown_period'],
        stop_loss=params['stop_loss'],
        take_profit=params['take_profit']
    )
    
    # 运行回测
    results = engine.run_backtest(market_data, strategy)
    
    return results

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置策略参数
    params = {
        'lookback_period': 14,  # RSI周期
        'overbought': 65,       # 降低超买阈值
        'oversold': 35,         # 提高超卖阈值
        'fast_ma': 8,           # 增加快线周期
        'slow_ma': 21,          # 增加慢线周期
        'position_size': 0.1,   # 基础仓位大小，实际仓位会根据市场环境动态调整
        'max_position': 0.8,    # 增加最大持仓比例
        'cooldown_period': 2,   # 减少交易冷却期
        'stop_loss': 0.03,      # 基础止损比例，实际止损会根据市场环境动态调整
        'take_profit': 0.1      # 基础止盈比例，实际止盈会根据市场环境动态调整
    }
    
    # 运行回测
    results = run_backtest(params)
    
    if results:
        # 提取回测结果
        metrics = results['metrics']
        total_return = metrics['total_return']
        sharpe_ratio = metrics['sharpe_ratio']
        max_drawdown = metrics['max_drawdown']
        win_rate = metrics['win_rate']
        total_trades = metrics['total_trades']
        
        # 输出结果
        print(f"总收益率: {total_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"胜率: {win_rate:.2%}")
        print(f"总交易次数: {total_trades}")
        
        # 绘制权益曲线
        equity_curve = results['equity']
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title('CPGWStrategy - 权益曲线')
        plt.xlabel('日期')
        plt.ylabel('资产价值')
        plt.grid(True)
        plt.savefig('cpgw_strategy_equity.png')
        plt.close()

if __name__ == '__main__':
    main() 