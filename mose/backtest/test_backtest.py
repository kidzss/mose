import logging
from mose.backtest.backtest_engine import BacktestEngine
import matplotlib.pyplot as plt

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        config_path='../monitor/configs/portfolio_config.json'
    )
    
    # 持仓股票
    portfolio_symbols = ['GOOG', 'TSLA', 'AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    
    # 运行回测
    results = engine.run_backtest(
        start_date='2023-01-01',  # 从2023年开始，这样可以有足够的历史数据
        end_date='2024-04-13',    # 到当前日期
        symbols=portfolio_symbols,
        initial_capital=1000000.0,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # 绘制结果
    engine.plot_results(save_path='backtest_results.png')
    
    # 生成报告
    report = engine.generate_report(save_path='backtest_report.txt')
    print(report)
    
    # 保存结果到文件
    import json
    with open('backtest_results.json', 'w') as f:
        json.dump({
            'initial_capital': results['initial_capital'],
            'final_capital': results['results']['equity_curve'][-1][1],
            'total_return': results['metrics']['total_return'],
            'annual_return': results['metrics']['annual_return'],
            'max_drawdown': results['metrics']['max_drawdown'],
            'sharpe_ratio': results['metrics']['sharpe_ratio'],
            'win_rate': results['metrics']['win_rate'],
            'total_trades': results['metrics']['total_trades'],
            'winning_trades': results['metrics']['win_rate'] * results['metrics']['total_trades']
        }, f, indent=4)
    
    # 保存交易记录
    import pandas as pd
    trades_df = pd.DataFrame(results['trades'])
    trades_df.to_csv('backtest_trades.csv', index=False)
    
    # 保存回测结果
    results_df = pd.DataFrame(results['results']['equity_curve'], columns=['date', 'portfolio_value'])
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df.set_index('date', inplace=True)
    results_df.to_csv('backtest_equity_curve.csv')
    
    # 保存每日收益率
    daily_returns_df = pd.DataFrame(results['results']['daily_returns'], columns=['date', 'daily_return'])
    daily_returns_df['date'] = pd.to_datetime(daily_returns_df['date'])
    daily_returns_df.set_index('date', inplace=True)
    daily_returns_df.to_csv('backtest_daily_returns.csv')

if __name__ == '__main__':
    main() 