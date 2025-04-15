import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from monitor.market_monitor import MarketMonitor
from utils.data_loader import DataLoader

def load_config(config_path):
    """加载回测配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_backtest():
    """执行回测"""
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'monitor', 'configs', 'backtest_config.json')
    config = load_config(config_path)
    params = config['backtest_params']
    
    # 初始化数据加载器
    data_loader = DataLoader()
    
    # 初始化市场监控器
    monitor = MarketMonitor()
    
    # 获取历史数据
    start_date = datetime.strptime(params['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(params['end_date'], '%Y-%m-%d')
    stock_data = {}
    
    for symbol in params['stock_pool']:
        try:
            data = data_loader.get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                stock_data[symbol] = data
        except Exception as e:
            print(f"获取{symbol}数据时出错: {str(e)}")
    
    if not stock_data:
        print("没有获取到任何股票数据")
        return
    
    # 初始化回测参数
    capital = params['initial_capital']
    positions = {}
    equity_curve = []
    drawdown_curve = []
    trades = []
    
    # 按日期遍历数据
    dates = sorted(list(stock_data.values())[0].index)
    for date in dates:
        try:
            # 获取当日市场数据
            market_data = {symbol: data.loc[date] for symbol, data in stock_data.items()}
            
            # 分析市场环境
            market_state = monitor.analyze_market_environment(market_data)
            
            # 生成交易信号
            signals = monitor._generate_signals(market_data)
            
            # 执行交易
            for symbol, signal in signals.items():
                if symbol not in positions:
                    positions[symbol] = 0
                
                current_price = market_data[symbol]['close']
                position_value = positions[symbol] * current_price
                
                # 根据信号调整仓位
                if signal == 'strong_buy' and position_value < capital * params['position_limits']['max_single_stock_exposure']:
                    # 买入
                    shares_to_buy = int((capital * 0.1) / current_price)
                    cost = shares_to_buy * current_price * (1 + params['transaction_cost'] + params['slippage'])
                    if cost <= capital:
                        positions[symbol] += shares_to_buy
                        capital -= cost
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'buy',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
                
                elif signal == 'strong_sell' and positions[symbol] > 0:
                    # 卖出
                    shares_to_sell = positions[symbol]
                    revenue = shares_to_sell * current_price * (1 - params['transaction_cost'] - params['slippage'])
                    positions[symbol] = 0
                    capital += revenue
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'revenue': revenue
                    })
            
            # 计算当日资产
            total_assets = capital
            for symbol, shares in positions.items():
                total_assets += shares * market_data[symbol]['close']
            
            # 记录权益曲线
            equity_curve.append({
                'date': date,
                'equity': total_assets,
                'capital': capital
            })
            
            # 计算回撤
            if equity_curve:
                peak = max([e['equity'] for e in equity_curve])
                drawdown = (peak - total_assets) / peak
                drawdown_curve.append({
                    'date': date,
                    'drawdown': drawdown
                })
            
        except Exception as e:
            print(f"处理{date}数据时出错: {str(e)}")
    
    # 计算回测指标
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('date', inplace=True)
    
    # 计算总收益率
    total_return = (equity_df['equity'].iloc[-1] - params['initial_capital']) / params['initial_capital']
    
    # 计算年化收益率
    days = (end_date - start_date).days
    annual_return = (1 + total_return) ** (365 / days) - 1
    
    # 计算夏普比率
    daily_returns = equity_df['equity'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * (252 ** 0.5))
    
    # 计算最大回撤
    max_drawdown = pd.DataFrame(drawdown_curve)['drawdown'].max()
    
    # 计算胜率
    winning_trades = len([t for t in trades if t['action'] == 'sell' and t['revenue'] > t['shares'] * t['price']])
    total_trades = len([t for t in trades if t['action'] == 'sell'])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 输出回测结果
    print("\n回测结果:")
    print(f"初始资金: {params['initial_capital']:,.2f}")
    print(f"最终资金: {equity_df['equity'].iloc[-1]:,.2f}")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"胜率: {win_rate:.2%}")
    print(f"总交易次数: {len(trades)}")
    
    # 绘制权益曲线和回撤曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(equity_df.index, equity_df['equity'])
    plt.title('权益曲线')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(pd.DataFrame(drawdown_curve).set_index('date')['drawdown'])
    plt.title('回撤曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/backtest_results.png')
    plt.close()

if __name__ == "__main__":
    run_backtest()
