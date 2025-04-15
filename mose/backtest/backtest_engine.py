import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from mose.monitor.market_monitor import MarketMonitor
from mose.strategy.combined_strategy import CombinedStrategy

class BacktestEngine:
    """
    回测引擎类，用于执行策略回测并生成分析报告
    """
    def __init__(self, config_path: str = None):
        """
        初始化回测引擎
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.market_monitor = MarketMonitor(config_path)
        self.strategy = CombinedStrategy()
        self.results = {}
        self.trades = []
        
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str],
        initial_capital: float = 1000000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005
    ) -> Dict:
        """
        运行回测
        
        参数:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            symbols: 股票代码列表
            initial_capital: 初始资金
            transaction_cost: 交易成本（百分比）
            slippage: 滑点（百分比）
            
        返回:
            Dict: 回测结果
        """
        try:
            # 初始化回测参数
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            self.positions = {symbol: 0 for symbol in symbols}
            self.transaction_cost = transaction_cost
            self.slippage = slippage
            
            # 获取历史数据
            historical_data = self._get_historical_data(symbols, start_date, end_date)
            if not historical_data:
                raise ValueError("无法获取历史数据")
                
            # 初始化回测结果
            self.results = {
                'equity_curve': [],
                'drawdown': [],
                'trades': [],
                'daily_returns': [],
                'positions': [],
                'market_states': []
            }
            
            # 执行回测
            dates = sorted(historical_data[symbols[0]].index)
            for date in dates:
                # 获取当日数据
                daily_data = {symbol: historical_data[symbol].loc[date] for symbol in symbols}
                
                # 分析市场状态
                market_state = self.market_monitor._analyze_market_state(daily_data)
                
                # 生成交易信号
                signals = self.strategy.generate_signals(daily_data, market_state)
                
                # 执行交易
                self._execute_trades(signals, daily_data, date)
                
                # 更新持仓价值
                portfolio_value = self._calculate_portfolio_value(daily_data)
                
                # 记录结果
                self._record_results(date, portfolio_value, market_state)
                
            # 计算回测指标
            metrics = self._calculate_metrics()
            
            return {
                'metrics': metrics,
                'results': self.results,
                'trades': self.trades
            }
            
        except Exception as e:
            self.logger.error(f"回测过程中发生错误: {str(e)}")
            raise
            
    def _get_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        获取历史数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            Dict[str, pd.DataFrame]: 历史数据字典
        """
        try:
            # 这里应该调用数据获取模块获取历史数据
            # 暂时使用模拟数据
            data = {}
            for symbol in symbols:
                dates = pd.date_range(start=start_date, end=end_date)
                prices = np.random.normal(100, 10, len(dates))
                volumes = np.random.normal(1000000, 100000, len(dates))
                data[symbol] = pd.DataFrame({
                    'open': prices,
                    'high': prices * 1.02,
                    'low': prices * 0.98,
                    'close': prices,
                    'volume': volumes
                }, index=dates)
            return data
            
        except Exception as e:
            self.logger.error(f"获取历史数据时出错: {str(e)}")
            return {}
            
    def _execute_trades(
        self,
        signals: Dict[str, float],
        daily_data: Dict[str, pd.Series],
        date: datetime
    ) -> None:
        """
        执行交易
        
        参数:
            signals: 交易信号字典
            daily_data: 当日数据
            date: 交易日期
        """
        for symbol, signal in signals.items():
            if symbol not in daily_data:
                continue
                
            price = daily_data[symbol]['close']
            current_position = self.positions[symbol]
            
            # 计算目标仓位
            target_position = int(self.current_capital * signal / price)
            
            # 计算需要交易的股数
            shares_to_trade = target_position - current_position
            
            if shares_to_trade != 0:
                # 计算交易成本
                trade_cost = abs(shares_to_trade * price * self.transaction_cost)
                # 计算滑点成本
                slippage_cost = abs(shares_to_trade * price * self.slippage)
                
                # 更新资金
                self.current_capital -= (shares_to_trade * price + trade_cost + slippage_cost)
                
                # 更新持仓
                self.positions[symbol] = target_position
                
                # 记录交易
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'shares': shares_to_trade,
                    'price': price,
                    'cost': trade_cost + slippage_cost,
                    'signal': signal
                })
                
    def _calculate_portfolio_value(self, daily_data: Dict[str, pd.Series]) -> float:
        """
        计算投资组合价值
        
        参数:
            daily_data: 当日数据
            
        返回:
            float: 投资组合价值
        """
        portfolio_value = self.current_capital
        for symbol, position in self.positions.items():
            if symbol in daily_data:
                portfolio_value += position * daily_data[symbol]['close']
        return portfolio_value
        
    def _record_results(
        self,
        date: datetime,
        portfolio_value: float,
        market_state: Dict
    ) -> None:
        """
        记录回测结果
        
        参数:
            date: 日期
            portfolio_value: 投资组合价值
            market_state: 市场状态
        """
        self.results['equity_curve'].append((date, portfolio_value))
        self.results['market_states'].append((date, market_state))
        
        # 计算当日收益率
        if len(self.results['equity_curve']) > 1:
            prev_value = self.results['equity_curve'][-2][1]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.results['daily_returns'].append((date, daily_return))
            
        # 计算回撤
        max_value = max([v for _, v in self.results['equity_curve']])
        drawdown = (max_value - portfolio_value) / max_value
        self.results['drawdown'].append((date, drawdown))
        
    def _calculate_metrics(self) -> Dict:
        """
        计算回测指标
        
        返回:
            Dict: 回测指标
        """
        # 提取数据
        equity_curve = pd.Series(
            [v for _, v in self.results['equity_curve']],
            index=[d for d, _ in self.results['equity_curve']]
        )
        daily_returns = pd.Series(
            [r for _, r in self.results['daily_returns']],
            index=[d for d, _ in self.results['daily_returns']]
        )
        drawdown = pd.Series(
            [d for _, d in self.results['drawdown']],
            index=[d for d, _ in self.results['drawdown']]
        )
        
        # 计算指标
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        max_drawdown = drawdown.max()
        
        # 计算交易统计
        trades = pd.DataFrame(self.trades)
        if not trades.empty:
            win_rate = (trades['shares'] * trades['price'] > 0).mean()
            avg_trade_return = trades['shares'].mean() * trades['price'].mean() / self.initial_capital
            profit_factor = abs(trades[trades['shares'] * trades['price'] > 0]['shares'].sum() * 
                              trades[trades['shares'] * trades['price'] > 0]['price'].mean()) / \
                           abs(trades[trades['shares'] * trades['price'] < 0]['shares'].sum() * 
                              trades[trades['shares'] * trades['price'] < 0]['price'].mean())
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0
            
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades)
        }
        
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        绘制回测结果图表
        
        参数:
            save_path: 图表保存路径，如果为None则显示图表
        """
        # 设置样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 绘制权益曲线
        equity_curve = pd.Series(
            [v for _, v in self.results['equity_curve']],
            index=[d for d, _ in self.results['equity_curve']]
        )
        axes[0].plot(equity_curve.index, equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        
        # 绘制回撤曲线
        drawdown = pd.Series(
            [d for _, d in self.results['drawdown']],
            index=[d for d, _ in self.results['drawdown']]
        )
        axes[1].fill_between(drawdown.index, drawdown.values * 100, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        
        # 绘制市场状态
        market_states = pd.Series(
            [s['market_state'] for _, s in self.results['market_states']],
            index=[d for d, _ in self.results['market_states']]
        )
        axes[2].plot(market_states.index, market_states.values)
        axes[2].set_title('Market State')
        axes[2].set_ylabel('State')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成回测报告
        
        参数:
            save_path: 报告保存路径，如果为None则返回报告内容
            
        返回:
            str: 报告内容
        """
        metrics = self._calculate_metrics()
        
        report = f"""
        ===== 回测报告 =====
        
        回测期间: {self.results['equity_curve'][0][0]} 至 {self.results['equity_curve'][-1][0]}
        初始资金: {self.initial_capital:,.2f}
        最终资金: {self.results['equity_curve'][-1][1]:,.2f}
        
        绩效指标:
        - 总收益率: {metrics['total_return']:.2%}
        - 年化收益率: {metrics['annual_return']:.2%}
        - 波动率: {metrics['volatility']:.2%}
        - 夏普比率: {metrics['sharpe_ratio']:.2f}
        - 最大回撤: {metrics['max_drawdown']:.2%}
        
        交易统计:
        - 总交易次数: {metrics['total_trades']}
        - 胜率: {metrics['win_rate']:.2%}
        - 平均交易收益率: {metrics['avg_trade_return']:.2%}
        - 盈亏比: {metrics['profit_factor']:.2f}
        
        市场状态分布:
        {pd.Series([s['market_state'] for _, s in self.results['market_states']]).value_counts().to_string()}
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建回测引擎
    engine = BacktestEngine(
        config_path='path_to_config_file.json'
    )
    
    # 持仓股票
    portfolio_symbols = ['GOOG', 'TSLA', 'AMD', 'NVDA', 'PFE', 'MSFT', 'TMDX']
    
    # 运行回测
    results = engine.run_backtest(
        start_date='2024-01-01',
        end_date='2025-04-13',
        symbols=portfolio_symbols,
        initial_capital=1000000.0,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # 绘制结果
    engine.plot_results()
    
    # 生成报告
    report = engine.generate_report()
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