import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from monitor.market_monitor import MarketMonitor
from strategy.combined_strategy import CombinedStrategy

class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, 
                 initial_capital: float = 1000000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001,
                 position_size: float = 0.2,  # 单次交易资金比例
                 max_position: int = 100,     # 最大持仓数量
                 cooldown_period: int = 5,    # 交易冷却期
                 stop_loss: float = 0.1,      # 止损比例
                 take_profit: float = 0.2):   # 止盈比例
        """
        初始化回测引擎
        
        参数:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            position_size: 单次交易资金比例
            max_position: 最大持仓数量
            cooldown_period: 交易冷却期
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size = position_size
        self.max_position = max_position
        self.cooldown_period = cooldown_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 回测状态
        self.portfolio_value = []  # 组合价值
        self.positions = {}        # 持仓情况
        self.trades = []          # 交易记录
        self.last_trade_date = {}  # 上次交易日期
        self.entry_prices = {}     # 入场价格
        
        self._initialize_backtest()
        
    def _process_signals(self, signals):
        """处理策略生成的信号"""
        try:
            if isinstance(signals, pd.DataFrame):
                # 如果是DataFrame,尝试获取signal列或最后一列的值
                if 'signal' in signals.columns:
                    signal = float(signals['signal'].iloc[-1])
                else:
                    # 如果没有signal列,使用最后一列
                    signal = float(signals.iloc[-1, -1])
            elif isinstance(signals, pd.Series):
                signal = float(signals.iloc[-1])
            else:
                signal = float(signals)
            
            return signal
        except Exception as e:
            self.logger.error(f"处理信号时出错: {str(e)}")
            return 0.0  # 出错时返回空仓信号

    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: Any,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        运行回测
        
        参数:
            data: 市场数据
            strategy: 策略实例
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        返回:
            回测结果字典
        """
        try:
            # 初始化回测结果
            self.positions = pd.Series(index=data.index, data=0.0)
            self.returns = pd.Series(index=data.index, data=0.0)
            self.equity = pd.Series(index=data.index, data=self.initial_capital)
            self.drawdown = pd.Series(index=data.index, data=0.0)
            
            # 记录交易
            self.trades = []
            self.current_position = 0.0
            self.entry_price = 0.0
            self.entry_date = None
            
            # 过滤日期范围
            if start_date:
                data = data[data.index >= pd.Timestamp(start_date)]
            if end_date:
                data = data[data.index <= pd.Timestamp(end_date)]
            
            # 计算技术指标
            df = strategy.calculate_indicators(data)
            
            # 生成交易信号
            signals = strategy.generate_signals(df)
            
            # 模拟交易
            for i in range(1, len(df)):
                current_price = float(df['close'].iloc[i])
                current_date = df.index[i]
                
                # 获取当前信号
                if isinstance(signals, pd.DataFrame):
                    current_signals = signals.iloc[i:i+1]
                else:
                    current_signals = signals[i]
                
                signal = self._process_signals(current_signals)
                
                # 更新持仓
                if signal == 1 and self.current_position <= 0:
                    # 开多仓
                    self._open_long_position(current_price, current_date)
                elif signal == -1 and self.current_position >= 0:
                    # 开空仓
                    self._open_short_position(current_price, current_date)
                elif signal == 0 and self.current_position != 0:
                    # 平仓
                    self._close_position(current_price, current_date)
                
                # 更新收益和回撤
                self._update_returns(current_price)
                self._update_drawdown()
                
                # 记录持仓
                self.positions.iloc[i] = self.current_position
            
            # 计算回测指标
            metrics = self._calculate_metrics()
            
            return {
                'positions': self.positions,
                'returns': self.returns,
                'equity': self.equity,
                'drawdown': self.drawdown,
                'trades': self.trades,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"运行回测时出错: {str(e)}")
            raise
        
    def _prepare_data(self, df: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """准备回测数据"""
        try:
            # 确保数据按时间排序
            df = df.sort_index()
            
            # 处理yfinance数据格式
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)  # 删除第二级索引（股票代码）
            
            # 重命名列名（不区分大小写）
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'VOLUME': 'volume'
            }
            
            # 重命名存在的列
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # 确保所需的列都存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据缺少必要的列: {missing_columns}")
            
            # 只保留需要的列
            df = df[required_columns]
            
            # 如果提供了日期范围，则进行过滤
            if start_date and end_date:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                df = df[df.index >= start_ts]
                df = df[df.index <= end_ts]
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备数据时出错: {e}")
            raise e
        
    def _initialize_backtest(self):
        """初始化回测状态"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_capital = []  # 记录每日资金
        self.daily_positions = []  # 记录每日持仓
        
        # 设置初始组合价值
        if hasattr(self, 'current_data') and not self.current_data.empty:
            self.portfolio_value = pd.Series(dtype=float)  # 使用 pandas Series 存储组合价值
            self.portfolio_value[self.current_data.index[0]] = self.initial_capital
            
    def _execute_trades(self, signals: Dict[str, int], current_price: float):
        """执行交易"""
        try:
            current_date = self.current_data.index[-1]
            
            for symbol, signal in signals.items():
                # 获取当前持仓
                current_position = self.positions.get(symbol, 0)
                
                # 检查是否在冷却期
                if symbol in self.last_trade_date:
                    days_since_last_trade = (current_date - self.last_trade_date[symbol]).days
                    if days_since_last_trade < self.cooldown_period:
                        continue
                
                # 检查止盈止损
                if current_position > 0 and symbol in self.entry_prices:
                    entry_price = self.entry_prices[symbol]
                    price_change = (current_price - entry_price) / entry_price
                    
                    # 止损 - 只在价格大幅下跌时触发
                    if price_change <= -self.stop_loss * 1.5:  # 增加止损触发条件
                        self._execute_sell(symbol, current_position, current_price)
                        self.last_trade_date[symbol] = current_date
                        del self.entry_prices[symbol]
                        continue
                    
                    # 止盈 - 只在价格大幅上涨时触发
                    if price_change >= self.take_profit * 1.5:  # 增加止盈触发条件
                        self._execute_sell(symbol, current_position, current_price)
                        self.last_trade_date[symbol] = current_date
                        del self.entry_prices[symbol]
                        continue
                
                if signal > 0:  # 买入信号 - 允许在有持仓时继续买入
                    # 计算可用资金，使用总资金而不是当前现金
                    total_value = self.current_capital
                    for pos_symbol, pos_quantity in self.positions.items():
                        total_value += pos_quantity * current_price
                    
                    available_capital = total_value * self.position_size
                    
                    # 计算可买入的股数
                    max_shares = int(available_capital / (current_price * (1 + self.commission_rate + self.slippage_rate)))
                    quantity = min(max_shares, self.max_position - current_position)  # 考虑当前持仓
                    
                    if quantity > 0:
                        self._execute_buy(symbol, quantity, current_price)
                        self.last_trade_date[symbol] = current_date
                        if symbol not in self.entry_prices:  # 只在首次买入时记录入场价格
                            self.entry_prices[symbol] = current_price
                    
                elif signal < 0 and current_position > 0:  # 卖出信号且有持仓
                    self._execute_sell(symbol, current_position, current_price)
                    self.last_trade_date[symbol] = current_date
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]

        except Exception as e:
            self.logger.error(f"执行交易时出错: {str(e)}")
            raise
        
    def _execute_buy(self, symbol: str, quantity: int, price: float):
        """执行买入操作"""
        try:
            # 计算总成本（包含手续费和滑点）
            total_cost = quantity * price * (1 + self.commission_rate + self.slippage_rate)
            
            if total_cost <= self.current_capital:
                # 更新持仓
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                # 更新资金
                self.current_capital -= total_cost
                
                # 记录交易
                trade = {
                    'datetime': self.current_data.index[-1],
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'cost': total_cost,
                    'position': self.positions[symbol],
                    'capital': self.current_capital
                }
                self.trades.append(trade)
                
                self.logger.info(f"买入 {symbol}: {quantity} 股 @ {price:.2f}, 总成本: {total_cost:.2f}, 当前持仓: {self.positions[symbol]}")
                
        except Exception as e:
            self.logger.error(f"执行买入操作时出错: {str(e)}")
            raise
        
    def _execute_sell(self, symbol: str, quantity: int, price: float):
        """执行卖出操作"""
        try:
            if symbol in self.positions and self.positions[symbol] >= quantity:
                # 计算总收入（考虑手续费和滑点）
                revenue = quantity * price * (1 - self.commission_rate - self.slippage_rate)
                
                # 更新持仓
                self.positions[symbol] -= quantity
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                
                # 更新资金
                self.current_capital += revenue
                
                # 记录交易
                trade = {
                    'datetime': self.current_data.index[-1],
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'revenue': revenue,
                    'position': self.positions.get(symbol, 0),
                    'capital': self.current_capital
                }
                self.trades.append(trade)
                
                self.logger.info(f"卖出 {symbol}: {quantity} 股 @ {price:.2f}, 总收入: {revenue:.2f}, 当前持仓: {self.positions.get(symbol, 0)}")
                
        except Exception as e:
            self.logger.error(f"执行卖出操作时出错: {str(e)}")
            raise
        
    def _calculate_results(self) -> Dict[str, Any]:
        """计算回测结果"""
        try:
            # 构建权益曲线
            dates = [record['date'] for record in self.daily_capital]
            capitals = [record['capital'] for record in self.daily_capital]
            equity_curve = pd.Series(capitals, index=dates)
            
            # 计算收益率序列
            returns = equity_curve.pct_change().dropna()
            
            # 计算累积收益率
            total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
            
            # 计算年化收益率
            days = (dates[-1] - dates[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # 计算夏普比率
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            cummax = equity_curve.cummax()
            drawdown = (equity_curve - cummax) / cummax
            max_drawdown = abs(drawdown.min())
            
            # 计算胜率
            if len(self.trades) > 0:
                winning_trades = sum(1 for trade in self.trades 
                                   if trade['action'] == 'sell' and 
                                   trade['revenue'] > trade.get('cost', 0))
                total_trades = len([t for t in self.trades if t['action'] == 'sell'])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            else:
                win_rate = 0
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': equity_curve.iloc[-1],
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(self.trades),
                'trades': self.trades,
                'equity_curve': equity_curve,
                'returns': returns,
                'drawdown': drawdown
            }
            
        except Exception as e:
            self.logger.error(f"计算回测结果时出错: {str(e)}")
            raise 

    def _open_long_position(self, price, date):
        """开多仓"""
        # 计算可用资金
        available_capital = self.equity.iloc[-1] * self.position_size
        
        # 计算可买入数量
        quantity = available_capital / price
        quantity = min(quantity, self.max_position)
        
        # 计算手续费
        commission = price * quantity * self.commission_rate
        
        # 更新持仓
        self.current_position = quantity
        self.entry_price = price
        self.entry_date = date
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'LONG',
            'price': price,
            'quantity': quantity,
            'commission': commission
        })

    def _open_short_position(self, price, date):
        """开空仓"""
        # 计算可用资金
        available_capital = self.equity.iloc[-1] * self.position_size
        
        # 计算可卖出数量
        quantity = available_capital / price
        quantity = min(quantity, self.max_position)
        
        # 计算手续费
        commission = price * quantity * self.commission_rate
        
        # 更新持仓
        self.current_position = -quantity
        self.entry_price = price
        self.entry_date = date
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'SHORT',
            'price': price,
            'quantity': quantity,
            'commission': commission
        })

    def _close_position(self, price, date):
        """平仓"""
        if self.current_position == 0:
            return
        
        # 计算手续费
        commission = abs(self.current_position) * price * self.commission_rate
        
        # 计算收益
        if self.current_position > 0:
            profit = (price - self.entry_price) * self.current_position
        else:
            profit = (self.entry_price - price) * abs(self.current_position)
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'CLOSE',
            'price': price,
            'quantity': abs(self.current_position),
            'commission': commission,
            'profit': profit
        })
        
        # 更新持仓
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_date = None

    def _update_returns(self, price):
        """更新收益"""
        if self.current_position == 0:
            return
        
        # 计算当日收益
        if self.current_position > 0:
            daily_return = (price - self.entry_price) / self.entry_price
        else:
            daily_return = (self.entry_price - price) / self.entry_price
        
        # 更新收益序列
        self.returns.iloc[-1] = daily_return
        
        # 更新权益
        self.equity.iloc[-1] = self.equity.iloc[-2] * (1 + daily_return)

    def _update_drawdown(self):
        """更新回撤"""
        # 计算当前权益相对于历史最高点的回撤
        rolling_max = self.equity.expanding().max()
        self.drawdown = (rolling_max - self.equity) / rolling_max

    def _calculate_metrics(self):
        """计算回测指标"""
        try:
            # 计算年化收益率
            total_days = (self.equity.index[-1] - self.equity.index[0]).days
            total_return = (self.equity.iloc[-1] / self.equity.iloc[0]) - 1
            annual_return = (1 + total_return) ** (365 / total_days) - 1
            
            # 计算夏普比率
            daily_returns = self.returns[self.returns != 0]
            if len(daily_returns) > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            max_drawdown = self.drawdown.max()
            
            # 计算胜率
            profitable_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
            total_trades = len(self.trades)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades
            }
            
        except Exception as e:
            self.logger.error(f"计算回测指标时出错: {str(e)}")
            return {} 