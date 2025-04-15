import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path

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
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: Any,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        运行回测
        
        参数:
            data: 回测数据
            strategy: 策略实例
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            回测结果字典
        """
        try:
            # 数据预处理
            df = self._prepare_data(data, start_date, end_date)
            if df.empty:
                raise ValueError("回测数据为空")
                
            # 初始化回测状态
            self._initialize_backtest()
            
            # 记录每日资金和持仓
            self.daily_capital = []
            self.daily_positions = []
            
            # 逐日回测
            for i in range(1, len(df)):
                # 更新市场数据
                self.current_data = df.iloc[:i+1]
                current_date = df.index[i]
                
                # 生成交易信号
                signals = strategy.generate_signals(self.current_data)
                
                # 获取当前价格
                current_price = float(df['close'].iloc[i])
                
                # 执行交易
                if isinstance(signals, pd.DataFrame) and 'signal' in signals.columns:
                    signal = signals['signal'].iloc[-1]
                else:
                    signal = signals.iloc[-1] if isinstance(signals, pd.Series) else signals
                    
                if signal != 0:
                    self._execute_trades({'stock': signal}, current_price)
                
                # 更新每日资金和持仓
                total_value = self.current_capital
                for symbol, quantity in self.positions.items():
                    total_value += quantity * current_price
                    
                self.daily_capital.append({
                    'date': current_date,
                    'capital': total_value
                })
                
                self.daily_positions.append({
                    'date': current_date,
                    'positions': self.positions.copy()
                })
            
            # 计算回测结果
            results = self._calculate_results()
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行出错: {str(e)}")
            raise
            
    def _prepare_data(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """准备回测数据"""
        try:
            # 转换日期字符串为时间戳，不设置时区
            if start_date and end_date:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
            
            # 过滤日期范围
            df = df[df.index >= start_ts]
            df = df[df.index <= end_ts]
        
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
                # 检查冷却期
                if symbol in self.last_trade_date:
                    days_since_last_trade = (current_date - self.last_trade_date[symbol]).days
                    if days_since_last_trade < self.cooldown_period:
                        continue
                
                current_position = self.positions.get(symbol, 0)
                
                # 检查止损和止盈
                if current_position > 0 and symbol in self.entry_prices:
                    entry_price = self.entry_prices[symbol]
                    price_change = (current_price - entry_price) / entry_price
                    
                    if price_change <= -self.stop_loss:
                        self.logger.info(f"触发止损: {symbol}, 价格变化: {price_change:.2%}")
                        self._execute_sell(symbol, current_position, current_price)
                        continue
                    elif price_change >= self.take_profit:
                        self.logger.info(f"触发止盈: {symbol}, 价格变化: {price_change:.2%}")
                        self._execute_sell(symbol, current_position, current_price)
                        continue
                
                if signal > 0 and current_position == 0:  # 买入信号且当前无持仓
                    # 计算可用资金
                    available_capital = self.current_capital * self.position_size
                    
                    # 计算可买入的股数
                    max_shares = int(available_capital / (current_price * (1 + self.commission_rate + self.slippage_rate)))
                    quantity = max(1, min(max_shares, self.max_position))
                    
                    if quantity > 0:
                        self._execute_buy(symbol, quantity, current_price)
                        self.last_trade_date[symbol] = current_date
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