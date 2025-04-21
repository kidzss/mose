import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from monitor.market_monitor import MarketMonitor
from strategy.combined_strategy import CombinedStrategy
from backtest.backtest_logger import BacktestLogger
from backtest.risk_manager import RiskManager
from backtest.position_manager import PositionManager
from backtest.market_analyzer import MarketAnalyzer

class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, 
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.001,
                 position_size: float = 0.1,
                 max_position: float = 0.5,
                 cooldown_period: int = 5,
                 stop_loss: float = 0.1,
                 take_profit: float = 0.2):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            position_size: 单次交易仓位比例
            max_position: 最大持仓比例
            cooldown_period: 交易冷却期
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.position_size = position_size
        self.max_position = max_position
        self.cooldown_period = cooldown_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 初始化记录器
        self.logger = BacktestLogger()
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.last_trade_date = None
        self.equity_curve = []
        self.trades = []
        self.drawdown = []
        self.monthly_returns = []
        self.current_month = None
        self.current_month_equity = self.initial_capital
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: Any,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 市场数据
            strategy: 策略实例
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        # 重置状态
        self.reset()
        
        # 设置日期范围
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 计算技术指标
        data = strategy.calculate_indicators(data)
        
        # 遍历每个交易日
        for date, row in data.iterrows():
            # 检查是否需要平仓
            if self.position != 0:
                # 检查止损止盈
                if self._check_stop_loss_take_profit(row['close']):
                    self._close_position(row['close'], date, 'stop_loss_take_profit')
                
                # 检查策略信号
                signal = strategy.generate_signals(data.loc[:date])
                if signal.iloc[-1]['signal'] == 0:  # 平仓信号
                    self._close_position(row['close'], date, 'strategy_signal')
            
            # 检查是否可以开仓
            if self.position == 0 and self._can_open_position(date):
                # 获取策略信号
                signal = strategy.generate_signals(data.loc[:date])
                if signal.iloc[-1]['signal'] != 0:  # 开仓信号
                    self._open_position(row['close'], date, signal.iloc[-1]['signal'])
            
            # 更新权益曲线
            self._update_equity_curve(row['close'], date)
            
            # 更新回撤
            self._update_drawdown(date)
            
            # 更新月度收益
            self._update_monthly_returns(date)
        
        # 计算回测结果
        results = self._calculate_results(data)
        
        # 记录结果
        self.logger.log_stock_result(data.name, results)
        
        return results
    
    def _check_stop_loss_take_profit(self, current_price: float) -> bool:
        """
        检查是否需要止损止盈
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否需要平仓
        """
        if self.position == 0:
            return False
        
        # 计算收益率
        returns = (current_price - self.entry_price) / self.entry_price
        if self.position > 0:  # 多头
            if returns <= -self.stop_loss or returns >= self.take_profit:
                return True
        else:  # 空头
            if returns >= self.stop_loss or returns <= -self.take_profit:
                return True
        
        return False
    
    def _can_open_position(self, date: datetime) -> bool:
        """
        检查是否可以开仓
        
        Args:
            date: 当前日期
            
        Returns:
            是否可以开仓
        """
        # 检查冷却期
        if self.last_trade_date and (date - self.last_trade_date).days < self.cooldown_period:
            return False
        
        return True
    
    def _open_position(self, price: float, date: datetime, signal: int):
        """
        开仓
        
        Args:
            price: 开仓价格
            date: 开仓日期
            signal: 交易信号
        """
        # 计算交易数量
        position_value = self.capital * self.position_size
        shares = position_value / price
        
        # 考虑滑点
        price = price * (1 + self.slippage_rate) if signal > 0 else price * (1 - self.slippage_rate)
        
        # 计算手续费
        commission = position_value * self.commission_rate
        
        # 更新状态
        self.position = shares * signal
        self.entry_price = price
        self.entry_date = date
        self.last_trade_date = date
        self.capital -= commission
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'open',
            'price': price,
            'shares': shares,
            'position': self.position,
            'commission': commission,
            'capital': self.capital
        })
    
    def _close_position(self, price: float, date: datetime, reason: str):
        """
        平仓
        
        Args:
            price: 平仓价格
            date: 平仓日期
            reason: 平仓原因
        """
        # 考虑滑点
        price = price * (1 - self.slippage_rate) if self.position > 0 else price * (1 + self.slippage_rate)
        
        # 计算收益
        returns = (price - self.entry_price) / self.entry_price
        profit = abs(self.position) * self.entry_price * returns
        
        # 计算手续费
        commission = abs(self.position) * price * self.commission_rate
        
        # 更新状态
        self.capital += profit - commission
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.last_trade_date = date
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'close',
            'price': price,
            'shares': abs(self.position),
            'position': 0,
            'profit': profit,
            'commission': commission,
            'capital': self.capital,
            'reason': reason
        })
    
    def _update_equity_curve(self, price: float, date: datetime):
        """
        更新权益曲线
        
        Args:
            price: 当前价格
            date: 当前日期
        """
        # 计算持仓价值
        position_value = 0
        if self.position != 0:
            position_value = self.position * price
        
        # 计算总权益
        equity = self.capital + position_value
        
        # 记录权益曲线
        self.equity_curve.append({
            'date': date,
            'equity': equity,
            'capital': self.capital,
            'position_value': position_value
        })
    
    def _update_drawdown(self, date: datetime):
        """
        更新回撤
        
        Args:
            date: 当前日期
        """
        if not self.equity_curve:
            return
        
        # 计算当前权益
        current_equity = self.equity_curve[-1]['equity']
        
        # 计算历史最高权益
        peak_equity = max([x['equity'] for x in self.equity_curve])
        
        # 计算回撤
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        
        # 记录回撤
        self.drawdown.append({
            'date': date,
            'drawdown': drawdown,
            'peak_equity': peak_equity,
            'current_equity': current_equity
        })
    
    def _update_monthly_returns(self, date: datetime):
        """
        更新月度收益
        
        Args:
            date: 当前日期
        """
        # 检查是否是新月份
        if self.current_month is None or date.month != self.current_month:
            if self.current_month is not None:
                # 计算上月收益
                monthly_return = (self.equity_curve[-1]['equity'] - self.current_month_equity) / self.current_month_equity
                self.monthly_returns.append({
                    'month': self.current_month,
                    'year': date.year,
                    'return': monthly_return
                })
            
            # 更新月份
            self.current_month = date.month
            self.current_month_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
    
    def _calculate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算回测结果
        
        Args:
            data: 市场数据
            
        Returns:
            回测结果
        """
        if not self.equity_curve:
            return None
        
        # 计算基本指标
        start_date = data.index[0]
        end_date = data.index[-1]
        total_days = (end_date - start_date).days
        years = total_days / 365.25
        
        # 计算收益
        total_return = (self.equity_curve[-1]['equity'] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算波动率
        returns = pd.Series([x['equity'] for x in self.equity_curve]).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 计算索提诺比率
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 计算最大回撤
        max_drawdown = max([x['drawdown'] for x in self.drawdown]) if self.drawdown else 0
        
        # 计算交易统计
        total_trades = len([t for t in self.trades if t['type'] == 'close'])
        winning_trades = len([t for t in self.trades if t['type'] == 'close' and t.get('profit', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均持仓天数
        holding_days = []
        entry_date = None
        for trade in self.trades:
            if trade['type'] == 'open':
                entry_date = trade['date']
            elif trade['type'] == 'close' and entry_date:
                holding_days.append((trade['date'] - entry_date).days)
                entry_date = None
        avg_holding_days = np.mean(holding_days) if holding_days else 0
        
        # 返回结果
        return {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve[-1]['equity'],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_holding_days': avg_holding_days,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'drawdown': self.drawdown,
            'monthly_returns': self.monthly_returns
        }
    
    def save_results(self, output_dir: str = 'backtest/results'):
        """
        保存回测结果
        
        Args:
            output_dir: 输出目录
        """
        self.logger.save_results(output_dir) 