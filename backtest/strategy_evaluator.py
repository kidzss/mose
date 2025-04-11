import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TradeRecord:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    profit_pct: float
    holding_days: int
    signal_type: str

@dataclass
class StrategyMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_profit_per_trade: float
    avg_holding_days: float
    total_return: float
    annual_return: float
    volatility: float
    downside_risk: float
    max_win_streak: int
    max_lose_streak: int
    max_drawdown_duration: int
    recovery_periods: int
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'avg_holding_days': self.avg_holding_days,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'downside_risk': self.downside_risk,
            'max_win_streak': self.max_win_streak,
            'max_lose_streak': self.max_lose_streak,
            'max_drawdown_duration': self.max_drawdown_duration,
            'recovery_periods': self.recovery_periods
        }

class StrategyEvaluator:
    def __init__(self, price_data: pd.DataFrame):
        """
        初始化策略评估器
        
        参数:
            price_data: 价格数据，包含OHLCV数据
        """
        self.df = price_data.copy()
        
        # 确保列名一致性
        if 'Close' in self.df.columns and 'close' not in self.df.columns:
            self.df['close'] = self.df['Close']
        elif 'close' in self.df.columns and 'Close' not in self.df.columns:
            self.df['Close'] = self.df['close']
        
        self.trades = []
        
    def calculate_returns(self, position_series: pd.Series) -> pd.Series:
        """计算每日收益率"""
        price_returns = self.df['Close'].pct_change()
        strategy_returns = position_series.shift(1) * price_returns
        return strategy_returns
        
    def calculate_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - (risk_free_rate / 252)  # 日度无风险收益率
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 1e-6
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_calmar_ratio(self, returns: pd.Series, annual_return: float) -> float:
        """计算卡尔马比率"""
        max_dd = self.calculate_drawdown(returns)
        return annual_return / max_dd if max_dd > 0 else 0
    
    def calculate_downside_risk(self, returns: pd.Series, min_acceptable_return: float = 0) -> float:
        """计算下行风险"""
        downside_returns = returns[returns < min_acceptable_return]
        return downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    def calculate_winning_streaks(self, trades: List[TradeRecord]) -> tuple:
        """计算连胜和连亏次数"""
        if not trades:
            return 0, 0
        
        current_win_streak = 0
        current_lose_streak = 0
        max_win_streak = 0
        max_lose_streak = 0
        
        for trade in trades:
            if trade.profit_pct > 0:
                current_win_streak += 1
                current_lose_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_lose_streak += 1
                current_win_streak = 0
                max_lose_streak = max(max_lose_streak, current_lose_streak)
                
        return max_win_streak, max_lose_streak
    
    def calculate_drawdown_duration(self, returns: pd.Series) -> tuple:
        """计算最大回撤持续期和恢复期"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # 找到最大回撤区间
        underwater = drawdown < 0
        if not underwater.any():
            return 0, 0
            
        # 计算最大回撤持续时间
        underwater_periods = underwater.astype(int)
        underwater_start = (underwater_periods.diff() == 1).astype(bool)
        underwater_end = (underwater_periods.diff() == -1).astype(bool)
        
        if not underwater_start.any() or not underwater_end.any():
            return len(underwater) if underwater.iloc[-1] else 0, 0
            
        underwater_start_indices = underwater_start[underwater_start].index
        underwater_end_indices = underwater_end[underwater_end].index
        
        # 确保配对
        if len(underwater_start_indices) > len(underwater_end_indices):
            # 当前仍在回撤中，将最后一个日期作为临时结束点
            temp_end = returns.index[-1]
            underwater_end_indices = underwater_end_indices.append(pd.Index([temp_end]))
            
        max_dd_duration = 0
        max_recovery_period = 0
        
        for i in range(len(underwater_start_indices)):
            if i >= len(underwater_end_indices):
                break
                
            start_idx = underwater_start_indices[i]
            end_idx = underwater_end_indices[i]
            
            duration = len(returns.loc[start_idx:end_idx])
            
            # 找到恢复点
            if i < len(underwater_start_indices) - 1:
                next_start_idx = underwater_start_indices[i+1]
                # 检查是否在下一次回撤前完全恢复
                recovery_points = cumulative[(cumulative.index > end_idx) & 
                                            (cumulative.index < next_start_idx) & 
                                            (cumulative >= running_max.loc[start_idx])]
                
                if not recovery_points.empty:
                    recovery_idx = recovery_points.index[0]
                    recovery_period = len(returns.loc[end_idx:recovery_idx])
                    max_recovery_period = max(max_recovery_period, recovery_period)
            
            max_dd_duration = max(max_dd_duration, duration)
            
        return max_dd_duration, max_recovery_period
        
    def analyze_trades(self, positions: pd.Series) -> StrategyMetrics:
        """
        分析交易记录，生成策略指标
        
        参数:
            positions: 持仓信号序列
            
        返回:
            策略指标
        """
        # 确保列名一致性
        if 'Close' not in self.df.columns and 'close' in self.df.columns:
            self.df['Close'] = self.df['close']
        elif 'close' not in self.df.columns and 'Close' in self.df.columns:
            self.df['close'] = self.df['Close']
            
        if positions.empty:
            return StrategyMetrics(
                total_trades=0, win_rate=0, profit_factor=0, max_drawdown=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_profit_per_trade=0, avg_holding_days=0,
                total_return=0, annual_return=0, volatility=0, downside_risk=0,
                max_win_streak=0, max_lose_streak=0, max_drawdown_duration=0,
                recovery_periods=0
            )
        
        # 生成交易记录
        trades = []
        in_position = False
        entry_date = None
        entry_price = 0
        
        for date, pos in positions.items():
            if not in_position and pos == 1:  # 开仓
                entry_date = date
                entry_price = self.df.loc[date, 'Close']
                in_position = True
            elif in_position and pos == 0:  # 平仓
                exit_price = self.df.loc[date, 'Close']
                profit_pct = (exit_price / entry_price - 1) * 100
                holding_days = (date - entry_date).days
                
                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit_pct=profit_pct,
                    holding_days=holding_days,
                    signal_type='LONG'
                ))
                in_position = False
                
        # 如果最后还持有仓位，用最后一个价格平仓
        if in_position:
            last_date = positions.index[-1]
            exit_price = self.df.loc[last_date, 'Close']
            profit_pct = (exit_price / entry_price - 1) * 100
            holding_days = (last_date - entry_date).days
            
            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=last_date,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                holding_days=holding_days,
                signal_type='LONG'
            ))
            
        # 保存交易记录
        self.trades = trades
            
        if not trades:
            return StrategyMetrics(
                total_trades=0, win_rate=0, profit_factor=0, max_drawdown=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_profit_per_trade=0, avg_holding_days=0,
                total_return=0, annual_return=0, volatility=0, downside_risk=0,
                max_win_streak=0, max_lose_streak=0, max_drawdown_duration=0,
                recovery_periods=0
            )
            
        # 计算基本指标
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.profit_pct > 0])
        win_rate = winning_trades / total_trades
        
        # 计算盈亏因子
        gains = sum(t.profit_pct for t in trades if t.profit_pct > 0)
        losses = abs(sum(t.profit_pct for t in trades if t.profit_pct < 0))
        profit_factor = gains / losses if losses != 0 else float('inf')
        
        # 计算平均指标
        avg_profit = sum(t.profit_pct for t in trades) / total_trades
        avg_holding = sum(t.holding_days for t in trades) / total_trades
        
        # 计算总收益和年化收益
        total_return = np.prod([1 + t.profit_pct/100 for t in trades]) - 1
        days = (trades[-1].exit_date - trades[0].entry_date).days
        annual_return = (1 + total_return) ** (365/days) - 1 if days > 0 else 0
        
        # 计算收益率序列
        returns = self.calculate_returns(positions)
        
        # 计算风险指标
        volatility = returns.std() * np.sqrt(252)
        max_dd = self.calculate_drawdown(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, annual_return)
        downside_risk = self.calculate_downside_risk(returns)
        
        # 计算连胜连亏
        max_win_streak, max_lose_streak = self.calculate_winning_streaks(trades)
        
        # 计算回撤持续期和恢复期
        max_dd_duration, recovery_periods = self.calculate_drawdown_duration(returns)
        
        return StrategyMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_profit_per_trade=avg_profit,
            avg_holding_days=avg_holding,
            total_return=total_return * 100,
            annual_return=annual_return * 100,
            volatility=volatility * 100,
            downside_risk=downside_risk * 100,
            max_win_streak=max_win_streak,
            max_lose_streak=max_lose_streak,
            max_drawdown_duration=max_dd_duration,
            recovery_periods=recovery_periods
        )
        
    def evaluate_strategy(self, positions: pd.Series) -> Dict:
        """评估策略表现"""
        # 分析交易记录
        metrics = self.analyze_trades(positions)
        
        # 保存交易记录
        self.trades = self.trades
        
        return {
            'metrics': metrics,
            'trades': self.trades
        }
        
    def generate_report(self) -> str:
        """生成策略评估报告"""
        if not self.trades:
            return """
策略评估报告
===========
没有交易记录可供分析
"""
            
        metrics = self.analyze_trades(self.trades)
        
        report = f"""
策略评估报告
===========
总体表现:
- 总交易次数: {metrics.total_trades}
- 胜率: {metrics.win_rate:.2%}
- 盈亏比: {metrics.profit_factor:.2f}
- 总收益率: {metrics.total_return:.2f}%
- 年化收益率: {metrics.annual_return:.2f}%

风险指标:
- 波动率: {metrics.volatility:.2f}%
- 下行风险: {metrics.downside_risk:.2f}%
- 最大回撤: {metrics.max_drawdown:.2f}%
- 最大回撤持续期: {metrics.max_drawdown_duration}天
- 恢复期: {metrics.recovery_periods}天

风险调整收益:
- 夏普比率: {metrics.sharpe_ratio:.2f}
- 索提诺比率: {metrics.sortino_ratio:.2f}
- 卡尔马比率: {metrics.calmar_ratio:.2f}

交易统计:
- 平均每笔收益: {metrics.avg_profit_per_trade:.2f}%
- 平均持仓天数: {metrics.avg_holding_days:.1f}天
- 最大连胜次数: {metrics.max_win_streak}次
- 最大连亏次数: {metrics.max_lose_streak}次

最近五笔交易:
"""
        
        for trade in self.trades[-5:]:
            report += f"""
交易记录:
- 开仓日期: {trade.entry_date.date()}
- 开仓价格: {trade.entry_price:.2f}
- 平仓日期: {trade.exit_date.date()}
- 平仓价格: {trade.exit_price:.2f}
- 收益率: {trade.profit_pct:.2f}%
- 持仓天数: {trade.holding_days}天
"""
        
        return report 