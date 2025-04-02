#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略评估工具模块

提供评估交易策略表现的各种指标和工具
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
from scipy import stats


def evaluate_strategy(
    returns: Union[np.ndarray, pd.Series],
    positions: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    annual_factor: int = 252,
    transaction_cost: float = 0.0,
    capital: float = 1.0
) -> Dict[str, float]:
    """
    评估交易策略表现
    
    参数:
        returns: 标的收益率序列
        positions: 交易策略持仓方向序列，值为1(多), -1(空), 0(不持仓)
        risk_free_rate: 无风险利率，年化
        annual_factor: 年化因子，日频为252，周频为52，月频为12
        transaction_cost: 交易成本，每笔交易占资金比例
        capital: 初始资金
        
    返回:
        包含各种评估指标的字典
    """
    # 转换为numpy数组
    if isinstance(returns, pd.Series):
        returns_np = returns.values
    else:
        returns_np = returns
        
    if isinstance(positions, pd.Series):
        positions_np = positions.values
    else:
        positions_np = positions
    
    # 确保长度一致
    assert len(returns_np) == len(positions_np), "收益率和持仓序列长度不一致"
    
    # 计算策略收益率
    strategy_returns = returns_np * positions_np
    
    # 计算交易成本
    if transaction_cost > 0:
        position_changes = np.diff(np.append(0, positions_np))
        position_changes = np.abs(position_changes)
        cost = position_changes * transaction_cost
        strategy_returns = strategy_returns - cost
    
    # 累积收益率
    cum_returns = np.cumprod(1 + strategy_returns) - 1
    
    # 年化收益率
    total_return = cum_returns[-1]
    n_periods = len(returns_np)
    annual_return = (1 + total_return) ** (annual_factor / n_periods) - 1
    
    # 最大回撤
    cumulative_wealth = (1 + cum_returns)
    previous_peaks = np.maximum.accumulate(cumulative_wealth)
    drawdowns = (cumulative_wealth - previous_peaks) / previous_peaks
    max_drawdown = np.min(drawdowns)
    
    # 收益风险比
    daily_return_std = np.std(strategy_returns) * np.sqrt(annual_factor)
    sharpe_ratio = (annual_return - risk_free_rate) / daily_return_std if daily_return_std > 0 else 0
    
    # 下行风险
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # 最大连续盈利/亏损
    win_streak, lose_streak = _get_max_streaks(strategy_returns)
    
    # 胜率
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
    
    # 盈亏比
    avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
    avg_loss = np.abs(np.mean(strategy_returns[strategy_returns < 0])) if np.any(strategy_returns < 0) else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 卡尔马比率
    calmar_ratio = -annual_return / max_drawdown if max_drawdown < 0 else 0
    
    # 计算Alpha和Beta
    if risk_free_rate > 0:
        # 计算超额收益
        excess_return = strategy_returns - risk_free_rate / annual_factor
        excess_market_return = returns_np - risk_free_rate / annual_factor
        
        # 计算Beta
        cov_matrix = np.cov(excess_return, excess_market_return)
        if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > 0:
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        else:
            beta = 0
            
        # 计算Alpha
        alpha = annual_return - risk_free_rate - beta * (np.mean(returns_np) * annual_factor - risk_free_rate)
    else:
        beta = 0
        alpha = 0
    
    # 整合结果
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "max_win_streak": win_streak,
        "max_lose_streak": lose_streak,
        "volatility": daily_return_std,
        "downside_risk": downside_std,
        "alpha": alpha,
        "beta": beta
    }


def _get_max_streaks(returns: np.ndarray) -> Tuple[int, int]:
    """
    计算最大连续盈利和亏损次数
    
    参数:
        returns: 收益率序列
        
    返回:
        (最大连续盈利次数, 最大连续亏损次数)
    """
    # 计算胜负序列，正收益为1，负收益为-1，零收益为0
    sign_changes = np.sign(returns)
    
    # 找出所有连续区间
    win_streaks = []
    lose_streaks = []
    
    current_streak = 0
    current_sign = 0
    
    for sign in sign_changes:
        if sign == 0:  # 忽略零收益
            continue
            
        if sign == current_sign:  # 同向，增加连续计数
            current_streak += 1
        else:  # 不同向，重置连续计数
            if current_sign > 0 and current_streak > 0:
                win_streaks.append(current_streak)
            elif current_sign < 0 and current_streak > 0:
                lose_streaks.append(current_streak)
                
            current_sign = sign
            current_streak = 1
    
    # 添加最后一段连续区间
    if current_sign > 0 and current_streak > 0:
        win_streaks.append(current_streak)
    elif current_sign < 0 and current_streak > 0:
        lose_streaks.append(current_streak)
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_lose_streak = max(lose_streaks) if lose_streaks else 0
    
    return max_win_streak, max_lose_streak


def plot_strategy_performance(
    returns: Union[np.ndarray, pd.Series],
    positions: Union[np.ndarray, pd.Series],
    benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
    risk_free_rate: float = 0.0,
    annual_factor: int = 252,
    transaction_cost: float = 0.0,
    figsize: Tuple[int, int] = (12, 9),
    title: str = "策略表现"
) -> plt.Figure:
    """
    绘制策略表现图表
    
    参数:
        returns: 标的收益率序列
        positions: 交易策略持仓方向序列
        benchmark_returns: 基准收益率序列，可选
        risk_free_rate: 无风险利率
        annual_factor: 年化因子
        transaction_cost: 交易成本
        figsize: 图形尺寸
        title: 图形标题
        
    返回:
        matplotlib图形对象
    """
    # 转换为pandas.Series
    if isinstance(returns, np.ndarray):
        if isinstance(positions, pd.Series):
            returns = pd.Series(returns, index=positions.index)
        else:
            returns = pd.Series(returns)
    
    if isinstance(positions, np.ndarray):
        if isinstance(returns, pd.Series):
            positions = pd.Series(positions, index=returns.index)
        else:
            positions = pd.Series(positions)
    
    # 计算策略收益率
    strategy_returns = returns * positions
    
    # 计算交易成本
    if transaction_cost > 0:
        position_changes = positions.diff().fillna(0).abs()
        costs = position_changes * transaction_cost
        strategy_returns = strategy_returns - costs
    
    # 计算策略累积收益
    strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
    
    # 计算基准累积收益
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = pd.Series(benchmark_returns, index=returns.index)
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
    
    # 评估策略
    metrics = evaluate_strategy(
        returns, 
        positions, 
        risk_free_rate=risk_free_rate,
        annual_factor=annual_factor,
        transaction_cost=transaction_cost
    )
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # 设置网格
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    # 子图1：累积收益曲线
    ax1 = fig.add_subplot(gs[0, :])
    strategy_cum_returns.mul(100).plot(ax=ax1, color='blue', label='策略')
    
    if benchmark_returns is not None:
        benchmark_cum_returns.mul(100).plot(ax=ax1, color='gray', label='基准', alpha=0.7)
        
    ax1.set_ylabel('累积收益率 (%)')
    ax1.set_title(f'{title} - 累积收益')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2：回撤
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    cumulative_wealth = (1 + strategy_cum_returns)
    previous_peaks = cumulative_wealth.expanding().max()
    drawdowns = ((cumulative_wealth - previous_peaks) / previous_peaks) * 100
    drawdowns.plot(ax=ax2, color='red', label='策略回撤')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title('回撤')
    ax2.grid(True)
    
    # 子图3：月度收益热图
    ax3 = fig.add_subplot(gs[2, 0])
    if isinstance(strategy_returns.index, pd.DatetimeIndex):
        monthly_returns = strategy_returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns = monthly_returns.to_frame('returns')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        # 创建月度收益透视表
        try:
            pivot = monthly_returns.pivot(index='year', columns='month', values='returns')
            pivot = pivot * 100  # 转为百分比
            
            im = ax3.imshow(pivot.values, cmap='RdYlGn', vmin=-10, vmax=10)
            ax3.set_title('月度收益 (%)')
            
            # 设置坐标轴标签
            ax3.set_yticks(range(len(pivot.index)))
            ax3.set_yticklabels(pivot.index)
            
            months = ['一月', '二月', '三月', '四月', '五月', '六月', 
                     '七月', '八月', '九月', '十月', '十一月', '十二月']
            ax3.set_xticks(range(len(months)))
            ax3.set_xticklabels(months, rotation=45)
            
            # 添加数值标签
            for i in range(len(pivot.index)):
                for j in range(len(months)):
                    if j < pivot.shape[1] and not np.isnan(pivot.values[i, j]):
                        text = ax3.text(j, i, f"{pivot.values[i, j]:.1f}",
                                       ha="center", va="center", color="black",
                                       fontsize=8)
            
            plt.colorbar(im, ax=ax3)
        except:
            ax3.text(0.5, 0.5, "无法生成月度收益热图", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('月度收益')
    else:
        ax3.text(0.5, 0.5, "需要日期索引来生成月度收益热图", 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('月度收益')
    
    # 子图4：绩效指标
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 添加性能指标文本框
    metrics_text = (
        f"年化收益率: {metrics['annual_return']*100:.2f}%\n"
        f"夏普比率: {metrics['sharpe_ratio']:.2f}\n"
        f"索提诺比率: {metrics['sortino_ratio']:.2f}\n"
        f"最大回撤: {metrics['max_drawdown']*100:.2f}%\n"
        f"卡尔马比率: {metrics['calmar_ratio']:.2f}\n"
        f"胜率: {metrics['win_rate']*100:.2f}%\n"
        f"盈亏比: {metrics['profit_loss_ratio']:.2f}\n"
        f"最大连胜: {metrics['max_win_streak']}\n"
        f"最大连亏: {metrics['max_lose_streak']}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax4.set_title('绩效指标')
    
    plt.tight_layout()
    
    return fig


def calculate_ic(
    signals: Union[np.ndarray, pd.DataFrame], 
    forward_returns: Union[np.ndarray, pd.Series],
    method: str = 'pearson',
    periods: List[int] = [1, 5, 10, 20]
) -> pd.DataFrame:
    """
    计算信息系数（Information Coefficient）
    
    参数:
        signals: 信号值，形状为[n_samples, n_signals]
        forward_returns: 未来收益率，形状为[n_samples]
        method: 相关系数方法，可选'pearson', 'spearman'或'kendall'
        periods: 未来收益周期列表
        
    返回:
        IC值DataFrame，行为周期，列为信号
    """
    # 转换为pandas对象
    if isinstance(signals, np.ndarray):
        if isinstance(forward_returns, pd.Series):
            signals = pd.DataFrame(signals, index=forward_returns.index)
        else:
            signals = pd.DataFrame(signals)
    
    if isinstance(forward_returns, np.ndarray):
        if isinstance(signals, pd.DataFrame):
            forward_returns = pd.Series(forward_returns, index=signals.index)
        else:
            forward_returns = pd.Series(forward_returns)
    
    # 初始化结果DataFrame
    if isinstance(signals, pd.DataFrame):
        ic_df = pd.DataFrame(index=periods, columns=signals.columns)
    else:
        ic_df = pd.DataFrame(index=periods, columns=[f'signal_{i}' for i in range(signals.shape[1])])
    
    # 计算不同周期的IC
    for period in periods:
        # 计算未来收益
        future_return = forward_returns.shift(-period)
        
        # 对每个信号计算IC
        for i, col in enumerate(ic_df.columns):
            if isinstance(signals, pd.DataFrame):
                signal = signals[col]
            else:
                signal = pd.Series(signals[:, i], index=forward_returns.index)
                
            # 计算相关系数
            if method == 'pearson':
                ic = signal.corr(future_return, method='pearson')
            elif method == 'spearman':
                ic = signal.corr(future_return, method='spearman')
            elif method == 'kendall':
                ic = signal.corr(future_return, method='kendall')
            else:
                raise ValueError(f"不支持的相关系数方法: {method}")
                
            ic_df.loc[period, col] = ic
    
    return ic_df


def plot_ic_heatmap(
    ic_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "信号IC热图"
) -> plt.Figure:
    """
    绘制IC热图
    
    参数:
        ic_df: IC值DataFrame，行为周期，列为信号
        figsize: 图形尺寸
        title: 图形标题
        
    返回:
        matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    im = ax.imshow(ic_df.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    
    # 添加颜色条
    plt.colorbar(im, ax=ax)
    
    # 设置坐标轴标签
    ax.set_xticks(range(len(ic_df.columns)))
    ax.set_xticklabels(ic_df.columns, rotation=45, ha='right')
    
    ax.set_yticks(range(len(ic_df.index)))
    ax.set_yticklabels(ic_df.index)
    
    # 添加数值标签
    for i in range(len(ic_df.index)):
        for j in range(len(ic_df.columns)):
            text = ax.text(j, i, f"{ic_df.values[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    ax.set_title(title)
    ax.set_ylabel("未来收益周期")
    ax.set_xlabel("信号")
    
    plt.tight_layout()
    
    return fig 