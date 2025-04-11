#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强的策略评估模块

提供更全面的策略评估指标和统计显著性测试
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy import stats
import pandas_ta as ta

def evaluate_strategy(returns: pd.Series, 
                     positions: Union[pd.Series, np.ndarray],
                     benchmark_returns: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.0,
                     frequency: str = 'daily',
                     transaction_costs: Optional[float] = None) -> Dict[str, float]:
    """
    评估交易策略的表现
    
    参数:
        returns: 资产收益率序列
        positions: 策略持仓序列，值为1表示多头，-1表示空头，0表示不持仓
        benchmark_returns: 基准收益率序列（可选）
        risk_free_rate: 无风险利率，用于计算风险调整后收益率
        frequency: 收益率频率，可选'daily', 'weekly', 'monthly', 'quarterly', 'annual'
        transaction_costs: 交易成本（作为收益率的百分比），None表示不考虑成本
        
    返回:
        包含各种表现指标的字典
    """
    # 频率转换为对应的年化因子
    freq_multiplier = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4,
        'annual': 1
    }
    periods = freq_multiplier.get(frequency.lower(), 252)
    
    # 确保positions和returns索引一致
    if isinstance(positions, pd.Series):
        # 获取共同的索引
        common_index = returns.index.intersection(positions.index)
        returns = returns.loc[common_index]
        positions = positions.loc[common_index]
    
    # 计算策略收益率
    strategy_returns = returns * positions
    
    # 处理交易成本（如果指定）
    if transaction_costs is not None and transaction_costs > 0:
        # 检测持仓变化
        position_changes = positions.diff().fillna(0)
        # 交易成本只在持仓变化时产生
        cost = np.abs(position_changes) * transaction_costs
        # 从策略收益中减去交易成本
        strategy_returns = strategy_returns - cost
    
    # 计算累积收益
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
    
    # 基础表现指标
    metrics = {}
    metrics['total_return'] = total_return
    
    # 年化收益率
    metrics['annual_return'] = annual_return(
        strategy_returns,
        period=frequency,
        annualization=periods
    )
    
    # 最大回撤
    metrics['max_drawdown'] = max_drawdown(strategy_returns)
    
    # 波动率和风险指标
    metrics['volatility'] = strategy_returns.std() * np.sqrt(periods)
    metrics['downside_risk'] = downside_risk(strategy_returns, required_return=0)
    
    # 风险调整后收益率指标
    metrics['sharpe_ratio'] = sharpe_ratio(
        strategy_returns,
        risk_free=risk_free_rate,
        period=frequency,
        annualization=periods
    )
    
    metrics['sortino_ratio'] = sortino_ratio(
        strategy_returns,
        required_return=risk_free_rate,
        period=frequency,
        annualization=periods
    )
    
    metrics['calmar_ratio'] = calmar_ratio(
        strategy_returns,
        period=frequency,
        annualization=periods
    )
    
    # 胜率和盈亏比
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns < 0]
    
    metrics['win_rate'] = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = np.abs(losing_trades.mean()) if len(losing_trades) > 0 else float('inf')
    metrics['profit_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 连胜连亏统计
    win_streak = 0
    lose_streak = 0
    max_win_streak = 0
    max_lose_streak = 0
    current_win_streak = 0
    current_lose_streak = 0
    
    for ret in strategy_returns:
        if ret > 0:
            current_win_streak += 1
            current_lose_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        elif ret < 0:
            current_lose_streak += 1
            current_win_streak = 0
            max_lose_streak = max(max_lose_streak, current_lose_streak)
        else:  # ret == 0
            # 不改变streak状态
            pass
    
    metrics['max_win_streak'] = max_win_streak
    metrics['max_lose_streak'] = max_lose_streak
    
    # 回撤分析
    dd_series = calculate_drawdowns(strategy_returns)
    metrics['avg_drawdown'] = dd_series.mean() if not dd_series.empty else 0
    metrics['max_drawdown_duration'] = calculate_max_drawdown_duration(strategy_returns)
    
    # 如果提供了基准，计算相对指标
    if benchmark_returns is not None:
        # 确保基准与策略收益率索引一致
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 0:
            strat_returns_aligned = strategy_returns.loc[common_index]
            bench_returns_aligned = benchmark_returns.loc[common_index]
            
            # 计算Alpha和Beta
            alpha, beta = alpha_beta(
                strat_returns_aligned, 
                bench_returns_aligned,
                risk_free=risk_free_rate,
                period=frequency,
                annualization=periods
            )
            
            metrics['alpha'] = alpha
            metrics['beta'] = beta
            
            # 计算相对基准的信息比率
            tracking_error = (strat_returns_aligned - bench_returns_aligned).std() * np.sqrt(periods)
            metrics['information_ratio'] = (metrics['annual_return'] - bench_returns_aligned.mean() * periods) / tracking_error if tracking_error > 0 else 0
            
            # 计算击败基准的时间百分比
            metrics['outperformance_rate'] = (strat_returns_aligned > bench_returns_aligned).mean()
            
            # 计算上行捕获率和下行捕获率
            up_market = bench_returns_aligned > 0
            down_market = bench_returns_aligned < 0
            
            if up_market.sum() > 0:
                metrics['upside_capture'] = strat_returns_aligned[up_market].mean() / bench_returns_aligned[up_market].mean()
            else:
                metrics['upside_capture'] = 1.0
                
            if down_market.sum() > 0:
                metrics['downside_capture'] = strat_returns_aligned[down_market].mean() / bench_returns_aligned[down_market].mean()
            else:
                metrics['downside_capture'] = 1.0
        else:
            # 如果没有共同的索引，设置为默认值
            metrics['alpha'] = 0.0
            metrics['beta'] = 0.0
            metrics['information_ratio'] = 0.0
            metrics['outperformance_rate'] = 0.0
            metrics['upside_capture'] = 1.0
            metrics['downside_capture'] = 1.0
    else:
        # 如果没有提供基准，设置为默认值
        metrics['alpha'] = 0.0
        metrics['beta'] = 0.0
    
    return metrics


def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """
    计算回撤序列
    
    参数:
        returns: 收益率序列
        
    返回:
        回撤序列
    """
    # 计算累积收益
    cumulative_returns = (1 + returns).cumprod()
    
    # 计算累积最大值
    running_max = cumulative_returns.cummax()
    
    # 计算回撤
    drawdowns = (cumulative_returns / running_max) - 1
    
    return drawdowns


def calculate_max_drawdown_duration(returns: pd.Series) -> int:
    """
    计算最大回撤持续时间（以交易日为单位）
    
    参数:
        returns: 收益率序列
        
    返回:
        最大回撤持续的天数
    """
    # 计算累积收益
    cumulative_returns = (1 + returns).cumprod()
    
    # 计算累积最大值
    running_max = cumulative_returns.cummax()
    
    # 找出新高点
    is_high = (running_max == cumulative_returns)
    
    # 确定最大回撤持续时间
    max_duration = 0
    current_duration = 0
    
    for is_max in is_high:
        if is_max:
            current_duration = 0
        else:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
    
    return max_duration


def calculate_ic(signals: pd.DataFrame, 
                returns: pd.Series,
                method: str = 'pearson',
                periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    计算信息系数 (IC)
    
    参数:
        signals: 信号DataFrame
        returns: 收益率Series
        method: 相关性计算方法，可选 'pearson', 'spearman', 'kendall'
        periods: 未来收益率计算期间列表
        
    返回:
        包含各个周期的IC值的DataFrame
    """
    correlation_methods = {
        'pearson': lambda x, y: x.corr(y, method='pearson'),
        'spearman': lambda x, y: x.corr(y, method='spearman'),
        'kendall': lambda x, y: x.corr(y, method='kendall')
    }
    
    corr_func = correlation_methods.get(method.lower(), correlation_methods['pearson'])
    
    # 计算不同周期的未来收益率
    future_returns = {}
    for period in periods:
        future_returns[period] = returns.shift(-period+1)
    
    # 计算信号与未来收益率的相关性
    ic_data = {}
    for period, future_ret in future_returns.items():
        # 确保索引一致
        common_index = signals.index.intersection(future_ret.dropna().index)
        if len(common_index) > 0:
            signal_aligned = signals.loc[common_index]
            returns_aligned = future_ret.loc[common_index]
            
            # 计算每个信号的IC
            ic_values = {}
            for col in signal_aligned.columns:
                ic = corr_func(signal_aligned[col], returns_aligned)
                ic_values[col] = ic
            
            ic_data[period] = ic_values
    
    # 转换为DataFrame
    ic_df = pd.DataFrame(ic_data).T
    
    return ic_df


def calculate_statistical_significance(strategy_returns: pd.Series, 
                                      benchmark_returns: Optional[pd.Series] = None,
                                      confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    计算策略表现的统计显著性
    
    参数:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列（可选）
        confidence_level: 置信水平
        
    返回:
        包含各项测试的p值和置信区间的字典
    """
    results = {}
    
    # 1. 均值t检验（策略收益率是否显著不为零）
    t_stat, p_value = stats.ttest_1samp(strategy_returns.dropna(), 0)
    
    # 计算置信区间
    se = strategy_returns.std() / np.sqrt(len(strategy_returns))
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(strategy_returns)-1)
    ci_lower = strategy_returns.mean() - t_critical * se
    ci_upper = strategy_returns.mean() + t_critical * se
    
    results['mean_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'significant': p_value < (1 - confidence_level)
    }
    
    # 2. 正态性检验（策略收益率是否服从正态分布）
    shapiro_stat, shapiro_p = stats.shapiro(strategy_returns.dropna())
    
    results['normality_test'] = {
        'statistic': shapiro_stat,
        'p_value': shapiro_p,
        'is_normal': shapiro_p >= (1 - confidence_level)
    }
    
    # 3. 自相关检验（策略收益率是否表现出自相关性）
    acf_1 = strategy_returns.autocorr(lag=1)
    
    # 计算自相关显著性
    se_acf = 1 / np.sqrt(len(strategy_returns))
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)
    acf_ci_lower = -z_critical * se_acf
    acf_ci_upper = z_critical * se_acf
    
    results['autocorrelation_test'] = {
        'acf_1': acf_1,
        'confidence_interval': (acf_ci_lower, acf_ci_upper),
        'significant': abs(acf_1) > z_critical * se_acf
    }
    
    # 4. 如果提供了基准，执行相对基准的测试
    if benchmark_returns is not None:
        # 确保索引一致
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 0:
            strat_returns_aligned = strategy_returns.loc[common_index]
            bench_returns_aligned = benchmark_returns.loc[common_index]
            
            # 4.1 双样本t检验（策略收益率是否显著优于基准）
            paired_t_stat, paired_p_value = stats.ttest_rel(
                strat_returns_aligned.dropna(), 
                bench_returns_aligned.dropna()
            )
            
            # 计算差异的置信区间
            diff = strat_returns_aligned - bench_returns_aligned
            diff_mean = diff.mean()
            diff_se = diff.std() / np.sqrt(len(diff))
            diff_t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(diff)-1)
            diff_ci_lower = diff_mean - diff_t_critical * diff_se
            diff_ci_upper = diff_mean + diff_t_critical * diff_se
            
            results['benchmark_comparison'] = {
                't_statistic': paired_t_stat,
                'p_value': paired_p_value,
                'confidence_interval': (diff_ci_lower, diff_ci_upper),
                'significant': paired_p_value < (1 - confidence_level)
            }
    
    return results


def evaluate_portfolio(returns: pd.Series,
                      weights: Union[pd.DataFrame, np.ndarray],
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.0,
                      frequency: str = 'daily',
                      transaction_costs: Optional[float] = None) -> Dict[str, pd.Series]:
    """
    评估投资组合的历史表现
    
    参数:
        returns: 各个资产的收益率DataFrame，每列是一个资产
        weights: 投资组合权重DataFrame或数组，索引与returns对应
        benchmark_returns: 基准收益率序列（可选）
        risk_free_rate: 无风险利率
        frequency: 收益率频率
        transaction_costs: 交易成本
        
    返回:
        包含各时间点评估指标的字典
    """
    # 检查输入
    if isinstance(returns, pd.Series):
        # 如果returns是Series，改为单列DataFrame
        returns = pd.DataFrame(returns)
    
    # 转换weights为DataFrame，如果不是的话
    if not isinstance(weights, pd.DataFrame):
        weights = pd.DataFrame(weights, index=returns.index, columns=returns.columns)
    
    # 确保weights和returns索引一致
    common_index = returns.index.intersection(weights.index)
    if len(common_index) == 0:
        raise ValueError("权重和收益率没有共同的索引")
    
    returns = returns.loc[common_index]
    weights = weights.loc[common_index]
    
    # 计算各时间点的投资组合收益率
    portfolio_returns = (weights * returns).sum(axis=1)
    
    # 处理交易成本（如果指定）
    if transaction_costs is not None and transaction_costs > 0:
        # 计算权重变化
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        # 计算交易成本
        costs = weight_changes * transaction_costs
        # 从组合收益中减去交易成本
        portfolio_returns = portfolio_returns - costs
    
    # 初始化结果字典
    results = {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': (1 + portfolio_returns).cumprod() - 1
    }
    
    # 计算滚动波动率（例如，30天波动率）
    results['rolling_volatility'] = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
    
    # 计算滚动夏普比率
    excess_returns = portfolio_returns - risk_free_rate / 252  # 假设daily
    results['rolling_sharpe'] = excess_returns.rolling(window=60).mean() / excess_returns.rolling(window=60).std() * np.sqrt(252)
    
    # 计算滚动回撤
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.rolling(window=252, min_periods=1).max()
    results['drawdowns'] = (cumulative_returns / rolling_max) - 1
    
    # 计算滚动Beta（如果提供了基准）
    if benchmark_returns is not None:
        # 确保基准与投资组合收益率索引一致
        common_index_benchmark = portfolio_returns.index.intersection(benchmark_returns.index)
        
        if len(common_index_benchmark) > 0:
            aligned_portfolio = portfolio_returns.loc[common_index_benchmark]
            aligned_benchmark = benchmark_returns.loc[common_index_benchmark]
            
            # 计算滚动Beta
            cov = aligned_portfolio.rolling(window=60).cov(aligned_benchmark)
            var = aligned_benchmark.rolling(window=60).var()
            results['rolling_beta'] = cov / var
            
            # 计算滚动相对表现
            results['relative_performance'] = (1 + aligned_portfolio).cumprod() / (1 + aligned_benchmark).cumprod()
    
    return results


def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    计算投资组合换手率
    
    参数:
        weights: 投资组合权重DataFrame，索引为时间，列为资产
        
    返回:
        每个时间点的换手率序列
    """
    # 计算每个时间点的权重变化绝对值之和的一半
    turnover = weights.diff().abs().sum(axis=1) / 2
    
    # 填充第一个值为0（因为没有前一个权重可以比较）
    turnover.iloc[0] = 0
    
    return turnover


def calculate_information_coefficient_decay(signals: pd.DataFrame, 
                                           returns: pd.Series,
                                           max_periods: int = 60,
                                           method: str = 'spearman') -> pd.DataFrame:
    """
    计算信息系数衰减曲线
    
    参数:
        signals: 信号DataFrame
        returns: 收益率Series
        max_periods: 最大预测期数
        method: 相关性计算方法
        
    返回:
        每个信号随时间的IC衰减曲线
    """
    # 初始化结果DataFrame
    decay_curves = pd.DataFrame(index=range(1, max_periods + 1), columns=signals.columns)
    
    # 对每个预测周期计算IC
    for period in range(1, max_periods + 1):
        ic = calculate_ic(signals, returns, method=method, periods=[period])
        decay_curves.loc[period] = ic.iloc[0]
    
    return decay_curves


def calculate_maximum_adverse_excursion(returns: pd.Series, positions: pd.Series) -> pd.Series:
    """
    计算最大不利偏离（每笔交易从开仓到平仓期间的最大不利变动）
    
    参数:
        returns: 收益率序列
        positions: 持仓序列
        
    返回:
        每笔交易的最大不利偏离序列
    """
    # 识别交易（持仓变为非零或从非零变为零）
    trade_entries = (positions != 0) & (positions.shift(1) == 0)
    trade_exits = (positions == 0) & (positions.shift(1) != 0)
    
    # 初始化结果
    mae_values = []
    trade_returns = []
    
    # 当前交易的开始索引
    current_entry = None
    current_position = 0
    
    # 遍历持仓序列
    for i, (idx, pos) in enumerate(positions.items()):
        # 检测新的开仓
        if trade_entries.iloc[i]:
            current_entry = idx
            current_position = pos
        
        # 如果有开仓，计算当前的累积收益
        if current_entry is not None:
            # 获取从开仓到当前的收益序列
            trade_period = returns.loc[current_entry:idx]
            
            # 计算累积收益
            cum_return = (1 + trade_period * current_position).cumprod() - 1
            
            # 交易结束时记录最大不利偏离和总收益
            if trade_exits.iloc[i]:
                mae = cum_return.min() if current_position > 0 else cum_return.max()
                total_return = cum_return.iloc[-1]
                
                mae_values.append(mae)
                trade_returns.append(total_return)
                
                # 重置当前交易
                current_entry = None
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'mae': mae_values,
        'return': trade_returns
    })
    
    return results


# 实现简单版本的替代函数
def annual_return(returns, period='daily', annualization=None):
    """计算年化收益率"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    return (1 + returns.mean()) ** annualization - 1

def sharpe_ratio(returns, risk_free=0.0, period='daily', annualization=None):
    """计算夏普比率"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    excess_returns = returns - risk_free / annualization
    return excess_returns.mean() / excess_returns.std() * np.sqrt(annualization)

def sortino_ratio(returns, required_return=0.0, period='daily', annualization=None):
    """计算索提诺比率"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    excess_returns = returns - required_return / annualization
    downside_returns = excess_returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_risk = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(annualization)
    
    if downside_risk == 0:
        return np.nan
    
    return excess_returns.mean() * annualization / downside_risk

def calmar_ratio(returns, period='daily', annualization=None):
    """计算卡尔玛比率"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    ann_return = annual_return(returns, period, annualization)
    max_dd = max_drawdown(returns)
    
    if max_dd == 0:
        return np.nan
    
    return ann_return / abs(max_dd)

def omega_ratio(returns, risk_free=0.0, required_return=0.0, period='daily', annualization=None):
    """计算欧米茄比率"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    threshold = required_return / annualization
    
    returns_less_thresh = returns - threshold
    numer = returns_less_thresh[returns_less_thresh > 0].sum()
    denom = -returns_less_thresh[returns_less_thresh < 0].sum()
    
    if denom == 0:
        return np.nan
    
    return numer / denom

def alpha_beta(returns, factor_returns, risk_free=0.0, period='daily', annualization=None):
    """计算alpha和beta"""
    if annualization is None:
        if period == 'daily':
            annualization = 252
        elif period == 'weekly':
            annualization = 52
        elif period == 'monthly':
            annualization = 12
        elif period == 'quarterly':
            annualization = 4
        elif period == 'annual':
            annualization = 1
        else:
            annualization = 252
    
    # 确保索引一致
    returns = returns.copy()
    factor_returns = factor_returns.copy()
    common_index = returns.index.intersection(factor_returns.index)
    returns = returns.loc[common_index]
    factor_returns = factor_returns.loc[common_index]
    
    # 计算超额收益
    excess_returns = returns - risk_free / annualization
    excess_factor_returns = factor_returns - risk_free / annualization
    
    # 计算 beta (协方差/方差)
    beta = np.cov(excess_returns, excess_factor_returns)[0, 1] / np.var(excess_factor_returns)
    
    # 计算 alpha (年化)
    alpha = (excess_returns.mean() - beta * excess_factor_returns.mean()) * annualization
    
    return alpha, beta

def max_drawdown(returns):
    """计算最大回撤"""
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns / peak) - 1
    return drawdown.min()

def downside_risk(returns, required_return=0.0):
    """计算下行风险"""
    diff = returns - required_return
    diff[diff > 0] = 0
    return np.sqrt(np.mean(diff ** 2))

# 使用示例
if __name__ == "__main__":
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    positions = pd.Series(np.sign(np.random.normal(0, 1, 252)), index=dates)
    
    # 评估策略
    metrics = evaluate_strategy(returns, positions)
    
    print("策略评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 创建模拟信号数据
    signals = pd.DataFrame({
        'signal1': np.random.normal(0, 1, 252),
        'signal2': np.random.normal(0, 1, 252),
        'signal3': np.random.normal(0, 1, 252)
    }, index=dates)
    
    # 计算信息系数
    ic = calculate_ic(signals, returns)
    print("\n信息系数:")
    print(ic)
    
    # 计算统计显著性
    significance = calculate_statistical_significance(returns * positions)
    print("\n统计显著性测试:")
    for test, results in significance.items():
        print(f"\n{test}:")
        for key, value in results.items():
            print(f"  {key}: {value}") 