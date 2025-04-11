#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
头寸规模管理模块

提供多种头寸规模计算和风险控制策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
from scipy import stats


def fixed_position_size(capital: float, 
                       percentage: float = 0.1) -> float:
    """
    计算固定比例头寸规模
    
    参数:
        capital: 当前资本金额
        percentage: 每笔交易使用的资本百分比
        
    返回:
        交易金额
    """
    return capital * percentage


def fixed_risk_position_size(capital: float,
                           risk_percentage: float,
                           stop_loss_percentage: float) -> float:
    """
    基于固定风险的头寸规模
    
    参数:
        capital: 当前资本金额
        risk_percentage: 每笔交易的风险百分比（资本的百分比）
        stop_loss_percentage: 止损点距离入场点的百分比
        
    返回:
        交易金额
    """
    if stop_loss_percentage <= 0:
        raise ValueError("止损百分比必须大于零")
    
    risk_amount = capital * risk_percentage
    position_size = risk_amount / stop_loss_percentage
    
    return position_size


def kelly_position_size(win_rate: float,
                       win_loss_ratio: float,
                       fraction: float = 1.0) -> float:
    """
    使用凯利公式计算最优头寸规模
    
    参数:
        win_rate: 获胜概率 (0-1)
        win_loss_ratio: 平均盈利/平均亏损比率
        fraction: 调整系数，通常使用半凯利 (0.5) 来降低风险
        
    返回:
        资本的最优配置比例
    """
    if win_rate <= 0 or win_rate >= 1:
        raise ValueError("获胜率必须在0到1之间")
    
    if win_loss_ratio <= 0:
        raise ValueError("盈亏比必须大于零")
    
    kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    # 防止负值（当数学期望为负时）
    kelly_percentage = max(0, kelly_percentage)
    
    # 应用调整系数
    kelly_percentage *= fraction
    
    return kelly_percentage


def adaptive_position_size(capital: float,
                         base_risk: float,
                         volatility: float,
                         volatility_lookback: float,
                         stop_loss_percentage: float,
                         max_risk: float = 0.05) -> float:
    """
    基于波动率自适应的头寸规模
    
    参数:
        capital: 当前资本金额
        base_risk: 基础风险百分比
        volatility: 当前波动率
        volatility_lookback: 历史平均波动率
        stop_loss_percentage: 止损点距离入场点的百分比
        max_risk: 最大允许风险百分比
        
    返回:
        交易金额
    """
    # 根据当前波动率相对于历史波动率的变化调整风险
    volatility_ratio = volatility_lookback / volatility if volatility > 0 else 1.0
    
    # 调整风险百分比
    adjusted_risk = min(base_risk * volatility_ratio, max_risk)
    
    # 计算头寸规模
    risk_amount = capital * adjusted_risk
    position_size = risk_amount / stop_loss_percentage
    
    return position_size


def optimal_f_position_size(trades_history: pd.Series, 
                           fraction: float = 0.5) -> float:
    """
    使用Optimal F方法计算头寸规模
    
    参数:
        trades_history: 历史交易收益率序列
        fraction: 调整系数，用于降低理论上的最优值带来的风险
        
    返回:
        资本的最优配置比例
    """
    if trades_history.empty:
        return 0.0
    
    # 找出最大亏损交易（以比例表示）
    worst_loss = trades_history.min()
    
    if worst_loss >= 0:
        # 如果没有亏损交易，采用保守值
        return 0.1 * fraction
    
    # 计算 Optimal F
    optimal_f = 1.0 / (-worst_loss)
    
    # 应用调整系数
    optimal_f *= fraction
    
    # 限制最大值
    optimal_f = min(optimal_f, 0.5)
    
    return optimal_f


def volatility_adjusted_position_size(capital: float,
                                    target_volatility: float,
                                    current_volatility: float,
                                    max_position_percentage: float = 0.5) -> float:
    """
    基于波动率目标的头寸规模
    
    参数:
        capital: 当前资本金额
        target_volatility: 目标波动率（年化）
        current_volatility: 当前波动率（年化）
        max_position_percentage: 最大头寸比例限制
        
    返回:
        交易金额
    """
    if current_volatility <= 0:
        return 0.0
    
    # 计算波动率调整系数
    volatility_ratio = target_volatility / current_volatility
    
    # 计算头寸比例
    position_percentage = min(volatility_ratio, max_position_percentage)
    
    # 计算头寸金额
    position_size = capital * position_percentage
    
    return position_size


def dynamic_pyramiding(base_position: float,
                      profit_percentage: float,
                      max_pyramids: int = 3,
                      reduction_factor: float = 0.7) -> List[float]:
    """
    计算动态金字塔交易头寸序列
    
    参数:
        base_position: 初始头寸大小
        profit_percentage: 触发添加头寸的利润百分比
        max_pyramids: 最大金字塔层数
        reduction_factor: 每层头寸减少的系数
        
    返回:
        添加头寸大小的列表
    """
    pyramid_positions = [base_position]
    
    for i in range(1, max_pyramids):
        # 计算下一层头寸大小
        next_position = base_position * (reduction_factor ** i)
        pyramid_positions.append(next_position)
    
    return pyramid_positions


def calculate_position_size(capital: float,
                          risk_model: str,
                          stop_loss_percentage: float,
                          params: Dict[str, float] = None) -> float:
    """
    根据选定的风险模型计算头寸规模
    
    参数:
        capital: 当前资本金额
        risk_model: 风险模型名称 ('fixed', 'fixed_risk', 'kelly', 'optimal_f', 'volatility')
        stop_loss_percentage: 止损点距离入场点的百分比
        params: 模型特定参数字典
        
    返回:
        交易金额
    """
    if params is None:
        params = {}
        
    if risk_model == 'fixed':
        percentage = params.get('percentage', 0.1)
        return fixed_position_size(capital, percentage)
    
    elif risk_model == 'fixed_risk':
        risk_percentage = params.get('risk_percentage', 0.01)
        return fixed_risk_position_size(capital, risk_percentage, stop_loss_percentage)
    
    elif risk_model == 'kelly':
        win_rate = params.get('win_rate', 0.5)
        win_loss_ratio = params.get('win_loss_ratio', 1.0)
        fraction = params.get('fraction', 0.5)
        kelly_pct = kelly_position_size(win_rate, win_loss_ratio, fraction)
        return capital * kelly_pct
    
    elif risk_model == 'optimal_f':
        trades_history = params.get('trades_history', pd.Series([]))
        fraction = params.get('fraction', 0.5)
        optimal_f_pct = optimal_f_position_size(trades_history, fraction)
        return capital * optimal_f_pct
    
    elif risk_model == 'volatility':
        target_volatility = params.get('target_volatility', 0.15)
        current_volatility = params.get('current_volatility', 0.20)
        max_position_percentage = params.get('max_position_percentage', 0.5)
        return volatility_adjusted_position_size(capital, target_volatility, current_volatility, max_position_percentage)
    
    else:
        raise ValueError(f"未知的风险模型: {risk_model}")


def position_sizer_factory(risk_model: str, 
                          default_params: Dict[str, float] = None) -> Callable:
    """
    创建特定风险模型的头寸规模计算函数
    
    参数:
        risk_model: 风险模型名称
        default_params: 默认参数字典
        
    返回:
        头寸规模计算函数
    """
    if default_params is None:
        default_params = {}
    
    def sizer(capital: float, stop_loss_percentage: float, **kwargs):
        # 合并默认参数和传入的参数
        params = default_params.copy()
        params.update(kwargs)
        
        return calculate_position_size(capital, risk_model, stop_loss_percentage, params)
    
    return sizer


def calculate_risk_of_ruin(win_rate: float, 
                         risk_per_trade: float, 
                         trades: int = 100) -> float:
    """
    计算破产风险
    
    参数:
        win_rate: 获胜概率 (0-1)
        risk_per_trade: 每笔交易的风险百分比
        trades: 交易次数
        
    返回:
        破产概率
    """
    if win_rate <= 0 or win_rate >= 1:
        raise ValueError("获胜率必须在0到1之间")
    
    if risk_per_trade <= 0 or risk_per_trade >= 1:
        raise ValueError("每笔交易风险必须在0到1之间")
    
    # 简化的破产风险计算（假设固定赢亏比为1）
    # R = (1-W/L)^N，其中W是获胜概率，L是失败概率，N是交易次数
    # 对于1:1的赢亏比，每次交易预期值为：(win_rate * 1) - ((1-win_rate) * 1)
    
    lose_rate = 1 - win_rate
    
    if win_rate > lose_rate:
        # 使用更精确的破产风险公式
        q = lose_rate / win_rate
        risk = (q ** (1 / risk_per_trade)) ** trades
    else:
        # 如果数学期望为负，破产几乎是确定的
        risk = 1.0 - 0.00001 * trades
    
    return min(risk, 1.0)


def calculate_maximal_drawdown_risk(position_sizes: Union[float, np.ndarray, pd.Series],
                                   volatility: float,
                                   correlation_matrix: Optional[np.ndarray] = None,
                                   confidence_level: float = 0.95,
                                   time_horizon: int = 20) -> float:
    """
    计算特定持仓配置的最大回撤风险
    
    参数:
        position_sizes: 各资产头寸大小（单一浮点数或数组）
        volatility: 资产波动率（单一资产）或波动率数组（多资产）
        correlation_matrix: 资产间相关性矩阵（对于多资产）
        confidence_level: VaR的置信水平
        time_horizon: 时间范围（交易日）
        
    返回:
        预估最大回撤风险（占资本的百分比）
    """
    # 转换position_sizes为数组
    if isinstance(position_sizes, (float, int)):
        position_sizes = np.array([position_sizes])
    elif isinstance(position_sizes, pd.Series):
        position_sizes = position_sizes.values
    
    # 转换volatility为数组
    if isinstance(volatility, (float, int)):
        volatility = np.array([volatility])
    
    # 对于单一资产情况
    if len(position_sizes) == 1:
        # 计算日波动率对应的VaR
        z_score = stats.norm.ppf(confidence_level)
        daily_var = position_sizes[0] * volatility[0] * z_score
        
        # 基于平方根规则估算时间范围内的VaR
        period_var = daily_var * np.sqrt(time_horizon)
        
        # 估算最大回撤（通常VaR的1.5-2倍）
        max_drawdown = period_var * 1.8
        
        # 转换为占总资本的百分比
        max_drawdown_pct = max_drawdown / np.sum(position_sizes)
        
        return max_drawdown_pct
    
    # 对于多资产情况
    else:
        if correlation_matrix is None:
            # 如果没有提供相关性矩阵，假设资产间不相关
            correlation_matrix = np.eye(len(position_sizes))
        
        # 计算波动率协方差矩阵
        volatility_diag = np.diag(volatility)
        covariance_matrix = volatility_diag @ correlation_matrix @ volatility_diag
        
        # 计算投资组合波动率
        portfolio_variance = position_sizes.T @ covariance_matrix @ position_sizes
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 计算投资组合VaR
        z_score = stats.norm.ppf(confidence_level)
        daily_var = portfolio_volatility * z_score
        
        # 基于平方根规则估算时间范围内的VaR
        period_var = daily_var * np.sqrt(time_horizon)
        
        # 估算最大回撤
        max_drawdown = period_var * 1.8
        
        # 转换为占总资本的百分比
        max_drawdown_pct = max_drawdown / np.sum(position_sizes)
        
        return max_drawdown_pct


def risk_adjusted_trade_sizing(capital: float,
                             entry_price: float,
                             stop_loss_price: float,
                             win_rate: float,
                             profit_factor: float,
                             risk_tolerance: float = 0.01,
                             volatility_factor: float = 1.0) -> Dict[str, float]:
    """
    综合风险调整的交易规模计算
    
    参数:
        capital: 当前资本金额
        entry_price: 入场价格
        stop_loss_price: 止损价格
        win_rate: 获胜概率 (0-1)
        profit_factor: 盈亏比
        risk_tolerance: 风险容忍度（资本百分比）
        volatility_factor: 波动率调整因子
        
    返回:
        包含各种指标的字典
    """
    # 计算止损距离百分比
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("价格必须大于零")
    
    is_long = entry_price > stop_loss_price
    
    if is_long:
        stop_distance_pct = (entry_price - stop_loss_price) / entry_price
    else:
        stop_distance_pct = (stop_loss_price - entry_price) / entry_price
    
    # 基于凯利公式计算最优头寸规模
    kelly_pct = kelly_position_size(win_rate, profit_factor, 0.5)  # 使用半凯利
    
    # 固定风险头寸规模
    fixed_risk_pos = fixed_risk_position_size(capital, risk_tolerance, stop_distance_pct)
    
    # 计算破产风险
    ruin_risk = calculate_risk_of_ruin(win_rate, risk_tolerance, 100)
    
    # 综合考虑各种因素，计算最终头寸规模
    # 1. 基于固定风险的基础头寸
    position_size = fixed_risk_pos
    
    # 2. 使用凯利比例作为上限
    max_kelly_position = capital * kelly_pct
    position_size = min(position_size, max_kelly_position)
    
    # 3. 考虑波动率因子
    position_size = position_size * volatility_factor
    
    # 4. 如果破产风险高，进一步减少头寸
    if ruin_risk > 0.1:  # 破产风险大于10%
        risk_adjustment = 1.0 - (ruin_risk - 0.1) * 5  # 随着风险增加而线性减少
        risk_adjustment = max(0.2, risk_adjustment)  # 至少保留20%
        position_size *= risk_adjustment
    
    # 计算实际交易单位
    units = position_size / entry_price
    
    # 返回结果
    return {
        'position_size': position_size,
        'units': units,
        'capital_percentage': position_size / capital,
        'kelly_percentage': kelly_pct,
        'stop_distance_percentage': stop_distance_pct,
        'ruin_risk': ruin_risk
    }


def create_portfolio_position_sizes(capital: float,
                                  volatilities: np.ndarray,
                                  correlation_matrix: np.ndarray,
                                  target_portfolio_volatility: float = 0.15,
                                  min_weight: float = 0.05,
                                  max_weight: float = 0.30) -> np.ndarray:
    """
    创建波动率平价的投资组合头寸规模
    
    参数:
        capital: 总资本金额
        volatilities: 各资产年化波动率数组
        correlation_matrix: 资产间相关性矩阵
        target_portfolio_volatility: 目标投资组合年化波动率
        min_weight: 单个资产的最小权重
        max_weight: 单个资产的最大权重
        
    返回:
        资产头寸金额数组
    """
    n_assets = len(volatilities)
    
    # 初始权重 - 基于风险平价
    initial_weights = 1.0 / (volatilities * n_assets)
    
    # 标准化权重使其总和为1
    initial_weights = initial_weights / np.sum(initial_weights)
    
    # 应用权重限制
    constrained_weights = np.clip(initial_weights, min_weight, max_weight)
    
    # 重新标准化
    constrained_weights = constrained_weights / np.sum(constrained_weights)
    
    # 计算初始投资组合波动率
    volatility_diag = np.diag(volatilities)
    covariance_matrix = volatility_diag @ correlation_matrix @ volatility_diag
    portfolio_variance = constrained_weights.T @ covariance_matrix @ constrained_weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # 调整规模以达到目标波动率
    scaling_factor = target_portfolio_volatility / portfolio_volatility
    
    # 计算最终头寸规模
    position_sizes = capital * constrained_weights * scaling_factor
    
    return position_sizes


# 使用示例
if __name__ == "__main__":
    # 示例资本
    capital = 100000.0
    
    # 计算固定比例头寸规模
    fixed_pos = fixed_position_size(capital, 0.1)
    print(f"固定比例头寸规模: {fixed_pos:.2f}")
    
    # 计算固定风险头寸规模
    fixed_risk_pos = fixed_risk_position_size(capital, 0.02, 0.05)
    print(f"固定风险头寸规模: {fixed_risk_pos:.2f}")
    
    # 凯利公式头寸规模
    kelly_pct = kelly_position_size(0.55, 1.5, 0.5)
    kelly_pos = capital * kelly_pct
    print(f"凯利头寸规模: {kelly_pos:.2f} ({kelly_pct:.2%})")
    
    # 综合风险调整的交易规模
    risk_adj = risk_adjusted_trade_sizing(
        capital=capital,
        entry_price=50.0,
        stop_loss_price=47.5,
        win_rate=0.55,
        profit_factor=1.5,
        risk_tolerance=0.01
    )
    
    print("\n综合风险调整的交易规模:")
    for key, value in risk_adj.items():
        print(f"  {key}: {value:.4f}") 