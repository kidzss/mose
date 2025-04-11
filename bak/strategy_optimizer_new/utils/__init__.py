#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具模块

提供各种信号组合和模型训练所需的工具函数
"""

from strategy_optimizer.utils.normalization import normalize_signals, normalize_features
from strategy_optimizer.utils.evaluation import evaluate_strategy, plot_strategy_performance, calculate_ic, plot_ic_heatmap
from strategy_optimizer.utils.signal_optimizer import SignalOptimizer
from strategy_optimizer.utils.technical_indicators import calculate_technical_indicators

# 导入增强的时间序列交叉验证模块
from strategy_optimizer.utils.time_series_cv import (
    TimeSeriesCV, 
    BlockingTimeSeriesCV, 
    NestedTimeSeriesCV,
    walk_forward_validation,
    multi_horizon_validation,
    regime_based_validation
)

# 导入增强的策略评估模块
from strategy_optimizer.utils.enhanced_evaluation import (
    evaluate_strategy,
    calculate_drawdowns,
    calculate_max_drawdown_duration,
    calculate_ic,
    calculate_statistical_significance,
    evaluate_portfolio,
    calculate_turnover,
    calculate_information_coefficient_decay,
    calculate_maximum_adverse_excursion
)

# 导入头寸规模管理模块
from strategy_optimizer.utils.position_sizing import (
    fixed_position_size,
    fixed_risk_position_size,
    kelly_position_size,
    adaptive_position_size,
    optimal_f_position_size,
    volatility_adjusted_position_size,
    dynamic_pyramiding,
    calculate_position_size,
    position_sizer_factory,
    calculate_risk_of_ruin,
    calculate_maximal_drawdown_risk,
    risk_adjusted_trade_sizing,
    create_portfolio_position_sizes
)

# 导入数据生成器模块
from strategy_optimizer.utils.data_generator import DataGenerator

__all__ = [
    # 标准化功能
    "normalize_signals",
    "normalize_features",
    
    # 基本评估功能
    "evaluate_strategy",
    "plot_strategy_performance",
    "calculate_ic",
    "plot_ic_heatmap",
    
    # 信号优化器
    "SignalOptimizer",
    
    # 技术指标
    "calculate_technical_indicators",
    
    # 增强的时间序列交叉验证
    "TimeSeriesCV",
    "BlockingTimeSeriesCV", 
    "NestedTimeSeriesCV",
    "walk_forward_validation",
    "multi_horizon_validation",
    "regime_based_validation",
    
    # 增强的策略评估
    "evaluate_strategy",
    "calculate_drawdowns",
    "calculate_max_drawdown_duration",
    "calculate_ic",
    "calculate_statistical_significance",
    "evaluate_portfolio",
    "calculate_turnover",
    "calculate_information_coefficient_decay",
    "calculate_maximum_adverse_excursion",
    
    # 头寸规模管理
    "fixed_position_size",
    "fixed_risk_position_size",
    "kelly_position_size",
    "adaptive_position_size",
    "optimal_f_position_size",
    "volatility_adjusted_position_size",
    "dynamic_pyramiding",
    "calculate_position_size",
    "position_sizer_factory",
    "calculate_risk_of_ruin",
    "calculate_maximal_drawdown_risk",
    "risk_adjusted_trade_sizing",
    "create_portfolio_position_sizes",
    
    # 数据生成器
    "DataGenerator"
] 