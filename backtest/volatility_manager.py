import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VolatilityMetrics:
    current_volatility: float  # 当前波动率
    historical_volatility: float  # 历史波动率
    relative_volatility: float  # 相对波动率
    volatility_trend: str  # 波动率趋势
    risk_level: str  # 风险等级

class VolatilityManager:
    def __init__(
        self,
        lookback_period: int = 20,  # 回看期
        volatility_threshold: float = 0.02,  # 波动率阈值
        max_position_multiplier: float = 2.0  # 最大仓位倍数
    ):
        """
        初始化波动率管理器
        
        参数:
            lookback_period: 计算波动率的回看期
            volatility_threshold: 波动率阈值
            max_position_multiplier: 最大仓位倍数
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.max_position_multiplier = max_position_multiplier
        
    def calculate_volatility_metrics(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None
    ) -> VolatilityMetrics:
        """
        计算波动率指标
        
        参数:
            prices: 价格序列
            returns: 收益率序列（可选）
            
        返回:
            volatility_metrics: 波动率指标
        """
        if returns is None:
            returns = prices.pct_change()
            
        # 计算当前波动率（最近N天）
        current_volatility = returns[-self.lookback_period:].std() * np.sqrt(252)
        
        # 计算历史波动率
        historical_volatility = returns.std() * np.sqrt(252)
        
        # 计算相对波动率
        relative_volatility = current_volatility / historical_volatility if historical_volatility != 0 else 1.0
        
        # 判断波动率趋势
        vol_ma_short = returns[-10:].std() * np.sqrt(252)
        vol_ma_long = returns[-30:].std() * np.sqrt(252)
        volatility_trend = "上升" if vol_ma_short > vol_ma_long else "下降"
        
        # 确定风险等级
        risk_level = self._determine_risk_level(current_volatility)
        
        return VolatilityMetrics(
            current_volatility=current_volatility,
            historical_volatility=historical_volatility,
            relative_volatility=relative_volatility,
            volatility_trend=volatility_trend,
            risk_level=risk_level
        )
        
    def _determine_risk_level(self, volatility: float) -> str:
        """
        根据波动率确定风险等级
        
        参数:
            volatility: 波动率
            
        返回:
            risk_level: 风险等级
        """
        if volatility < self.volatility_threshold * 0.5:
            return "低风险"
        elif volatility < self.volatility_threshold:
            return "中低风险"
        elif volatility < self.volatility_threshold * 2:
            return "中风险"
        elif volatility < self.volatility_threshold * 3:
            return "中高风险"
        else:
            return "高风险"
            
    def calculate_position_multiplier(
        self,
        volatility_metrics: VolatilityMetrics
    ) -> float:
        """
        计算仓位乘数
        
        参数:
            volatility_metrics: 波动率指标
            
        返回:
            multiplier: 仓位乘数
        """
        # 基于相对波动率计算基础乘数
        base_multiplier = 1 / volatility_metrics.relative_volatility
        
        # 根据波动率趋势调整
        if volatility_metrics.volatility_trend == "上升":
            base_multiplier *= 0.8  # 波动率上升时降低仓位
        else:
            base_multiplier *= 1.2  # 波动率下降时增加仓位
            
        # 确保乘数在合理范围内
        return max(0.2, min(self.max_position_multiplier, base_multiplier))
        
    def adjust_position_size(
        self,
        base_position: float,
        volatility_metrics: VolatilityMetrics
    ) -> float:
        """
        根据波动率调整仓位大小
        
        参数:
            base_position: 基础仓位
            volatility_metrics: 波动率指标
            
        返回:
            adjusted_position: 调整后的仓位
        """
        # 计算仓位乘数
        multiplier = self.calculate_position_multiplier(volatility_metrics)
        
        # 调整仓位
        adjusted_position = base_position * multiplier
        
        return adjusted_position
        
    def calculate_volatility_based_stops(
        self,
        current_price: float,
        volatility_metrics: VolatilityMetrics,
        direction: str = 'long'
    ) -> Tuple[float, float]:
        """
        基于波动率计算止损止盈价格
        
        参数:
            current_price: 当前价格
            volatility_metrics: 波动率指标
            direction: 交易方向
            
        返回:
            (stop_loss, take_profit): 止损和止盈价格
        """
        # 根据波动率计算价格波动范围
        price_range = current_price * volatility_metrics.current_volatility
        
        if direction == 'long':
            stop_loss = current_price - (price_range * 2)  # 2倍波动率止损
            take_profit = current_price + (price_range * 3)  # 3倍波动率止盈
        else:
            stop_loss = current_price + (price_range * 2)
            take_profit = current_price - (price_range * 3)
            
        return stop_loss, take_profit
        
    def generate_volatility_report(
        self,
        volatility_metrics: VolatilityMetrics,
        position_multiplier: float
    ) -> str:
        """
        生成波动率分析报告
        
        参数:
            volatility_metrics: 波动率指标
            position_multiplier: 仓位乘数
            
        返回:
            report: 分析报告
        """
        report = f"""
波动率分析报告
===========

波动率指标:
- 当前波动率: {volatility_metrics.current_volatility:.2%}
- 历史波动率: {volatility_metrics.historical_volatility:.2%}
- 相对波动率: {volatility_metrics.relative_volatility:.2f}
- 波动率趋势: {volatility_metrics.volatility_trend}
- 风险等级: {volatility_metrics.risk_level}

仓位调整:
- 仓位乘数: {position_multiplier:.2f}

建议:
"""
        # 添加建议
        if volatility_metrics.volatility_trend == "上升":
            report += "- 波动率呈上升趋势，建议降低仓位，收紧止损\n"
        else:
            report += "- 波动率呈下降趋势，可以考虑适度增加仓位\n"
            
        if volatility_metrics.relative_volatility > 1.5:
            report += "- 当前波动率显著高于历史水平，建议保持谨慎\n"
        elif volatility_metrics.relative_volatility < 0.7:
            report += "- 当前波动率低于历史水平，可以考虑更积极的策略\n"
            
        return report 