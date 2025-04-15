import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass

@dataclass
class VolatilityMetrics:
    current_volatility: float  # 当前波动率
    historical_volatility: float  # 历史波动率
    relative_volatility: float  # 相对波动率
    volatility_trend: str  # 波动率趋势
    risk_level: str  # 风险等级

class VolatilityManager:
    """波动率管理器"""
    
    def __init__(self):
        """初始化波动率管理器"""
        self.logger = logging.getLogger(__name__)
        
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """
        计算波动率
        
        参数:
            returns: 收益率序列
            window: 计算窗口
            
        返回:
            波动率
        """
        try:
            if returns.empty or len(returns) < window:
                return 0.0
                
            # 计算滚动波动率
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            
            return volatility.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"计算波动率时发生错误: {str(e)}")
            return 0.0
            
    def adjust_position_size(
        self,
        current_position: float,
        volatility: float,
        target_volatility: float = 0.2,
        max_position: float = 1.0
    ) -> float:
        """
        根据波动率调整仓位
        
        参数:
            current_position: 当前仓位
            volatility: 当前波动率
            target_volatility: 目标波动率
            max_position: 最大仓位
            
        返回:
            调整后的仓位
        """
        try:
            if volatility <= 0:
                return current_position
                
            # 计算波动率调整因子
            vol_ratio = target_volatility / volatility
            
            # 调整仓位
            new_position = current_position * vol_ratio
            
            # 限制最大仓位
            return min(max(0, new_position), max_position)
            
        except Exception as e:
            self.logger.error(f"调整仓位时发生错误: {str(e)}")
            return current_position
            
    def calculate_stop_loss(
        self,
        price: float,
        volatility: float,
        multiplier: float = 2.0,
        min_distance: float = 0.02
    ) -> float:
        """
        计算止损价格
        
        参数:
            price: 当前价格
            volatility: 当前波动率
            multiplier: 波动率乘数
            min_distance: 最小止损距离
            
        返回:
            止损价格
        """
        try:
            # 计算止损距离
            stop_distance = max(volatility * multiplier, min_distance)
            
            # 计算止损价格
            stop_price = price * (1 - stop_distance)
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"计算止损价格时发生错误: {str(e)}")
            return price * 0.95  # 默认5%止损

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
        current_volatility = returns[-20:].std() * np.sqrt(252)
        
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
        if volatility < 0.02 * 0.5:
            return "低风险"
        elif volatility < 0.02:
            return "中低风险"
        elif volatility < 0.02 * 2:
            return "中风险"
        elif volatility < 0.02 * 3:
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
        return max(0.2, min(2.0, base_multiplier))
        
    def calculate_volatility_based_stops(
        self,
        price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        计算基于波动率的止损和止盈价格
        
        参数:
            price: 当前价格
            volatility: 当前波动率
            
        返回:
            (stop_loss, take_profit): 止损和止盈价格
        """
        try:
            # 计算止损距离（2倍波动率）
            stop_distance = volatility * 2
            
            # 计算止盈距离（3倍波动率）
            profit_distance = volatility * 3
            
            # 计算止损和止盈价格
            stop_loss = price * (1 - stop_distance)
            take_profit = price * (1 + profit_distance)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"计算止损止盈价格时发生错误: {str(e)}")
            return price * 0.95, price * 1.05  # 默认5%止损，5%止盈
        
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