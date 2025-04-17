import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PositionInfo:
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    holding_days: int
    risk_exposure: float

class PositionManager:
    def __init__(
        self,
        initial_capital: float,
        max_position_size: float = 0.2,  # 单个持仓最大比例
        max_total_position: float = 0.8,  # 最大总仓位
        base_risk_per_trade: float = 0.02  # 每笔交易基础风险
    ):
        """
        初始化仓位管理器
        
        参数:
            initial_capital: 初始资金
            max_position_size: 单个持仓最大比例
            max_total_position: 最大总仓位
            base_risk_per_trade: 每笔交易基础风险
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_total_position = max_total_position
        self.base_risk_per_trade = base_risk_per_trade
        self.positions: Dict[str, PositionInfo] = {}
        
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算凯利公式最优仓位
        
        参数:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        返回:
            kelly_fraction: 最优仓位比例
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
            
        q = 1 - win_rate
        kelly = (win_rate/abs(avg_loss)) - (q/avg_win)
        
        # 通常使用半凯利以降低风险
        return max(0, min(self.max_position_size, kelly * 0.5))
        
    def adjust_position_by_volatility(
        self,
        base_position: float,
        volatility: float,
        avg_volatility: float
    ) -> float:
        """
        根据波动率调整仓位
        
        参数:
            base_position: 基础仓位
            volatility: 当前波动率
            avg_volatility: 平均波动率
            
        返回:
            adjusted_position: 调整后的仓位
        """
        if avg_volatility == 0:
            return base_position
            
        # 波动率比值
        vol_ratio = volatility / avg_volatility
        
        # 波动率越高，仓位越小
        if vol_ratio > 1:
            return base_position / vol_ratio
        else:
            # 波动率低于平均时，可以适当增加仓位，但不超过最大限制
            return min(base_position * (2 - vol_ratio), self.max_position_size)
            
    def adjust_position_by_trend(
        self,
        base_position: float,
        trend_strength: float  # 趋势强度 [-1, 1]
    ) -> float:
        """
        根据趋势强度调整仓位
        
        参数:
            base_position: 基础仓位
            trend_strength: 趋势强度
            
        返回:
            adjusted_position: 调整后的仓位
        """
        # 趋势越强，仓位越大
        trend_factor = 0.5 + (abs(trend_strength) * 0.5)
        return base_position * trend_factor
        
    def adjust_position_by_correlation(
        self,
        base_position: float,
        portfolio_correlation: float
    ) -> float:
        """
        根据相关性调整仓位
        
        参数:
            base_position: 基础仓位
            portfolio_correlation: 与组合的相关性
            
        返回:
            adjusted_position: 调整后的仓位
        """
        # 相关性越高，仓位越小
        correlation_factor = 1 - abs(portfolio_correlation)
        return base_position * (0.5 + correlation_factor * 0.5)
        
    def calculate_dynamic_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        volatility: float,
        avg_volatility: float,
        trend_strength: float,
        portfolio_correlation: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算动态仓位大小
        
        参数:
            symbol: 股票代码
            price: 当前价格
            stop_loss: 止损价格
            volatility: 当前波动率
            avg_volatility: 平均波动率
            trend_strength: 趋势强度
            portfolio_correlation: 与组合的相关性
            win_rate: 历史胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        返回:
            position_size: 仓位大小
        """
        # 1. 计算风险金额
        risk_amount = self.current_capital * self.base_risk_per_trade
        
        # 2. 计算基础仓位
        risk_per_share = abs(price - stop_loss)
        base_position = risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # 3. 使用凯利公式调整
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        position = base_position * kelly_fraction
        
        # 4. 根据波动率调整
        position = self.adjust_position_by_volatility(position, volatility, avg_volatility)
        
        # 5. 根据趋势强度调整
        position = self.adjust_position_by_trend(position, trend_strength)
        
        # 6. 根据相关性调整
        position = self.adjust_position_by_correlation(position, portfolio_correlation)
        
        # 7. 确保不超过最大持仓限制
        max_shares = (self.current_capital * self.max_position_size) / price
        position = min(position, max_shares)
        
        # 8. 检查总仓位限制
        total_exposure = sum(pos.position_size * pos.current_price 
                           for pos in self.positions.values())
        remaining_exposure = self.current_capital * self.max_total_position - total_exposure
        position = min(position, remaining_exposure / price)
        
        return position
        
    def update_position(
        self,
        symbol: str,
        current_price: float,
        stop_loss: float,
        take_profit: float
    ) -> None:
        """
        更新持仓信息
        
        参数:
            symbol: 股票代码
            current_price: 当前价格
            stop_loss: 止损价格
            take_profit: 止盈价格
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            pos.stop_loss = stop_loss
            pos.take_profit = take_profit
            pos.unrealized_pnl = (current_price / pos.entry_price - 1) * 100
            pos.risk_exposure = abs(current_price - stop_loss) * pos.position_size
            
    def get_portfolio_risk(self) -> Tuple[float, bool]:
        """
        获取组合风险状态
        
        返回:
            (total_risk: 总风险, is_safe: 是否安全)
        """
        total_risk = sum(pos.risk_exposure for pos in self.positions.values())
        portfolio_heat = total_risk / self.current_capital
        is_safe = portfolio_heat <= 0.3  # 总风险不超过30%
        
        return portfolio_heat, is_safe 