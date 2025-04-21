import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .volatility_manager import VolatilityManager, VolatilityMetrics
import itertools

@dataclass
class PositionInfo:
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    holding_days: int
    stop_loss: float
    take_profit: float

@dataclass
class RiskMetrics:
    value_at_risk: float  # 在险价值
    max_drawdown: float  # 最大回撤
    volatility: float  # 波动率
    beta: float  # 贝塔系数
    correlation: float  # 与基准的相关性
    position_exposure: float  # 持仓敞口

class RiskManager:
    """风险管理器"""
    
    def __init__(self):
        """初始化风险管理器"""
        self.logger = logging.getLogger(__name__)
        self.volatility_manager = VolatilityManager()
        
    def evaluate_risk(self, returns: pd.Series, positions: pd.Series) -> Dict:
        """
        评估风险
        
        参数:
            returns: 收益率序列
            positions: 持仓序列
            
        返回:
            风险指标字典
        """
        try:
            if returns.empty or positions.empty:
                return {
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'value_at_risk': 0.0
                }
                
            # 计算波动率
            volatility = returns.std() * np.sqrt(252)
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # 计算最大回撤
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # 计算VaR
            value_at_risk = np.percentile(returns, 5)  # 95% VaR
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'value_at_risk': value_at_risk
            }
            
        except Exception as e:
            self.logger.error(f"评估风险时发生错误: {str(e)}")
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'value_at_risk': 0.0
            }
            
    def check_risk_limits(self, risk_metrics: Dict, limits: Dict) -> Dict:
        """
        检查风险限制
        
        参数:
            risk_metrics: 风险指标字典
            limits: 风险限制字典
            
        返回:
            风险检查结果
        """
        try:
            results = {}
            
            # 检查波动率限制
            if 'volatility_limit' in limits:
                results['volatility_breach'] = risk_metrics['volatility'] > limits['volatility_limit']
                
            # 检查最大回撤限制
            if 'max_drawdown_limit' in limits:
                results['max_drawdown_breach'] = risk_metrics['max_drawdown'] > limits['max_drawdown_limit']
                
            # 检查VaR限制
            if 'var_limit' in limits:
                results['var_breach'] = abs(risk_metrics['value_at_risk']) > limits['var_limit']
                
            return results
            
        except Exception as e:
            self.logger.error(f"检查风险限制时发生错误: {str(e)}")
            return {}

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        计算建仓数量，考虑波动率调整
        """
        # 获取价格数据
        prices = self.get_historical_prices(symbol)  # 假设这个方法已经存在
        
        # 计算波动率指标
        volatility_metrics = self.volatility_manager.calculate_volatility_metrics(prices)
        
        # 计算基础仓位
        base_position = (self.initial_capital * risk_per_trade) / price
        
        # 根据波动率调整仓位
        adjusted_position = self.volatility_manager.adjust_position_size(
            base_position,
            volatility_metrics
        )
        
        return adjusted_position
        
    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        trailing_percent: float = 0.05
    ) -> float:
        """
        计算跟踪止损价格
        
        参数:
            current_price: 当前价格
            highest_price: 持仓期间最高价格
            trailing_percent: 跟踪止损百分比
            
        返回:
            stop_price: 止损价格
        """
        # 跟踪止损会随着价格上涨而上移
        stop_price = highest_price * (1 - trailing_percent)
        return max(stop_price, current_price * 0.95)  # 不低于固定止损
        
    def calculate_atr_stop(
        self,
        current_price: float,
        atr: float,
        multiplier: float = 2.0,
        direction: str = 'long'
    ) -> float:
        """
        基于ATR的止损价格计算
        
        参数:
            current_price: 当前价格
            atr: 平均真实波幅
            multiplier: ATR乘数
            direction: 交易方向 ('long' 或 'short')
            
        返回:
            stop_price: 止损价格
        """
        if direction == 'long':
            return current_price - (atr * multiplier)
        else:
            return current_price + (atr * multiplier)
            
    def calculate_time_based_stop(
        self,
        entry_price: float,
        holding_days: int,
        max_holding_days: int = 20,
        min_profit_threshold: float = 0.02
    ) -> Tuple[bool, float]:
        """
        基于时间的止损策略
        
        参数:
            entry_price: 入场价格
            holding_days: 已持有天数
            max_holding_days: 最大持有天数
            min_profit_threshold: 最小利润阈值
            
        返回:
            (should_stop: 是否应该止损, stop_price: 止损价格)
        """
        # 时间止损：持仓时间越长，止损越严格
        time_factor = holding_days / max_holding_days
        required_profit = min_profit_threshold * (1 + time_factor)
        stop_price = entry_price * (1 + required_profit)
        
        # 如果超过最大持有时间，直接止损
        should_stop = holding_days >= max_holding_days
        
        return should_stop, stop_price
        
    def calculate_volatility_stop(
        self,
        current_price: float,
        volatility: float,
        z_score: float = 2.0,
        direction: str = 'long'
    ) -> float:
        """
        基于波动率的止损价格计算
        
        参数:
            current_price: 当前价格
            volatility: 波动率
            z_score: 标准差倍数
            direction: 交易方向 ('long' 或 'short')
            
        返回:
            stop_price: 止损价格
        """
        # 使用正态分布的置信区间作为止损点
        if direction == 'long':
            return current_price * (1 - volatility * z_score)
        else:
            return current_price * (1 + volatility * z_score)
            
    def calculate_support_resistance_stop(
        self,
        current_price: float,
        support_level: float,
        resistance_level: float,
        direction: str = 'long'
    ) -> float:
        """
        基于支撑/压力位的止损价格计算
        
        参数:
            current_price: 当前价格
            support_level: 支撑位
            resistance_level: 压力位
            direction: 交易方向 ('long' 或 'short')
            
        返回:
            stop_price: 止损价格
        """
        if direction == 'long':
            return support_level * 0.98  # 略低于支撑位
        else:
            return resistance_level * 1.02  # 略高于压力位
            
    def get_combined_stop_loss(
        self,
        current_price: float,
        highest_price: float,
        atr: float,
        volatility: float,
        holding_days: int,
        entry_price: float,
        support_level: float,
        resistance_level: float,
        direction: str = 'long'
    ) -> float:
        """
        综合多个止损策略，返回最优止损价格
        
        参数:
            current_price: 当前价格
            highest_price: 持仓期间最高价格
            atr: ATR值
            volatility: 波动率
            holding_days: 持有天数
            entry_price: 入场价格
            support_level: 支撑位
            resistance_level: 压力位
            direction: 交易方向
            
        返回:
            stop_price: 最终止损价格
        """
        # 计算各种止损价格
        trailing_stop = self.calculate_trailing_stop(current_price, highest_price)
        atr_stop = self.calculate_atr_stop(current_price, atr, direction=direction)
        _, time_stop = self.calculate_time_based_stop(entry_price, holding_days)
        vol_stop = self.calculate_volatility_stop(current_price, volatility, direction=direction)
        sr_stop = self.calculate_support_resistance_stop(
            current_price, support_level, resistance_level, direction
        )
        
        if direction == 'long':
            # 多头取最高的止损价格（最保守）
            return max(trailing_stop, atr_stop, time_stop, vol_stop, sr_stop)
        else:
            # 空头取最低的止损价格（最保守）
            return min(trailing_stop, atr_stop, time_stop, vol_stop, sr_stop)
        
    def calculate_take_profit(
        self,
        price: float,
        stop_loss: float,
        method: str = 'risk_reward'
    ) -> float:
        """计算止盈价格"""
        risk = price - stop_loss
        if method == 'risk_reward':
            # 基于风险收益比的止盈
            take_profit = price + (risk * 2.0)
        else:
            # 默认使用固定比例
            take_profit = price * 1.1
            
        return take_profit
        
    def check_portfolio_risk(self) -> Tuple[float, bool]:
        """检查组合风险"""
        total_risk = 0
        for pos in self.positions.values():
            # 计算每个持仓的风险暴露
            risk_exposure = abs(pos.current_price - pos.stop_loss) * pos.position_size
            total_risk += risk_exposure
            
        # 计算组合热度
        portfolio_heat = total_risk / self.initial_capital
        
        # 检查是否超过最大风险限制
        is_safe = portfolio_heat <= 0.3
        
        return portfolio_heat, is_safe
        
    def should_close_position(self, position: PositionInfo) -> Tuple[bool, str]:
        """检查是否应该平仓"""
        # 止损检查
        if position.current_price <= position.stop_loss:
            return True, "触发止损"
            
        # 止盈检查
        if position.current_price >= position.take_profit:
            return True, "触发止盈"
            
        # 持仓时间检查
        if position.holding_days > 20:  # 可配置
            return True, "超过最大持仓时间"
            
        # 浮动盈亏检查
        if position.unrealized_pnl < -0.1:  # 可配置
            return True, "超过最大浮动亏损"
            
        return False, ""
        
    def calculate_risk_metrics(
        self,
        symbol: str,
        price: float,
        volatility: float,
        portfolio_value: float
    ) -> RiskMetrics:
        """计算风险指标"""
        # 计算止损价格
        stop_loss = self.get_combined_stop_loss(
            price,
            price,
            0,
            volatility,
            0,
            0,
            0,
            0,
            'long'
        )
        
        # 计算止盈价格
        take_profit = self.calculate_take_profit(price, stop_loss)
        
        # 计算持仓规模
        position_size = self.calculate_position_size(symbol, price)
        
        # 计算最大损失金额
        max_loss = (price - stop_loss) * position_size
        
        # 计算风险收益比
        risk = price - stop_loss
        reward = take_profit - price
        risk_reward_ratio = reward / risk if risk != 0 else 0
        
        # 计算组合热度
        portfolio_heat, _ = self.check_portfolio_risk()
        
        return RiskMetrics(
            value_at_risk=self.calculate_value_at_risk(returns),
            max_drawdown=self.calculate_max_drawdown(returns),
            volatility=self.calculate_volatility(returns),
            beta=self.calculate_beta(returns),
            correlation=self.calculate_correlation(returns),
            position_exposure=self.calculate_position_exposure(positions)
        )
        
    def update_position(
        self,
        symbol: str,
        current_price: float,
        position_size: float = None
    ) -> None:
        """更新持仓信息"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # 更新持仓规模（如果提供）
        if position_size is not None:
            position.position_size = position_size
            
        # 更新当前价格和未实现盈亏
        position.current_price = current_price
        position.unrealized_pnl = (
            (current_price - position.entry_price) / position.entry_price
        )
        
        # 更新持仓天数
        position.holding_days += 1
        
    def generate_risk_report(self, returns: pd.Series, positions: pd.Series) -> str:
        """
        生成详细的风险分析报告
        
        参数:
            returns: 收益率序列
            positions: 持仓序列
            
        返回:
            report: 风险分析报告
        """
        # 计算风险指标
        metrics = self.evaluate_risk(returns, positions)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率2%
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # 计算索提诺比率
        downside_std = returns[returns < 0].std()
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # 计算最大连续亏损天数
        negative_returns = returns < 0
        max_consecutive_losses = max(
            sum(1 for _ in group) for key, group in itertools.groupby(negative_returns) if key
        )
        
        report = f"""
详细风险分析报告
==============

基础风险指标:
- 在险价值(VaR {self.var_confidence_level:.0%}): {metrics.value_at_risk:.2%}
- 条件在险价值(CVaR): {self.calculate_conditional_var(returns):.2%}
- 最大回撤: {metrics.max_drawdown:.2%}
- 波动率(年化): {metrics.volatility:.2%}
- Beta系数: {metrics.beta:.2f}
- 相关系数: {metrics.correlation:.2f}

收益风险比率:
- 夏普比率: {sharpe_ratio:.2f}
- 索提诺比率: {sortino_ratio:.2f}
- 信息比率: {(returns.mean() / returns.std() * np.sqrt(252)):.2f}

持仓风险分析:
- 平均持仓比例: {metrics.position_exposure:.2%}
- 当前总风险敞口: {self.calculate_total_risk_exposure():.2%}
- 最大连续亏损天数: {max_consecutive_losses}天

风险状态评估:
{self._get_var_advice(metrics.value_at_risk)}
{self._get_drawdown_advice(metrics.max_drawdown)}
{self._get_volatility_advice(metrics.volatility)}
{self._get_exposure_advice(metrics.position_exposure)}

建议操作:
{self._generate_risk_recommendations(metrics)}
"""
        return report
        
    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> str:
        """生成风险管理建议"""
        recommendations = []
        
        # VaR建议
        if metrics.value_at_risk > self.base_risk_per_trade * 2:
            recommendations.append("- 建议降低单笔交易风险，考虑减小持仓规模")
            
        # 回撤建议
        if metrics.max_drawdown > self.max_drawdown_limit * 0.8:
            recommendations.append("- 接近最大回撤限制，建议及时止损或减仓")
            
        # 波动率建议
        if metrics.volatility > 0.3:
            recommendations.append("- 市场波动率较高，建议提高止损位置，减少交易频率")
            
        # Beta建议
        if metrics.beta > 1.5:
            recommendations.append("- 组合Beta偏高，建议增加低Beta股票的配置")
        elif metrics.beta < 0.5:
            recommendations.append("- 组合Beta偏低，可以考虑适当增加进攻性")
            
        # 持仓建议
        if metrics.position_exposure > self.max_total_position * 0.9:
            recommendations.append("- 总持仓接近上限，建议控制新开仓位")
            
        if not recommendations:
            recommendations.append("- 当前风险指标处于合理水平，可以维持现有策略")
            
        return "\n".join(recommendations)
        
    def calculate_total_risk_exposure(self) -> float:
        """计算当前总风险敞口"""
        if not self.positions:
            return 0.0
            
        total_exposure = sum(
            pos.position_size * pos.current_price / self.current_capital
            for pos in self.positions.values()
        )
        
        return total_exposure
        
    def calculate_value_at_risk(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算在险价值(VaR)"""
        if returns.empty:
            return 0
        return abs(np.percentile(returns, (1 - confidence_level) * 100))
        
    def calculate_conditional_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        计算条件在险价值(CVaR)
        
        参数:
            returns: 收益率序列
            confidence_level: 置信水平
        """
        var = self.calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
        
    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """计算回撤序列"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
        
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        if returns.empty:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def calculate_volatility(self, returns: pd.Series) -> float:
        """计算波动率"""
        if returns.empty or len(returns) < 2:
            return 0
        return returns.std() * np.sqrt(252)
        
    def calculate_beta(self, returns: pd.Series) -> float:
        """计算贝塔系数"""
        if returns.empty or self.benchmark_data is None:
            return 1.0
            
        benchmark_returns = self.benchmark_data['Close'].pct_change()
        cov = np.cov(returns, benchmark_returns)[0][1]
        benchmark_var = benchmark_returns.var()
        
        return cov / benchmark_var if benchmark_var != 0 else 1.0
        
    def calculate_correlation(self, returns: pd.Series) -> float:
        """计算与基准的相关性"""
        if returns.empty or self.benchmark_data is None:
            return 0.0
            
        benchmark_returns = self.benchmark_data['Close'].pct_change()
        return returns.corr(benchmark_returns)
        
    def calculate_position_exposure(self, positions: pd.Series) -> float:
        """计算持仓敞口"""
        if positions.empty:
            return 0
        return positions.mean()
        
    def evaluate_risk(self, returns: pd.Series, positions: pd.Series) -> RiskMetrics:
        """评估风险指标"""
        return RiskMetrics(
            value_at_risk=self.calculate_value_at_risk(returns),
            max_drawdown=self.calculate_max_drawdown(returns),
            volatility=self.calculate_volatility(returns),
            beta=self.calculate_beta(returns),
            correlation=self.calculate_correlation(returns),
            position_exposure=self.calculate_position_exposure(positions)
        )
        
    def _get_var_advice(self, var: float) -> str:
        """生成VaR建议"""
        if var > 0.03:
            return "- VaR较高，建议降低仓位或使用更严格的止损"
        return "- VaR在可接受范围内"
        
    def _get_drawdown_advice(self, drawdown: float) -> str:
        """生成最大回撤建议"""
        if drawdown > 0.2:
            return "- 最大回撤过大，建议优化止损策略"
        return "- 最大回撤在可控范围内"
        
    def _get_volatility_advice(self, volatility: float) -> str:
        """生成波动率建议"""
        if volatility > 0.3:
            return "- 波动率较高，建议降低交易频率或使用更保守的仓位"
        return "- 波动率在合理范围内"
        
    def _get_exposure_advice(self, exposure: float) -> str:
        """生成持仓建议"""
        if exposure > 0.8:
            return "- 持仓过重，建议适当减仓"
        return "- 持仓水平合理"
        
    def calculate_stop_loss(self, symbol: str, entry_price: float, direction: str = 'long') -> Tuple[float, float]:
        """
        计算止损价格，考虑波动率
        """
        # 获取价格数据
        prices = self.get_historical_prices(symbol)
        
        # 计算波动率指标
        volatility_metrics = self.volatility_manager.calculate_volatility_metrics(prices)
        
        # 计算基于波动率的止损止盈
        stop_loss, take_profit = self.volatility_manager.calculate_volatility_based_stops(
            entry_price,
            volatility_metrics,
            direction
        )
        
        return stop_loss, take_profit
        
    def update_risk_metrics(self, symbol: str) -> Dict:
        """
        更新风险指标
        """
        # 获取价格数据
        prices = self.get_historical_prices(symbol)
        
        # 计算波动率指标
        volatility_metrics = self.volatility_manager.calculate_volatility_metrics(prices)
        
        # 计算仓位乘数
        position_multiplier = self.volatility_manager.calculate_position_multiplier(volatility_metrics)
        
        # 生成风险报告
        risk_report = self.volatility_manager.generate_volatility_report(
            volatility_metrics,
            position_multiplier
        )
        
        return {
            'volatility_metrics': volatility_metrics,
            'position_multiplier': position_multiplier,
            'risk_report': risk_report
        }

    def get_historical_prices(self, symbol: str, lookback_days: int = 252) -> pd.Series:
        """
        获取历史价格数据
        
        参数:
            symbol: 股票代码
            lookback_days: 回看天数
            
        返回:
            prices: 价格序列
        """
        if symbol not in self.data.columns:
            raise ValueError(f"数据中不存在股票 {symbol}")
            
        # 获取收盘价序列
        prices = self.data[symbol].tail(lookback_days)
        
        # 确保数据完整性
        if len(prices) < lookback_days / 2:  # 允许一定的数据缺失
            raise ValueError(f"股票 {symbol} 的历史数据不足")
            
        return prices 

    def calculate_atr(
        self,
        symbol: str,
        period: int = 14
    ) -> float:
        """
        计算ATR（平均真实波幅）
        
        参数:
            symbol: 股票代码
            period: ATR计算周期
            
        返回:
            atr: ATR值
        """
        # 获取价格数据
        high = self.data[f"{symbol}_high"].tail(period + 1)
        low = self.data[f"{symbol}_low"].tail(period + 1)
        close = self.data[f"{symbol}_close"].tail(period + 1)
        
        # 计算真实波幅
        tr1 = high - low  # 当日高低价差
        tr2 = abs(high - close.shift(1))  # 当日高点与前收差
        tr3 = abs(low - close.shift(1))  # 当日低点与前收差
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr 