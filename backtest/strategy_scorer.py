import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyScorer")

@dataclass
class StrategyScore:
    """策略评分结果"""
    strategy_name: str
    symbol: str
    total_score: float
    profit_score: float     # 盈利能力得分(2分)
    adaptability_score: float  # 适应性得分(2分)
    robustness_score: float    # 稳健性得分(1分)
    details: Dict  # 详细指标数据

class StrategyScorer:
    """
    策略评分系统 - 专注于为单个股票选择最优策略
    
    评分维度:
    1. 盈利能力 (2分) - 评估策略的盈利能力
    2. 适应性 (2分) - 评估策略在不同市场环境下的表现
    3. 稳健性 (1分) - 评估策略的风险控制能力
    """
    
    def __init__(self, market_regime: str = 'normal'):
        """
        初始化评分系统
        
        参数:
            market_regime: 市场环境，可选值: 'normal', 'volatile', 'trending', 'range'
        """
        # 基础权重
        self.base_weights = {
            'profit': 2.0,      # 盈利能力权重
            'adaptability': 2.0, # 适应性权重
            'robustness': 1.0    # 稳健性权重
        }
        
        # 根据市场环境调整权重
        self.market_regime = market_regime
        self.weights = self._adjust_weights_for_market(market_regime)
        
        logger.info(f"StrategyScorer初始化完成，市场环境: {market_regime}, 权重: {self.weights}")
        
    def _adjust_weights_for_market(self, market_regime: str) -> Dict[str, float]:
        """
        根据市场环境调整权重
        
        参数:
            market_regime: 市场环境
            
        返回:
            调整后的权重
        """
        weights = self.base_weights.copy()
        
        if market_regime == 'volatile':
            # 高波动市场，增加稳健性权重
            weights['profit'] = 1.5
            weights['adaptability'] = 1.5
            weights['robustness'] = 2.0
        elif market_regime == 'trending':
            # 趋势市场，增加盈利能力权重
            weights['profit'] = 2.5
            weights['adaptability'] = 1.5
            weights['robustness'] = 1.0
        elif market_regime == 'range':
            # 震荡市场，增加适应性权重
            weights['profit'] = 1.5
            weights['adaptability'] = 2.5
            weights['robustness'] = 1.0
            
        # 归一化权重，使总和为5
        total = sum(weights.values())
        for key in weights:
            weights[key] = weights[key] * 5 / total
            
        return weights
        
    def set_market_regime(self, market_regime: str) -> None:
        """
        设置市场环境并调整权重
        
        参数:
            market_regime: 市场环境
        """
        self.market_regime = market_regime
        self.weights = self._adjust_weights_for_market(market_regime)
        logger.info(f"更新市场环境为: {market_regime}, 新权重: {self.weights}")
        
    def calculate_score(self, metrics: Dict) -> StrategyScore:
        """
        计算策略总分
        
        参数:
            metrics: 策略评估指标
            
        返回:
            策略评分结果
        """
        # 1. 盈利能力得分 (最高2分)
        profit_score = self._calculate_profit_score(
            metrics.get('total_return', 0),
            metrics.get('win_rate', 0),
            metrics.get('profit_factor', 0),
            metrics.get('sharpe_ratio', 0)
        )
        
        # 2. 适应性得分 (最高2分)
        adaptability_score = self._calculate_adaptability_score(
            metrics.get('monthly_returns', []),  # 每月收益率列表
            metrics.get('drawdown_periods', []), # 回撤期列表
            metrics.get('trade_intervals', []),  # 交易间隔列表
            metrics.get('recovery_speed', 0)     # 恢复速度
        )
        
        # 3. 稳健性得分 (最高1分)
        robustness_score = self._calculate_robustness_score(
            metrics.get('max_drawdown', 0),
            metrics.get('volatility', 0),
            metrics.get('avg_holding_days', 0),
            metrics.get('downside_deviation', 0)
        )
        
        # 应用权重
        weighted_profit = profit_score * (self.weights['profit'] / 2.0)
        weighted_adaptability = adaptability_score * (self.weights['adaptability'] / 2.0)
        weighted_robustness = robustness_score * (self.weights['robustness'] / 1.0)
        
        # 计算总分
        total_score = weighted_profit + weighted_adaptability + weighted_robustness
        
        return StrategyScore(
            strategy_name=metrics.get('strategy_name', 'Unknown'),
            symbol=metrics.get('symbol', 'Unknown'),
            total_score=total_score,
            profit_score=profit_score,
            adaptability_score=adaptability_score,
            robustness_score=robustness_score,
            details=metrics
        )
    
    def _calculate_profit_score(self, total_return: float, win_rate: float, 
                              profit_factor: float, sharpe_ratio: float) -> float:
        """
        计算盈利能力得分 (2分)
        
        参数:
            total_return: 总收益率
            win_rate: 胜率
            profit_factor: 盈亏比
            sharpe_ratio: 夏普比率
            
        返回:
            盈利能力得分 (0-2)
        """
        # 收益率得分 (0.8分)
        # 根据股票自身的波动特性来动态调整基准
        return_score = min(max(total_return / 20.0, 0), 0.8)
        
        # 胜率得分 (0.4分)
        win_rate_score = min(win_rate * 0.5, 0.4)
        
        # 盈亏比得分 (0.4分)
        pf_score = min(profit_factor / 5.0, 0.4)
        
        # 夏普比率得分 (0.4分)
        sharpe_score = min(max(sharpe_ratio / 3.0, 0), 0.4)
        
        return return_score + win_rate_score + pf_score + sharpe_score
    
    def _calculate_adaptability_score(self, monthly_returns: List[float], 
                                    drawdown_periods: List[Dict], 
                                    trade_intervals: List[int],
                                    recovery_speed: float) -> float:
        """
        计算策略适应性得分 (2分)
        
        参数:
            monthly_returns: 月度收益率列表
            drawdown_periods: 回撤期列表
            trade_intervals: 交易间隔列表
            recovery_speed: 恢复速度
            
        返回:
            适应性得分 (0-2)
        """
        # 市场适应性得分 (0.8分)
        # 检查策略在不同市场环境下的表现
        if not monthly_returns:
            market_score = 0
        else:
            # 计算月度收益的稳定性
            returns_std = np.std(monthly_returns) if len(monthly_returns) > 1 else 0
            market_score = min(0.8 / (1 + returns_std * 5), 0.8)
        
        # 恢复能力得分 (0.6分)
        # 评估策略从回撤中恢复的能力
        if not drawdown_periods:
            recovery_score = 0.3  # 如果没有回撤，给一个中等分数
        else:
            # 使用提供的恢复速度，或者计算平均恢复天数
            if recovery_speed > 0:
                recovery_score = min(0.6 * recovery_speed, 0.6)
            else:
                avg_recovery_days = np.mean([p.get('recovery_days', 60) for p in drawdown_periods])
                recovery_score = 0.6 * (1 - min(avg_recovery_days / 60, 1))
        
        # 交易时机把握得分 (0.6分)
        # 评估策略捕捉交易机会的能力
        if not trade_intervals:
            timing_score = 0.3  # 如果没有交易，给一个中等分数
        else:
            # 理想的交易间隔应该既不太频繁也不太稀疏
            avg_interval = np.mean(trade_intervals)
            timing_score = 0.6 * (1 - abs(avg_interval - 10) / 20)
            timing_score = max(0, min(timing_score, 0.6))
            
        return market_score + recovery_score + timing_score
    
    def _calculate_robustness_score(self, max_drawdown: float, volatility: float, 
                                  avg_holding_days: float, downside_deviation: float) -> float:
        """
        计算策略稳健性得分 (1分)
        
        参数:
            max_drawdown: 最大回撤
            volatility: 波动率
            avg_holding_days: 平均持仓天数
            downside_deviation: 下行波动率
            
        返回:
            稳健性得分 (0-1)
        """
        # 风险控制得分 (0.4分)
        # 最大回撤不超过15%为最优
        risk_score = 0.4 * max(0, 1 - abs(max_drawdown) / 15)
        
        # 波动控制得分 (0.3分)
        # 年化波动率不超过20%为最优
        vol_score = 0.3 * max(0, 1 - volatility / 0.2)
        
        # 持仓周期得分 (0.2分)
        # 平均持仓5-15天为最优
        holding_score = 0.2 * (1 - min(abs(avg_holding_days - 10) / 10, 1))
        
        # 下行风险得分 (0.1分)
        # 下行波动率越低越好
        downside_score = 0.1 * max(0, 1 - downside_deviation / 0.1)
        
        return risk_score + vol_score + holding_score + downside_score
    
    def rank_strategies(self, scores: List[StrategyScore], symbol: str = None) -> List[StrategyScore]:
        """
        对策略进行排名
        
        参数:
            scores: 策略评分列表
            symbol: 股票代码，如果提供则只对该股票的策略进行排名
            
        返回:
            排序后的策略评分列表
        """
        if symbol:
            # 只对特定股票的策略进行排名
            scores = [s for s in scores if s.symbol == symbol]
        return sorted(scores, key=lambda x: x.total_score, reverse=True)
    
    def generate_score_report(self, scores: List[StrategyScore], symbol: str = None) -> str:
        """
        生成评分报告
        
        参数:
            scores: 策略评分列表
            symbol: 股票代码，如果提供则只对该股票的策略进行排名
            
        返回:
            评分报告文本
        """
        report = "策略评分报告\n"
        report += "=" * 50 + "\n\n"
        
        if symbol:
            report += f"股票代码: {symbol}\n\n"
        
        # 对策略进行排名
        ranked_scores = self.rank_strategies(scores, symbol)
        
        for rank, score in enumerate(ranked_scores, 1):
            report += f"第{rank}名: {score.strategy_name}\n"
            report += f"总分: {score.total_score:.2f}/5.0\n"
            report += f"  - 盈利能力得分: {score.profit_score:.2f}/2.0 (权重: {self.weights['profit']:.1f})\n"
            report += f"  - 适应性得分: {score.adaptability_score:.2f}/2.0 (权重: {self.weights['adaptability']:.1f})\n"
            report += f"  - 稳健性得分: {score.robustness_score:.2f}/1.0 (权重: {self.weights['robustness']:.1f})\n"
            report += "\n详细指标:\n"
            report += f"  - 总收益率: {score.details.get('total_return', 0):.2f}%\n"
            report += f"  - 胜率: {score.details.get('win_rate', 0):.2%}\n"
            report += f"  - 盈亏比: {score.details.get('profit_factor', 0):.2f}\n"
            report += f"  - 最大回撤: {score.details.get('max_drawdown', 0):.2f}%\n"
            report += f"  - 夏普比率: {score.details.get('sharpe_ratio', 0):.2f}\n"
            report += f"  - 年化波动率: {score.details.get('volatility', 0):.2%}\n"
            report += f"  - 平均持仓天数: {score.details.get('avg_holding_days', 0):.1f}\n"
            report += "-" * 50 + "\n\n"
            
        return report
        
    def get_market_regime_from_data(self, market_data: pd.DataFrame) -> str:
        """
        从市场数据判断市场环境
        
        参数:
            market_data: 市场数据
            
        返回:
            市场环境: 'normal', 'volatile', 'trending', 'range'
        """
        try:
            if market_data is None or market_data.empty or len(market_data) < 20:
                return 'normal'
                
            # 计算波动率
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1]
                
                # 计算趋势强度
                price_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-20] - 1)
                price_range = (market_data['close'].iloc[-20:].max() - market_data['close'].iloc[-20:].min()) / market_data['close'].iloc[-20]
                trend_strength = abs(price_change) / price_range if price_range > 0 else 0
                
                # 判断市场环境
                if volatility > 0.02:  # 高波动
                    return 'volatile'
                elif trend_strength > 0.7:  # 强趋势
                    return 'trending'
                elif trend_strength < 0.3:  # 弱趋势
                    return 'range'
                    
            return 'normal'
            
        except Exception as e:
            logger.error(f"判断市场环境时出错: {e}")
            return 'normal' 