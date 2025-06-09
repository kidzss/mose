import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from enum import Enum
from datetime import datetime

from .market_environment_classifier import MarketEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalEvaluator")

class SignalStrength(Enum):
    """信号强度枚举类型"""
    VERY_STRONG = "非常强"
    STRONG = "强"
    MODERATE = "中等"
    WEAK = "弱"
    VERY_WEAK = "非常弱"
    INVALID = "无效"


class SignalQualityEvaluator:
    """
    信号质量评估器
    
    对生成的交易信号进行多维度评估，过滤低质量信号，提高决策质量。
    评估维度包括：
    1. 技术指标一致性
    2. 市场环境契合度
    3. 多时间框架确认
    4. 成交量支持度
    5. 风险回报比
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化信号质量评估器
        
        参数:
            config: 配置字典，可包含评估参数
        """
        logger.info("初始化信号质量评估器")
        
        # 默认配置
        self.default_config = {
            # 质量评分阈值，低于此值的信号将被过滤
            'quality_threshold': 0.6,  # 总体质量阈值，范围0-1
            
            # 各评分维度权重
            'weights': {
                'technical_consistency': 0.25,  # 技术指标一致性权重
                'environment_fit': 0.20,        # 市场环境契合度权重
                'timeframe_confirmation': 0.20, # 多时间框架确认权重
                'volume_support': 0.15,         # 成交量支持度权重
                'risk_reward': 0.20            # 风险回报比权重
            },
            
            # 技术指标一致性参数
            'min_consistent_indicators': 3,    # 最小一致指标数量
            'indicator_groups': {
                'trend': ['sma_crossover', 'macd', 'adx', 'supertrend'],
                'momentum': ['rsi', 'stochastic', 'cci', 'mfi'],
                'volatility': ['bollinger_bands', 'atr', 'keltner_channels'],
                'support_resistance': ['pivot_points', 'fibonacci', 'price_channels']
            },
            
            # 多时间框架确认参数
            'timeframes': ['daily', 'weekly', '4h', '1h'],
            'timeframe_weights': {'daily': 0.4, 'weekly': 0.3, '4h': 0.2, '1h': 0.1},
            
            # 风险回报比参数
            'min_reward_risk_ratio': 2.0,  # 最小回报风险比
            'optimal_reward_risk_ratio': 3.0,  # 理想回报风险比
            
            # 增强规则
            'enable_volume_filter': True,  # 启用成交量过滤
            'volume_threshold': 1.5,       # 成交量阈值（相对于N日均量）
            
            # 高级设置
            'adaptive_threshold': True,    # 根据市场环境自适应调整阈值
            'market_regime_thresholds': {
                MarketEnvironment.STRONG_UPTREND: 0.55,
                MarketEnvironment.WEAK_UPTREND: 0.60,
                MarketEnvironment.STRONG_DOWNTREND: 0.65,
                MarketEnvironment.WEAK_DOWNTREND: 0.65,
                MarketEnvironment.RANGE_BOUND: 0.70,
                MarketEnvironment.CHOPPY: 0.75,
                MarketEnvironment.UNKNOWN: 0.70
            }
        }
        
        # 应用用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        logger.info(f"配置参数: {self.config}")
    
    def evaluate_signal(
            self,
            signal_data: Dict[str, Any],
            market_data: pd.DataFrame,
            market_environment: MarketEnvironment,
            additional_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """
        评估交易信号质量
        
        参数:
            signal_data: 信号数据，包含方向、入场价、止损价、目标价等
            market_data: 市场数据，包含OHLCV和技术指标
            market_environment: 当前市场环境
            additional_data: 其他辅助数据
            
        返回:
            包含评估结果的字典:
            - quality_score: 总体质量分数 (0-1)
            - strength: 信号强度枚举
            - dimension_scores: 各维度评分
            - passed_threshold: 是否通过质量阈值
            - recommendations: 改进建议
        """
        logger.info(f"开始评估交易信号: 方向={signal_data['direction']}, 环境={market_environment.value}")
        logger.debug(f"信号数据: {signal_data}")
        
        try:
            dimension_scores = {}
            recommendations = []
            
            # 1. 评估技术指标一致性
            tech_score, tech_details = self._evaluate_technical_consistency(signal_data, market_data)
            dimension_scores['technical_consistency'] = tech_score
            
            if tech_score < 0.6:
                recommendations.append("技术指标一致性较低，建议检查冲突指标")
                
            # 2. 评估市场环境契合度
            env_score, env_details = self._evaluate_environment_fit(signal_data, market_environment)
            dimension_scores['environment_fit'] = env_score
            
            if env_score < 0.6:
                recommendations.append(f"信号与当前市场环境({market_environment.value})不够匹配")
                
            # 3. 评估多时间框架确认
            if additional_data and 'timeframe_data' in additional_data:
                tf_score, tf_details = self._evaluate_timeframe_confirmation(
                    signal_data, additional_data['timeframe_data']
                )
            else:
                tf_score, tf_details = 0.5, {"reason": "缺少多时间框架数据，使用中性评分"}
                recommendations.append("缺少多时间框架数据，无法进行全面评估")
                
            dimension_scores['timeframe_confirmation'] = tf_score
            
            if tf_score < 0.6 and 'timeframe_data' in (additional_data or {}):
                recommendations.append("多时间框架确认度不足，存在反向信号")
                
            # 4. 评估成交量支持度
            vol_score, vol_details = self._evaluate_volume_support(signal_data, market_data)
            dimension_scores['volume_support'] = vol_score
            
            if vol_score < 0.5:
                recommendations.append("成交量支持不足，可能影响信号可靠性")
                
            # 5. 评估风险回报比
            if 'stop_loss' in signal_data and 'target_price' in signal_data and 'entry_price' in signal_data:
                rr_score, rr_details = self._evaluate_risk_reward(signal_data)
            else:
                rr_score, rr_details = 0.5, {"reason": "缺少止损/目标价数据，使用中性评分"}
                recommendations.append("未设置止损或目标价，无法评估风险回报比")
                
            dimension_scores['risk_reward'] = rr_score
            
            # 计算总体质量分数
            quality_score = self._calculate_overall_score(dimension_scores)
            
            # 确定信号强度
            strength = self._determine_signal_strength(quality_score)
            
            # 判断是否通过质量阈值
            threshold = self._get_adaptive_threshold(market_environment)
            passed_threshold = quality_score >= threshold
            
            if not passed_threshold:
                recommendations.append(f"信号质量({quality_score:.2f})低于阈值({threshold:.2f})，建议不执行或降低仓位")
            
            result = {
                'quality_score': quality_score,
                'strength': strength,
                'dimension_scores': dimension_scores,
                'passed_threshold': passed_threshold,
                'recommendations': recommendations,
                'details': {
                    'technical_consistency': tech_details,
                    'environment_fit': env_details,
                    'timeframe_confirmation': tf_details,
                    'volume_support': vol_details,
                    'risk_reward': rr_details
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"信号评估过程中发生错误: {str(e)}", exc_info=True)
            return {
                'quality_score': 0.0,
                'strength': SignalStrength.INVALID,
                'dimension_scores': {},
                'passed_threshold': False,
                'recommendations': [f"信号评估过程中出错: {str(e)}"],
                'details': {'error': str(e)}
            }
            
    def _evaluate_technical_consistency(
            self,
            signal_data: Dict[str, Any],
            market_data: pd.DataFrame
        ) -> Tuple[float, Dict[str, Any]]:
        """评估技术指标一致性"""
        # 获取信号方向，1表示买入，-1表示卖出
        signal_direction = signal_data.get('direction', 0)
        if signal_direction == 0:
            return 0.0, {"reason": "信号方向不明确"}
            
        # 从信号数据中提取技术指标信号
        indicator_signals = signal_data.get('indicator_signals', {})
        if not indicator_signals:
            # 尝试从市场数据中提取常见指标
            indicator_signals = self._extract_indicator_signals_from_market_data(market_data, signal_direction)
            
        # 按组统计支持信号方向的指标数量
        group_consistency = {}
        total_indicators = 0
        consistent_indicators = 0
        
        for group, indicators in self.config['indicator_groups'].items():
            group_total = 0
            group_consistent = 0
            
            for indicator in indicators:
                if indicator in indicator_signals:
                    group_total += 1
                    total_indicators += 1
                    
                    # 检查指标信号是否与整体方向一致
                    if (indicator_signals[indicator] > 0 and signal_direction > 0) or \
                       (indicator_signals[indicator] < 0 and signal_direction < 0):
                        group_consistent += 1
                        consistent_indicators += 1
                        
            if group_total > 0:
                group_consistency[group] = group_consistent / group_total
                
        # 如果没有指标信号，给一个中性分数
        if total_indicators == 0:
            return 0.5, {"reason": "没有可用的技术指标数据"}
            
        # 计算整体一致性分数
        overall_consistency = consistent_indicators / total_indicators
        
        # 计算加权分数，权重更高的组对整体分数影响更大
        weighted_score = overall_consistency
        
        # 收集详细信息
        details = {
            'total_indicators': total_indicators,
            'consistent_indicators': consistent_indicators,
            'overall_consistency': overall_consistency,
            'group_consistency': group_consistency,
            'has_minimum_indicators': consistent_indicators >= self.config['min_consistent_indicators']
        }
        
        # 如果一致指标数量少于最小要求，降低分数
        if consistent_indicators < self.config['min_consistent_indicators']:
            weighted_score *= (consistent_indicators / self.config['min_consistent_indicators'])
        
        return weighted_score, details
    
    def _extract_indicator_signals_from_market_data(
            self,
            market_data: pd.DataFrame,
            signal_direction: int
        ) -> Dict[str, float]:
        """从市场数据中提取常见技术指标信号"""
        signals = {}
        
        # 检查市场数据中的常见指标
        if 'rsi' in market_data.columns:
            rsi = market_data['rsi'].iloc[-1]
            if rsi > 70:
                signals['rsi'] = -1  # 超买，看跌信号
            elif rsi < 30:
                signals['rsi'] = 1   # 超卖，看涨信号
            else:
                signals['rsi'] = 0
                
        if 'macd' in market_data.columns and 'macd_signal' in market_data.columns:
            macd = market_data['macd'].iloc[-1]
            signal_line = market_data['macd_signal'].iloc[-1]
            if macd > signal_line:
                signals['macd'] = 1  # MACD在信号线上方，看涨
            else:
                signals['macd'] = -1  # MACD在信号线下方，看跌
                
        if 'sma_20' in market_data.columns and 'sma_50' in market_data.columns:
            sma_20 = market_data['sma_20'].iloc[-1]
            sma_50 = market_data['sma_50'].iloc[-1]
            close = market_data['close'].iloc[-1]
            
            if close > sma_20 and sma_20 > sma_50:
                signals['sma_crossover'] = 1  # 价格在短期均线上方，短期均线在长期均线上方，看涨
            elif close < sma_20 and sma_20 < sma_50:
                signals['sma_crossover'] = -1  # 价格在短期均线下方，短期均线在长期均线下方，看跌
            else:
                signals['sma_crossover'] = 0
                
        if 'adx' in market_data.columns and 'plus_di' in market_data.columns and 'minus_di' in market_data.columns:
            adx = market_data['adx'].iloc[-1]
            plus_di = market_data['plus_di'].iloc[-1]
            minus_di = market_data['minus_di'].iloc[-1]
            
            if adx > 25:
                if plus_di > minus_di:
                    signals['adx'] = 1  # ADX值大且+DI>-DI，看涨
                else:
                    signals['adx'] = -1  # ADX值大且-DI>+DI，看跌
            else:
                signals['adx'] = 0  # ADX值小，无明确趋势
                
        # 其他可能的指标...
        
        return signals
    
    def _evaluate_environment_fit(
            self,
            signal_data: Dict[str, Any],
            market_environment: MarketEnvironment
        ) -> Tuple[float, Dict[str, Any]]:
        """评估信号与市场环境的契合度"""
        # 获取信号方向，1表示买入，-1表示卖出
        signal_direction = signal_data.get('direction', 0)
        
        # 根据不同市场环境与信号方向的契合度评分
        environment_fit_scores = {
            # 上升趋势环境
            MarketEnvironment.STRONG_UPTREND: {
                1: 0.9,   # 买入信号在强上升趋势中评分高
                -1: 0.2   # 卖出信号在强上升趋势中评分低
            },
            MarketEnvironment.WEAK_UPTREND: {
                1: 0.7,   # 买入信号在弱上升趋势中评分较高
                -1: 0.3   # 卖出信号在弱上升趋势中评分较低
            },
            # 下降趋势环境
            MarketEnvironment.STRONG_DOWNTREND: {
                1: 0.2,   # 买入信号在强下降趋势中评分低
                -1: 0.9   # 卖出信号在强下降趋势中评分高
            },
            MarketEnvironment.WEAK_DOWNTREND: {
                1: 0.3,   # 买入信号在弱下降趋势中评分较低
                -1: 0.7   # 卖出信号在弱下降趋势中评分较高
            },
            # 区间震荡环境
            MarketEnvironment.RANGE_BOUND: {
                1: 0.6,   # 买入信号在区间下方较好
                -1: 0.6   # 卖出信号在区间上方较好
            },
            # 混沌无序环境
            MarketEnvironment.CHOPPY: {
                1: 0.4,   # 混沌环境中信号可靠性下降
                -1: 0.4
            },
            MarketEnvironment.UNKNOWN: {
                1: 0.5,   # 未知环境中给予中性评分
                -1: 0.5
            }
        }
        
        # 获取基本契合度分数
        base_score = environment_fit_scores.get(
            market_environment, environment_fit_scores[MarketEnvironment.UNKNOWN]
        ).get(signal_direction, 0.5)
        
        # 进一步根据信号特性和环境进行调整
        adjusted_score = base_score
        
        # 例如，在区间震荡环境中，如果是在区间下方买入或区间上方卖出，调高分数
        if market_environment == MarketEnvironment.RANGE_BOUND:
            price_location = signal_data.get('price_location', None)
            if price_location == 'lower_bound' and signal_direction == 1:
                adjusted_score += 0.2
            elif price_location == 'upper_bound' and signal_direction == -1:
                adjusted_score += 0.2
                
        # 收集详细信息
        details = {
            'base_score': base_score,
            'adjusted_score': adjusted_score,
            'environment': market_environment.value,
            'signal_direction': '买入' if signal_direction > 0 else '卖出',
        }
        
        return min(adjusted_score, 1.0), details
    
    def _evaluate_timeframe_confirmation(
            self,
            signal_data: Dict[str, Any],
            timeframe_data: Dict[str, pd.DataFrame]
        ) -> Tuple[float, Dict[str, Any]]:
        """评估多时间框架确认"""
        # 获取信号方向
        signal_direction = signal_data.get('direction', 0)
        if signal_direction == 0:
            return 0.5, {"reason": "信号方向不明确"}
            
        # 统计各时间框架下的趋势方向
        timeframe_trends = {}
        total_weight = 0
        weighted_agreement = 0
        
        for timeframe, weight in self.config['timeframe_weights'].items():
            if timeframe in timeframe_data:
                df = timeframe_data[timeframe]
                
                # 使用简单的方向判断逻辑
                short_ma = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['close'].mean()
                long_ma = df['close'].rolling(30).mean().iloc[-1] if len(df) >= 30 else df['close'].mean()
                
                if short_ma > long_ma:
                    trend_direction = 1  # 上升趋势
                else:
                    trend_direction = -1  # 下降趋势
                
                timeframe_trends[timeframe] = trend_direction
                
                # 如果方向一致，增加确认分数
                if trend_direction == signal_direction:
                    weighted_agreement += weight
                
                total_weight += weight
        
        # 如果没有足够的时间框架数据，给一个中性分数
        if total_weight == 0:
            return 0.5, {"reason": "无可用的多时间框架数据"}
        
        # 计算最终确认分数
        confirmation_score = weighted_agreement / total_weight
        
        # 收集详细信息
        details = {
            'timeframe_trends': timeframe_trends,
            'weighted_agreement': weighted_agreement,
            'total_weight': total_weight,
            'used_timeframes': list(timeframe_trends.keys())
        }
        
        return confirmation_score, details
    
    def _evaluate_volume_support(
            self,
            signal_data: Dict[str, Any],
            market_data: pd.DataFrame
        ) -> Tuple[float, Dict[str, Any]]:
        """评估成交量支持度"""
        if 'volume' not in market_data.columns:
            return 0.5, {"reason": "缺少成交量数据"}
            
        # 获取当前成交量和最近的均值
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 获取信号方向
        signal_direction = signal_data.get('direction', 0)
        
        # 基本成交量评分
        if volume_ratio >= self.config['volume_threshold']:
            volume_score = min(volume_ratio / (self.config['volume_threshold'] * 2), 1.0)
        else:
            volume_score = volume_ratio / self.config['volume_threshold']
            
        # 考虑成交量与价格变化的一致性
        price_change = market_data['close'].pct_change().iloc[-1]
        
        # 成交量与价格变化方向一致时，是好的信号
        # - 上涨时成交量增加：强烈看涨
        # - 下跌时成交量增加：强烈看跌
        volume_price_agreement = False
        if (price_change > 0 and signal_direction > 0 and volume_ratio > 1) or \
           (price_change < 0 and signal_direction < 0 and volume_ratio > 1):
            volume_price_agreement = True
            volume_score = min(volume_score * 1.2, 1.0)
            
        # 收集详细信息
        details = {
            'volume_ratio': volume_ratio,
            'volume_threshold': self.config['volume_threshold'],
            'volume_price_agreement': volume_price_agreement,
            'price_change': price_change
        }
        
        return volume_score, details
    
    def _evaluate_risk_reward(
            self,
            signal_data: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Any]]:
        """评估风险回报比"""
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        target_price = signal_data['target_price']
        
        # 如果目标价与当前价格相同，或止损与当前价格相同，表示无法计算
        if entry_price == target_price or entry_price == stop_loss:
            return 0.5, {"reason": "入场价与目标价或止损价相同，无法计算风险回报比"}
            
        # 计算风险
        risk = abs(entry_price - stop_loss)
        
        # 计算潜在回报
        reward = abs(target_price - entry_price)
        
        # 计算回报风险比
        reward_risk_ratio = reward / risk if risk > 0 else 0
        
        # 基于回报风险比评分
        min_ratio = self.config['min_reward_risk_ratio']
        optimal_ratio = self.config['optimal_reward_risk_ratio']
        
        if reward_risk_ratio >= optimal_ratio:
            rr_score = 1.0
        elif reward_risk_ratio >= min_ratio:
            rr_score = 0.5 + (reward_risk_ratio - min_ratio) / (optimal_ratio - min_ratio) * 0.5
        else:
            rr_score = 0.5 * (reward_risk_ratio / min_ratio)
        
        # 收集详细信息
        details = {
            'reward_risk_ratio': reward_risk_ratio,
            'risk': risk,
            'reward': reward,
            'min_ratio': min_ratio,
            'optimal_ratio': optimal_ratio
        }
        
        return rr_score, details
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """计算总体质量分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.config['weights'].get(dimension, 0.0)
            total_score += score * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        return total_score / total_weight
    
    def _determine_signal_strength(self, quality_score: float) -> SignalStrength:
        """根据质量分数确定信号强度"""
        if quality_score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif quality_score >= 0.75:
            return SignalStrength.STRONG
        elif quality_score >= 0.6:
            return SignalStrength.MODERATE
        elif quality_score >= 0.4:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
            
    def _get_adaptive_threshold(self, market_environment: MarketEnvironment) -> float:
        """获取根据市场环境自适应的阈值"""
        if self.config['adaptive_threshold']:
            return self.config['market_regime_thresholds'].get(
                market_environment, self.config['quality_threshold']
            )
        else:
            return self.config['quality_threshold']

    def _evaluate_trend_alignment(self, signal_direction, market_environment, market_data):
        """评估信号方向与市场趋势的一致性"""
        try:
            # 简化版趋势一致性评估
            sma_20 = market_data['sma_20'].iloc[-1] if 'sma_20' in market_data.columns else None
            sma_50 = market_data['sma_50'].iloc[-1] if 'sma_50' in market_data.columns else None
            
            if sma_20 is None or sma_50 is None:
                logger.warning("缺少趋势评估所需的移动平均线数据")
                return 0.5
            
            # 判断整体趋势
            market_trend = 0  # 0=无趋势, 1=上升趋势, -1=下降趋势
            
            if sma_20 > sma_50:
                market_trend = 1
                logger.debug("当前市场趋势: 上升 (SMA20 > SMA50)")
            elif sma_20 < sma_50:
                market_trend = -1
                logger.debug("当前市场趋势: 下降 (SMA20 < SMA50)")
            else:
                logger.debug("当前市场趋势: 横盘 (SMA20 ≈ SMA50)")
                
            # 考虑市场环境
            if market_environment in [MarketEnvironment.STRONG_UPTREND, MarketEnvironment.WEAK_UPTREND]:
                market_trend = max(market_trend, 0.5)  # 至少倾向于上升
                logger.debug("市场环境为上升趋势")
            elif market_environment in [MarketEnvironment.STRONG_DOWNTREND, MarketEnvironment.WEAK_DOWNTREND]:
                market_trend = min(market_trend, -0.5)  # 至少倾向于下降
                logger.debug("市场环境为下降趋势")
            elif market_environment == MarketEnvironment.RANGE_BOUND:
                market_trend *= 0.5  # 弱化任何趋势判断
                logger.debug("市场环境为区间震荡")
                
            # 计算一致性分数
            if signal_direction == 0:  # 中性信号
                score = 0.5
            elif (signal_direction > 0 and market_trend > 0) or (signal_direction < 0 and market_trend < 0):
                # 方向一致
                score = 0.5 + 0.5 * abs(market_trend)
                logger.debug(f"信号方向与市场趋势一致，得分: {score:.2f}")
            else:
                # 方向相反
                score = 0.5 - 0.5 * abs(market_trend)
                logger.debug(f"信号方向与市场趋势相反，得分: {score:.2f}")
                
            return score
            
        except Exception as e:
            logger.error(f"评估趋势一致性时出错: {str(e)}", exc_info=True)
            return 0.5