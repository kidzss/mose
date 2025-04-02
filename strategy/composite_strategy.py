import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Type

from .strategy_base import Strategy, MarketRegime
from .signal_interface import (
    SignalType, SignalTimeframe, SignalStrength, 
    SignalMetadata, SignalComponent, SignalCombiner
)


class CompositeStrategy(Strategy):
    """
    组合策略 (优化版)
    
    结合多个独立策略的信号，使用权重和过滤规则生成最终交易信号。
    特点:
    1. 可以同时运行多个不同类型的策略
    2. 使用加权投票或其他方法合并信号
    3. 自适应调整各策略的权重
    4. 内置冲突解决机制
    5. 提供综合风险管理
    6. 根据市场环境动态调整各策略权重
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None, strategies: Optional[List[Strategy]] = None):
        """
        初始化组合策略
        
        参数:
            parameters: 策略参数字典
            strategies: 策略实例列表
        """
        # 设置默认参数
        default_params = {
            # 组合方法
            'combination_method': 'weighted_average',  # 'weighted_average', 'majority_vote', 'min', 'max'
            'adaptive_weights': True,                  # 是否自适应调整权重
            'minimum_consensus': 0.5,                  # 最小共识比例(0-1)
            
            # 信号确认
            'confirmation_threshold': 0.6,             # 信号确认阈值
            'minimum_strategies': 2,                   # 最小确认策略数
            
            # 风险管理
            'overall_risk_limit': 0.8,                 # 整体风险限制(0-1)
            'max_correlated_exposure': 0.6,            # 最大相关曝险
            
            # 策略权重调整参数
            'performance_lookback': 60,                # 性能回顾期(天)
            'weight_update_frequency': 30,             # 权重更新频率(天)
            
            # 其他参数
            'use_market_regime': True,                 # 是否使用市场环境
            
            # 市场环境权重调整
            'market_regime_weight_adjust': True,       # 是否根据市场环境调整权重
            'bullish_weight_boost': 1.5,               # 牛市权重提升因子
            'bearish_weight_boost': 1.5,               # 熊市权重提升因子
            'ranging_weight_boost': 1.5,               # 震荡市场权重提升因子
            
            # 策略权重动态调整幅度
            'weight_adjustment_rate': 0.8,             # 权重调整速率(0-1)
            
            # 信号一致性奖励
            'consensus_reward': 0.3,                   # 信号一致时的权重奖励
            
            # 策略评估
            'strategy_metric_weights': {               # 评估指标的权重
                'win_rate': 0.3,                       # 胜率的权重
                'avg_profit': 0.3,                     # 平均收益的权重
                'sharpe_ratio': 0.2,                   # 夏普比率的权重
                'consistency': 0.2                     # 一致性的权重
            }
        }
        
        # 更新默认参数
        if parameters:
            default_params.update(parameters)
            
        # 初始化基类
        super().__init__('CompositeStrategy', default_params)
        
        # 初始化成员策略列表和权重
        self.strategies = strategies or []
        
        # 初始化策略权重为均等权重
        self._initialize_weights()
        
        # 记录策略性能
        self.strategy_performance = {strategy.name: {'wins': 0, 'losses': 0, 'total': 0, 
                                                     'bullish_score': 0.5, 'bearish_score': 0.5, 
                                                     'ranging_score': 0.5, 'volatile_score': 0.5} 
                                   for strategy in self.strategies}
        
        # 上次更新权重的日期
        self.last_weight_update = None
        
        # 保存原始权重以便市场环境调整时参考
        self.base_weights = {}
        
        # 记录不同市场环境下的性能
        self.market_regime_performance = {
            MarketRegime.BULLISH: {},
            MarketRegime.BEARISH: {},
            MarketRegime.RANGING: {},
            MarketRegime.VOLATILE: {}
        }
    
    def _initialize_weights(self):
        """初始化策略权重"""
        self.strategy_weights = {}
        if self.strategies:
            # 初始时设置均等权重
            equal_weight = 1.0 / len(self.strategies)
            for strategy in self.strategies:
                self.strategy_weights[strategy.name] = equal_weight
                # 同时保存基础权重
                self.base_weights[strategy.name] = equal_weight
        
    def add_strategy(self, strategy: Strategy, weight: Optional[float] = None):
        """
        添加策略到组合
        
        参数:
            strategy: 策略实例
            weight: 策略权重，如果为None则自动平衡权重
        """
        # 添加策略
        self.strategies.append(strategy)
        
        # 更新权重
        if weight is not None:
            # 直接设置权重
            self.strategy_weights[strategy.name] = weight
            self.base_weights[strategy.name] = weight
            
            # 缩放其他策略的权重，使总和为1
            weight_sum = sum(self.strategy_weights.values())
            if weight_sum > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= weight_sum
                    self.base_weights[name] /= weight_sum
        else:
            # 重新平衡所有权重
            self._initialize_weights()
            
        # 初始化性能记录
        self.strategy_performance[strategy.name] = {
            'wins': 0, 'losses': 0, 'total': 0,
            'bullish_score': 0.5, 'bearish_score': 0.5, 
            'ranging_score': 0.5, 'volatile_score': 0.5
        }
        
        # 初始化不同市场环境下的性能记录
        for regime in self.market_regime_performance:
            self.market_regime_performance[regime][strategy.name] = 0.5
        
        self.logger.info(f"添加策略 {strategy.name}，权重: {self.strategy_weights[strategy.name]:.4f}")
    
    def remove_strategy(self, strategy_name: str):
        """
        从组合中移除策略
        
        参数:
            strategy_name: 策略名称
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                self.strategies.pop(i)
                
                # 从权重字典中移除
                if strategy_name in self.strategy_weights:
                    del self.strategy_weights[strategy_name]
                    
                if strategy_name in self.base_weights:
                    del self.base_weights[strategy_name]
                
                # 从性能记录中移除
                if strategy_name in self.strategy_performance:
                    del self.strategy_performance[strategy_name]
                
                # 从市场环境性能记录中移除
                for regime in self.market_regime_performance:
                    if strategy_name in self.market_regime_performance[regime]:
                        del self.market_regime_performance[regime][strategy_name]
                
                # 重新平衡权重
                self._initialize_weights()
                
                self.logger.info(f"移除策略 {strategy_name}")
                return
                
        self.logger.warning(f"未找到策略 {strategy_name}")
    
    def update_strategy_weights(self, data: pd.DataFrame):
        """
        根据策略性能更新权重
        
        参数:
            data: 历史价格数据
        """
        if not self.parameters['adaptive_weights']:
            return  # 如果不启用自适应权重，直接返回
            
        # 检查是否需要更新权重
        current_date = data.index[-1]
        if (self.last_weight_update is not None and 
            (current_date - self.last_weight_update).days < self.parameters['weight_update_frequency']):
            return  # 未到更新时间
            
        # 检查历史数据是否足够
        lookback = self.parameters['performance_lookback']
        if len(data) < lookback:
            return  # 数据不足
            
        # 记录当前日期
        self.last_weight_update = current_date
        
        # 评估每个策略的性能
        strategy_performance = {}
        
        # 确定当前市场环境
        current_regime = self.get_market_regime(data)
        self.logger.info(f"当前市场环境: {current_regime}")
        
        for strategy in self.strategies:
            # 获取策略信号
            try:
                signals = strategy.generate_signals(data.iloc[-lookback:])
                
                # 计算信号的盈亏情况
                if 'signal' in signals.columns:
                    # 简单的盈亏计算（基于下一天的价格变化）
                    signals['next_return'] = signals['close'].pct_change().shift(-1)
                    signals['profit'] = signals['signal'] * signals['next_return']
                    
                    # 计算胜率和平均收益
                    win_rate = (signals['profit'] > 0).mean()
                    avg_profit = signals['profit'].mean()
                    
                    # 计算夏普比率
                    if signals['profit'].std() > 0:
                        sharpe_ratio = signals['profit'].mean() / signals['profit'].std() * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                        
                    # 计算信号一致性 (相邻信号变化频率的倒数)
                    signal_changes = (signals['signal'].diff() != 0).sum()
                    consistency = 1.0 - min(0.9, signal_changes / len(signals))
                    
                    # 使用加权方式计算最终评分
                    metric_weights = self.parameters['strategy_metric_weights']
                    
                    # 组合绩效分数
                    if np.isnan(win_rate) or np.isnan(avg_profit):
                        performance_score = 0.5  # 默认中性评分
                    else:
                        performance_score = (
                            win_rate * metric_weights['win_rate'] +
                            (avg_profit * 100) * metric_weights['avg_profit'] +
                            (sharpe_ratio / 3.0) * metric_weights['sharpe_ratio'] +
                            consistency * metric_weights['consistency']
                        )
                        performance_score = max(0.1, min(1.0, performance_score))  # 限制在0.1-1.0之间
                    
                    strategy_performance[strategy.name] = performance_score
                    
                    # 更新不同市场环境下的性能评分
                    # 根据近期数据确定市场环境进行分类评估
                    regimes = [self.get_market_regime(data.iloc[i-20:i+1]) for i in range(20, len(data), 20)]
                    
                    # 对每种市场环境分别计算性能
                    for regime in set(regimes):
                        if regime in self.market_regime_performance:
                            # 找出该环境下的时间段
                            indices = [i for i, r in enumerate(regimes) if r == regime]
                            if indices:
                                # 计算这些时间段的性能
                                regime_signals = []
                                for idx in indices:
                                    start_idx = max(0, idx * 20)
                                    end_idx = min(len(signals), (idx + 1) * 20)
                                    regime_signals.extend(signals.iloc[start_idx:end_idx].index.tolist())
                                
                                if regime_signals:
                                    regime_data = signals.loc[regime_signals]
                                    if len(regime_data) > 0:
                                        regime_win_rate = (regime_data['profit'] > 0).mean()
                                        regime_avg_profit = regime_data['profit'].mean()
                                        
                                        # 计算该环境下的性能评分
                                        if np.isnan(regime_win_rate) or np.isnan(regime_avg_profit):
                                            regime_score = 0.5
                                        else:
                                            regime_score = regime_win_rate * 0.6 + (regime_avg_profit * 100) * 0.4
                                            regime_score = max(0.1, min(1.0, regime_score))
                                        
                                        # 更新环境下的评分，使用指数平滑
                                        old_score = self.market_regime_performance[regime].get(strategy.name, 0.5)
                                        new_score = old_score * 0.7 + regime_score * 0.3
                                        self.market_regime_performance[regime][strategy.name] = new_score
                    
                    # 更新策略性能记录中的环境评分
                    self.strategy_performance[strategy.name].update({
                        'bullish_score': self.market_regime_performance[MarketRegime.BULLISH].get(strategy.name, 0.5),
                        'bearish_score': self.market_regime_performance[MarketRegime.BEARISH].get(strategy.name, 0.5),
                        'ranging_score': self.market_regime_performance[MarketRegime.RANGING].get(strategy.name, 0.5),
                        'volatile_score': self.market_regime_performance[MarketRegime.VOLATILE].get(strategy.name, 0.5)
                    })
                    
                    self.logger.debug(f"策略 {strategy.name} 性能评分: {performance_score:.4f}, 当前环境评分: {self.market_regime_performance[current_regime].get(strategy.name, 0.5):.4f}")
                    
            except Exception as e:
                self.logger.error(f"评估策略 {strategy.name} 性能时出错: {e}")
                strategy_performance[strategy.name] = 0.5  # 出错时给予中性权重
                
        # 如果成功评估了所有策略，更新权重
        if strategy_performance:
            # 获取总权重
            total_score = sum(strategy_performance.values())
            
            if total_score > 0:
                # 基于性能更新权重
                for name, score in strategy_performance.items():
                    if name in self.strategy_weights:
                        # 使用调整率平滑权重变化
                        adjustment_rate = self.parameters['weight_adjustment_rate']
                        new_weight = score / total_score
                        self.base_weights[name] = self.base_weights[name] * (1 - adjustment_rate) + new_weight * adjustment_rate
            
            # 确保基础权重总和为1
            weight_sum = sum(self.base_weights.values())
            if weight_sum > 0:
                for name in self.base_weights:
                    self.base_weights[name] /= weight_sum
            
            # 如果启用市场环境权重调整，应用市场环境权重
            if self.parameters['market_regime_weight_adjust']:
                self._adjust_weights_for_market_regime(current_regime)
            else:
                # 否则直接使用基础权重
                self.strategy_weights = self.base_weights.copy()
            
            self.logger.info(f"更新策略权重: {self.strategy_weights}")
            
    def _adjust_weights_for_market_regime(self, regime: MarketRegime):
        """
        根据当前市场环境调整策略权重
        
        参数:
            regime: 当前市场环境
        """
        # 复制基础权重
        adjusted_weights = self.base_weights.copy()
        
        # 根据市场环境和每个策略在该环境下的表现调整权重
        regime_boost = 1.0
        regime_score_key = 'ranging_score'  # 默认使用震荡市场评分
        
        if regime == MarketRegime.BULLISH:
            regime_boost = self.parameters['bullish_weight_boost']
            regime_score_key = 'bullish_score'
            self.logger.info("牛市环境，调整策略权重")
        elif regime == MarketRegime.BEARISH:
            regime_boost = self.parameters['bearish_weight_boost']
            regime_score_key = 'bearish_score'
            self.logger.info("熊市环境，调整策略权重")
        elif regime == MarketRegime.RANGING:
            regime_boost = self.parameters['ranging_weight_boost']
            regime_score_key = 'ranging_score'
            self.logger.info("震荡环境，调整策略权重")
        
        # 基于当前环境下的性能调整权重
        for strategy_name in adjusted_weights:
            if strategy_name in self.strategy_performance:
                # 获取策略在当前环境下的评分
                regime_score = self.strategy_performance[strategy_name].get(regime_score_key, 0.5)
                
                # 根据环境评分调整权重
                boost_factor = 1 + (regime_score - 0.5) * (regime_boost - 1)
                adjusted_weights[strategy_name] *= boost_factor
        
        # 确保权重总和为1
        weight_sum = sum(adjusted_weights.values())
        if weight_sum > 0:
            for name in adjusted_weights:
                adjusted_weights[name] /= weight_sum
        
        # 更新策略权重
        self.strategy_weights = adjusted_weights
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成组合策略信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法生成信号")
            return pd.DataFrame()
            
        try:
            # 确保有足够的数据来评估市场环境
            if len(data) >= 50:  # 至少需要50条数据来评估市场环境
                # 更新策略权重
                self.update_strategy_weights(data)
                
            # 生成每个策略的信号
            all_signals = {}
            
            # 如果已经定义了策略则使用，否则返回空结果
            if not self.strategies:
                self.logger.warning("没有定义任何策略")
                return data.copy()
                
            # 生成每个策略的信号
            for strategy in self.strategies:
                try:
                    signal_df = strategy.generate_signals(data)
                    if 'signal' in signal_df.columns:
                        all_signals[strategy.name] = signal_df
                    else:
                        self.logger.warning(f"策略 {strategy.name} 没有生成signal列")
                except Exception as e:
                    self.logger.error(f"策略 {strategy.name} 生成信号时出错: {e}")
                    
            # 如果没有获取到任何信号，返回输入数据
            if not all_signals:
                self.logger.warning("没有策略生成有效信号")
                return data.copy()
                
            # 创建结果DataFrame
            result = data.copy()
            result['signal'] = 0
            
            # 确定组合方法
            method = self.parameters['combination_method']
            
            # 组合信号
            if method == 'weighted_average':
                self._combine_signals_weighted_average(result, all_signals)
            elif method == 'majority_vote':
                self._combine_signals_majority_vote(result, all_signals)
            elif method == 'min':
                self._combine_signals_min(result, all_signals)
            elif method == 'max':
                self._combine_signals_max(result, all_signals)
            else:
                self.logger.warning(f"未知的组合方法: {method}，使用加权平均")
                self._combine_signals_weighted_average(result, all_signals)
            
            # 应用信号确认
            min_threshold = self.parameters['confirmation_threshold']
            scaled_signal = result['raw_signal'] if 'raw_signal' in result else result['signal']
            
            # 将信号标准化到-1至1的范围内
            if abs(scaled_signal).max() > 1.0:
                scaled_signal = scaled_signal / abs(scaled_signal).max()
                
            # 弱信号过滤
            for i in range(len(result)):
                signal = scaled_signal.iloc[i]
                
                # 小于阈值的信号被归零
                if abs(signal) < min_threshold:
                    result.loc[result.index[i], 'signal'] = 0
                    
            # 检查是否需要应用市场环境调整
            if self.parameters['use_market_regime']:
                result = self.adjust_for_market_regime(data, result)
                
            # 处理最终信号值
            # 将连续信号转换为离散信号(-1, 0, 1)
            if 'signal' in result.columns:
                # 使用三元信号体系
                for i in range(len(result)):
                    signal = result['signal'].iloc[i]
                    if signal > 0.01:  # 大于0.01视为买入
                        result.loc[result.index[i], 'signal'] = 1
                    elif signal < -0.01:  # 小于-0.01视为卖出
                        result.loc[result.index[i], 'signal'] = -1
                    else:
                        result.loc[result.index[i], 'signal'] = 0
                
            return result
            
        except Exception as e:
            self.logger.error(f"组合策略生成信号时出错: {e}")
            return data.copy()
            
    def _combine_signals_weighted_average(self, result: pd.DataFrame, all_signals: Dict[str, pd.DataFrame]):
        """
        使用加权平均法组合信号
        
        参数:
            result: 结果DataFrame
            all_signals: 所有策略的信号
        """
        # 创建新列存储原始信号值
        result['raw_signal'] = 0.0
        
        # 对每个时间点计算加权信号
        for date in result.index:
            # 获取日期对应的每个策略信号
            weighted_sum = 0.0
            weight_sum = 0.0
            
            signals_this_date = {}
            
            # 收集当前日期所有策略的信号
            for strategy_name, signals in all_signals.items():
                if date in signals.index:
                    signal_value = signals.loc[date, 'signal']
                    if not np.isnan(signal_value):
                        signals_this_date[strategy_name] = signal_value
            
            # 检查信号一致性
            if len(signals_this_date) >= 2:  # 至少有两个策略
                # 检查所有有效信号是否一致(全为买入或全为卖出)
                is_consistent = all(s > 0 for s in signals_this_date.values()) or all(s < 0 for s in signals_this_date.values())
                
                # 如果信号一致，则增加权重
                consensus_boost = self.parameters['consensus_reward'] if is_consistent else 0.0
            else:
                consensus_boost = 0.0
            
            # 计算加权和
            for strategy_name, signal_value in signals_this_date.items():
                if strategy_name in self.strategy_weights:
                    weight = self.strategy_weights[strategy_name] * (1 + consensus_boost)
                    weighted_sum += signal_value * weight
                    weight_sum += weight
            
            # 应用加权平均
            if weight_sum > 0:
                result.loc[date, 'raw_signal'] = weighted_sum / weight_sum
                
        # 将原始信号应用到信号列
        result['signal'] = result['raw_signal']
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 在组合策略中，我们可能需要一些额外的指标来评估市场状态
        # 例如，计算波动率、趋势强度等
        
        # 波动率指标
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 趋势强度指标
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['trend_strength'] = (df['ma50'] / df['ma200'] - 1) * 100
        
        # 市场宽度指标（对于单一股票没有意义，但在多股票组合中有用）
        # 这里只是一个示例，实际使用中可能需要更复杂的计算
        
        return df
    
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """根据市场环境调整信号"""
        # 调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 在组合策略中，我们可以为不同的市场环境分配不同的策略权重
        # 这可以让在特定市场环境中表现更好的策略具有更大的影响力
        
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 根据市场环境动态调整策略权重
        if regime == MarketRegime.VOLATILE:
            # 在高波动市场中，可能偏好均值回归或低波动策略
            # 这里只是示例，实际应用中需要根据具体策略调整
            self.logger.info("高波动市场环境，调整策略权重")
            
            for strategy in self.strategies:
                # 增加均值回归策略的权重
                if "MeanReversion" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.5
                # 增加TDI策略的权重（适合高波动市场）
                elif "TDIStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.4
                # 降低趋势跟踪策略的权重
                elif "Momentum" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 0.7
                # 降低CPGW策略的权重，因为其容易在高波动市场产生过多信号
                elif "CPGWStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 0.6
                    
            # 重新归一化权重
            weight_sum = sum(self.strategy_weights.values())
            if weight_sum > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= weight_sum
                    
        elif regime == MarketRegime.BULLISH:
            # 在牛市中，可能偏好动量或趋势跟踪策略
            self.logger.info("牛市环境，调整策略权重")
            
            for strategy in self.strategies:
                # 增加动量策略的权重
                if "Momentum" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.3
                # 增加CPGW策略的权重，因为其在趋势市场中表现良好
                elif "CPGWStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.3
                # 增加突破策略的权重
                elif "Breakout" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.2
                # 增加黄金三角策略的权重
                elif "GoldTriangleStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.3
                    
            # 重新归一化权重
            weight_sum = sum(self.strategy_weights.values())
            if weight_sum > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= weight_sum
        
        elif regime == MarketRegime.BEARISH:
            # 在熊市中，可能偏好反向或保守策略
            self.logger.info("熊市环境，调整策略权重")
            
            for strategy in self.strategies:
                # 增加均值回归策略的权重
                if "MeanReversion" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.2
                # 增加TDI策略的权重
                elif "TDIStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.2
                # 增加Market Forecast策略的权重
                elif "MarketForecastStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.3
                # 减少趋势跟踪策略的权重
                elif "Momentum" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 0.8
                    
            # 重新归一化权重
            weight_sum = sum(self.strategy_weights.values())
            if weight_sum > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= weight_sum
        
        elif regime == MarketRegime.RANGING:
            # 在震荡市场中，可能偏好均值回归策略
            self.logger.info("震荡市场环境，调整策略权重")
            
            for strategy in self.strategies:
                # 增加均值回归策略的权重
                if "MeanReversion" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.5
                # 增加TDI策略的权重
                elif "TDIStrategy" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 1.3
                # 减少趋势跟踪和突破策略的权重
                elif "Momentum" in strategy.__class__.__name__ or "Breakout" in strategy.__class__.__name__:
                    self.strategy_weights[strategy.name] *= 0.7
                    
            # 重新归一化权重
            weight_sum = sum(self.strategy_weights.values())
            if weight_sum > 0:
                for name in self.strategy_weights:
                    self.strategy_weights[name] /= weight_sum
        
        return adjusted_signals
    
    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """计算仓位大小"""
        # 调用基类的仓位计算
        base_position = super().get_position_size(data, signal)
        
        # 在组合策略中，可以考虑各个策略的一致性来调整仓位
        # 一致性越高，仓位越大
        
        # 计算信号一致性
        consensus = 0
        if hasattr(self, 'strategies') and self.strategies:
            # 计算同向信号的策略数量
            same_direction = 0
            for strategy in self.strategies:
                strategy_name = strategy.name
                if f"signal_{strategy_name}" in data.columns:
                    if signal > 0 and data[f"signal_{strategy_name}"].iloc[-1] > 0:
                        same_direction += 1
                    elif signal < 0 and data[f"signal_{strategy_name}"].iloc[-1] < 0:
                        same_direction += 1
                        
            # 计算一致性
            consensus = same_direction / len(self.strategies) if self.strategies else 0
            
            # 根据一致性调整仓位
            consensus_factor = 0.7 + (consensus * 0.3)  # 保证至少有70%的基础仓位
            base_position *= consensus_factor
            
        # 应用整体风险限制
        return min(base_position, self.parameters['overall_risk_limit'])
    
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止损价格"""
        # 基于波动率的动态止损
        if 'volatility' in data.columns:
            volatility = data['volatility'].iloc[-1]
            
            # 根据波动率确定止损百分比
            stop_pct = min(0.03 + (volatility * 0.5), 0.10)  # 止损范围：3%-10%
            
            if position > 0:  # 多头止损
                return entry_price * (1 - stop_pct)
            elif position < 0:  # 空头止损
                return entry_price * (1 + stop_pct)
                
        # 如果没有波动率数据，使用基类的止损计算方法
        return super().get_stop_loss(data, entry_price, position)
    
    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """计算止盈价格"""
        # 尝试从子策略获取止盈价格
        take_profits = []
        for strategy in self.strategies:
            try:
                tp = strategy.get_take_profit(data, entry_price, position)
                if tp > 0:
                    take_profits.append(tp)
            except Exception:
                pass  # 忽略获取止盈失败的策略
                
        if take_profits:
            # 使用中位数作为组合策略的止盈价格
            median_tp = np.median(take_profits)
            return median_tp
            
        # 如果无法从子策略获取，使用基类的止盈计算方法
        return super().get_take_profit(data, entry_price, position)
    
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float,
                                stop_loss: float, position: int) -> float:
        """是否应该调整止损价格(追踪止损)"""
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 在强趋势市场中，使用追踪止损
        if ((regime == MarketRegime.BULLISH and position > 0) or
            (regime == MarketRegime.BEARISH and position < 0)):
            
            # 计算利润百分比
            profit_pct = (current_price / entry_price - 1) * 100 if position > 0 else (1 - current_price / entry_price) * 100
            
            # 如果已有一定利润，开始调整止损
            if profit_pct > 5:  # 5%利润
                new_stop = current_price * (0.97 if position > 0 else 1.03)  # 3%追踪止损
                
                # 仅在新止损更有利时更新
                if (position > 0 and new_stop > stop_loss) or (position < 0 and new_stop < stop_loss):
                    return new_stop
                    
        return stop_loss 