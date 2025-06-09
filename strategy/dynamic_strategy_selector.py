import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from .market_environment_classifier import MarketEnvironment, MarketEnvironmentClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_selector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrategySelector")

class DynamicStrategySelector:
    """动态策略选择器：根据市场环境和历史表现动态选择最优策略"""
    
    def __init__(self, config=None):
        """
        初始化动态策略选择器
        
        参数:
        config (dict, optional): 配置参数
        """
        logger.info("初始化动态策略选择器")
        
        # 默认配置
        self.default_config = {
            'performance_history_file': 'strategy_performance_history.json',
            'recency_weight': 0.7,  # 近期表现权重
            'minimum_samples': 5,    # 最少样本数以评估策略
            'strategy_rotation_threshold': 0.2  # 切换策略的阈值
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 初始化市场环境分类器
        self.environment_classifier = MarketEnvironmentClassifier()
        
        # 初始化策略表现历史
        self.performance_history = self._load_performance_history()
        
        # 可用策略及其默认权重
        self.available_strategies = {
            'trend_following': {
                'weight': 0.5,
                'suitable_environments': [
                    MarketEnvironment.STRONG_UPTREND, 
                    MarketEnvironment.WEAK_UPTREND, 
                    MarketEnvironment.STRONG_DOWNTREND, 
                    MarketEnvironment.WEAK_DOWNTREND
                ]
            },
            'mean_reversion': {
                'weight': 0.5,
                'suitable_environments': [MarketEnvironment.RANGE_BOUND]
            },
            'breakout': {
                'weight': 0.5,
                'suitable_environments': [
                    MarketEnvironment.STRONG_UPTREND, 
                    MarketEnvironment.STRONG_DOWNTREND
                ]
            },
            'momentum': {
                'weight': 0.5,
                'suitable_environments': [
                    MarketEnvironment.STRONG_UPTREND, 
                    MarketEnvironment.STRONG_DOWNTREND
                ]
            },
            'volatility': {
                'weight': 0.5,
                'suitable_environments': [MarketEnvironment.CHOPPY]
            }
        }
        
        logger.info(f"动态策略选择器初始化完成，已加载 {len(self.available_strategies)} 个可用策略")
    
    def get_best_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        根据当前市场环境和历史表现获取最佳策略
        
        参数:
        data (DataFrame): 包含OHLCV和技术指标的DataFrame
        
        返回:
        dict: 包含最佳策略和详细信息的字典
        """
        logger.info("开始选择最佳策略...")
        
        try:
            # 1. 确定当前市场环境
            env_result = self.environment_classifier.classify_environment(data)
            current_environment = env_result['environment']
            confidence = env_result['confidence']
            
            logger.info(f"当前市场环境: {current_environment.value}, 置信度: {confidence:.2f}")
            
            # 2. 获取适合当前环境的策略
            suitable_strategies = self._get_suitable_strategies(current_environment)
            logger.info(f"适合当前环境的策略: {list(suitable_strategies.keys())}")
            
            # 3. 评估每个策略的历史表现
            strategy_scores = self._evaluate_strategy_performance(suitable_strategies, current_environment)
            logger.info("策略评分:")
            for strategy, score in strategy_scores.items():
                logger.info(f"  - {strategy}: {score:.2f}")
            
            # 4. 结合环境适合度和历史表现计算最终权重
            strategy_weights = self._calculate_strategy_weights(suitable_strategies, strategy_scores, confidence)
            
            # 5. 选择得分最高的策略作为主策略
            primary_strategy = max(strategy_weights, key=strategy_weights.get)
            logger.info(f"选择的主策略: {primary_strategy}, 权重: {strategy_weights[primary_strategy]:.2f}")
            
            # 6. 更新策略
            current_strategy_weights = {
                strategy: weight for strategy, weight in strategy_weights.items()
                if weight > 0.1  # 只保留权重大于10%的策略
            }
            
            return {
                'primary_strategy': primary_strategy,
                'environment': current_environment,
                'confidence': confidence,
                'strategy_weights': strategy_weights
            }
            
        except Exception as e:
            logger.error(f"选择最佳策略时出错: {str(e)}", exc_info=True)
            # 出错时返回默认策略
            return {
                'primary_strategy': 'trend_following',
                'environment': MarketEnvironment.UNKNOWN,
                'confidence': 0.0,
                'strategy_weights': {'trend_following': 1.0}
            }
    
    def update_performance(self, strategy: str, environment: MarketEnvironment, 
                          performance_metric: float) -> None:
        """
        更新策略在特定环境中的表现历史
        
        参数:
        strategy (str): 策略名称
        environment (MarketEnvironment): 市场环境
        performance_metric (float): 表现指标（如收益率、夏普比率等）
        """
        try:
            logger.info(f"更新策略表现: {strategy}, 环境: {environment.value}, 表现指标: {performance_metric:.2f}")
            
            # 确保策略和环境存在
            if strategy not in self.performance_history:
                self.performance_history[strategy] = {}
                
            env_name = environment.value
            if env_name not in self.performance_history[strategy]:
                self.performance_history[strategy][env_name] = []
            
            # 添加新的表现记录
            record = {
                'timestamp': datetime.now().isoformat(),
                'metric': performance_metric
            }
            
            self.performance_history[strategy][env_name].append(record)
            
            # 保存更新后的历史
            self._save_performance_history()
            logger.info("策略表现历史已更新并保存")
            
        except Exception as e:
            logger.error(f"更新策略表现时出错: {str(e)}", exc_info=True)
    
    def _get_suitable_strategies(self, environment: MarketEnvironment) -> Dict[str, Dict]:
        """获取适合当前环境的策略"""
        suitable_strategies = {}
        
        try:
            for strategy_name, strategy_info in self.available_strategies.items():
                if environment in strategy_info['suitable_environments']:
                    suitable_strategies[strategy_name] = strategy_info
                    
            if not suitable_strategies:
                logger.warning(f"没有找到适合环境 {environment.value} 的策略，使用所有可用策略")
                suitable_strategies = self.available_strategies
                
            return suitable_strategies
            
        except Exception as e:
            logger.error(f"获取适合策略时出错: {str(e)}", exc_info=True)
            return self.available_strategies
    
    def _evaluate_strategy_performance(self, strategies: Dict[str, Dict], 
                                     environment: MarketEnvironment) -> Dict[str, float]:
        """
        评估每个策略在当前环境中的历史表现
        返回每个策略的评分
        """
        strategy_scores = {}
        env_name = environment.value
        
        try:
            logger.info(f"评估策略在环境 {env_name} 中的历史表现")
            
            for strategy_name in strategies:
                # 默认分数
                default_score = 0.5
                
                # 如果有历史表现记录
                if (strategy_name in self.performance_history and 
                    env_name in self.performance_history[strategy_name] and
                    len(self.performance_history[strategy_name][env_name]) >= self.config['minimum_samples']):
                    
                    # 获取表现记录
                    records = self.performance_history[strategy_name][env_name]
                    
                    # 计算加权表现分数 (新的记录权重更高)
                    total_weight = 0
                    weighted_sum = 0
                    recency_factor = self.config['recency_weight']
                    
                    for i, record in enumerate(sorted(records, key=lambda x: x['timestamp'])):
                        # 越新的记录权重越高
                        weight = (1.0 - recency_factor) + recency_factor * (i + 1) / len(records)
                        metric_value = record['metric']
                        
                        weighted_sum += weight * metric_value
                        total_weight += weight
                    
                    score = weighted_sum / total_weight if total_weight > 0 else default_score
                    logger.info(f"  - {strategy_name}: 样本数={len(records)}, 加权得分={score:.2f}")
                else:
                    score = default_score
                    logger.info(f"  - {strategy_name}: 样本数不足，使用默认分数={score:.2f}")
                
                strategy_scores[strategy_name] = score
                
            return strategy_scores
            
        except Exception as e:
            logger.error(f"评估策略表现时出错: {str(e)}", exc_info=True)
            # 出错时为每个策略分配相同的默认分数
            return {strategy: 0.5 for strategy in strategies}
    
    def _calculate_strategy_weights(self, suitable_strategies: Dict[str, Dict],
                                  performance_scores: Dict[str, float],
                                  confidence: float) -> Dict[str, float]:
        """
        计算每个策略的最终权重
        考虑策略的适合度和历史表现
        """
        strategy_weights = {}
        
        try:
            logger.info("计算策略最终权重")
            
            for strategy_name, strategy_info in suitable_strategies.items():
                # 考虑基础权重
                base_weight = strategy_info['weight']
                
                # 考虑表现
                performance_weight = performance_scores.get(strategy_name, 0.5)
                
                # 结合基础权重和表现
                combined_weight = base_weight * 0.4 + performance_weight * 0.6
                
                strategy_weights[strategy_name] = combined_weight
            
            # 标准化权重使总和为1
            if strategy_weights:
                total_weight = sum(strategy_weights.values())
                if total_weight > 0:
                    strategy_weights = {
                        s: w/total_weight for s, w in strategy_weights.items()
                    }
                    
            logger.info("策略权重计算结果:")
            for strategy, weight in strategy_weights.items():
                logger.info(f"  - {strategy}: {weight:.2f}")
                    
            return strategy_weights
            
        except Exception as e:
            logger.error(f"计算策略权重时出错: {str(e)}", exc_info=True)
            # 出错时均匀分配权重
            strategies = list(suitable_strategies.keys())
            return {s: 1.0/len(strategies) for s in strategies} if strategies else {'trend_following': 1.0}
    
    def _load_performance_history(self) -> Dict[str, Dict]:
        """加载策略表现历史"""
        try:
            file_path = self.config['performance_history_file']
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    history = json.load(file)
                logger.info(f"从 {file_path} 加载了策略表现历史")
                return history
            else:
                logger.info(f"策略表现历史文件不存在，将创建新文件")
                return {}
        except Exception as e:
            logger.error(f"加载策略表现历史时出错: {str(e)}", exc_info=True)
            return {}
    
    def _save_performance_history(self) -> None:
        """保存策略表现历史"""
        try:
            file_path = self.config['performance_history_file']
            with open(file_path, 'w') as file:
                json.dump(self.performance_history, file, indent=2)
            logger.info(f"策略表现历史已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存策略表现历史时出错: {str(e)}", exc_info=True) 