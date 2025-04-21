import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

from .strategy_base import Strategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .cpgw_strategy import CPGWStrategy
from .market_analysis import MarketAnalysis
from .market_sentiment_strategy import MarketSentimentStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .breakout_strategy import BreakoutStrategy
from .intraday_momentum_strategy import IntradayMomentumStrategy
from .position_manager import PositionManager
from .tdi_strategy import TDIStrategy
from .uss_gold_triangle_risk import USSGoldTriangleRisk

logger = logging.getLogger(__name__)

class CombinedStrategy(Strategy):
    """
    组合策略类
    将多个策略的信号组合成一个最终信号
    """
    
    def __init__(self, name: str = "Combined Strategy", parameters: dict = None):
        """初始化策略"""
        default_params = {
            'weight_tdi': 0.4,
            'weight_niuniu': 0.3,
            'weight_gold_triangle': 0.3,
        }
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)
        
        # 初始化子策略
        self.tdi_strategy = TDIStrategy()
        self.niuniu_strategy = NiuniuStrategyV3()
        self.gold_triangle_strategy = USSGoldTriangleRisk()
        
        # 设置基础参数
        self.parameters = default_params.copy()  # 使用default_params作为基础参数
        self.parameters.update({
            'cooldown_period': 10,
            'risk_params': {
                'max_drawdown': 0.25,
                'volatility_limit': 0.35,
                'position_limit': 0.9
            },
            'weight_adjust_params': {
                'market_regime_factor': 1.15,
                'volatility_factor': 1.1,
                'sentiment_factor': 1.05,
                'min_adjust_interval': 5
            }
        })
        
        # 初始化各个策略
        self.strategies = {
            # 长期策略
            'cpgw': CPGWStrategy(),
            'niuniu': NiuniuStrategyV3(),
            
            # 短期策略
            'intraday': IntradayMomentumStrategy(),
            'breakout': BreakoutStrategy(),
            
            # 中期策略
            'trend': TrendFollowingStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'bollinger': BollingerBandsStrategy(),
        }
        
        # 初始化仓位管理器
        self.position_manager = PositionManager()
        
        # 初始化市场分析器
        self.market_analyzer = MarketAnalysis()
        
        # 初始化市场情绪策略
        self.sentiment_strategy = MarketSentimentStrategy()
        
        # 设置冷却期
        self.cooldown_period = self.parameters['cooldown_period']
        
        # 设置风险参数
        self.risk_params = self.parameters['risk_params']
        
        # 设置权重调整参数
        self.weight_adjust_params = self.parameters['weight_adjust_params']
        
        # 记录上次调整时间
        self.last_adjust_time = None
        
        # 设置基础权重
        self.base_weights = {
            # 长期策略
            'cpgw': 0.25,
            'niuniu': 0.25,
            
            # 短期策略
            'intraday': 0.10,
            'breakout': 0.10,
            
            # 中期策略
            'trend': 0.10,
            'momentum': 0.10,
            'mean_reversion': 0.05,
            'bollinger': 0.05,
        }
        
        # 设置权重限制
        self.weight_limits = {
            # 长期策略
            'cpgw': (0.15, 0.35),
            'niuniu': (0.15, 0.35),
            
            # 短期策略
            'intraday': (0.05, 0.15),
            'breakout': (0.05, 0.15),
            
            # 中期策略
            'trend': (0.05, 0.15),
            'momentum': (0.05, 0.15),
            'mean_reversion': (0.02, 0.08),
            'bollinger': (0.02, 0.08),
        }
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 记录初始化完成
        self.logger.info("CombinedStrategy initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的技术指标"""
        df = data.copy()
        
        # 计算各个子策略的指标
        df = self.tdi_strategy.calculate_indicators(df)
        df = self.niuniu_strategy.calculate_indicators(df)
        df = self.gold_triangle_strategy.calculate_indicators(df)
        
        return df
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = data.copy()
        
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        # 获取各个策略的信号
        tdi_signals = self.tdi_strategy.generate_signals(df)['signal']
        niuniu_signals = self.niuniu_strategy.generate_signals(df)['signal']
        gold_triangle_signals = self.gold_triangle_strategy.generate_signals(df)['signal']
        
        # 加权组合信号
        df['signal'] = (
            self.parameters['weight_tdi'] * tdi_signals +
            self.parameters['weight_niuniu'] * niuniu_signals +
            self.parameters['weight_gold_triangle'] * gold_triangle_signals
        )
        
        # 信号标准化
        df['signal'] = np.sign(df['signal'])  # 转换为 -1, 0, 1
        
        return df
        
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """获取仓位大小"""
        # 使用子策略的平均建议仓位
        tdi_size = self.tdi_strategy.get_position_size(data, signal)
        niuniu_size = self.niuniu_strategy.get_position_size(data, signal)
        gold_triangle_size = self.gold_triangle_strategy.get_position_size(data, signal)
        
        return (tdi_size + niuniu_size + gold_triangle_size) / 3
        
    def get_stop_loss(self, data: pd.DataFrame, current_price: float, direction: int) -> float:
        """获取止损价格"""
        # 使用最保守的止损价格
        tdi_stop = self.tdi_strategy.get_stop_loss(data, current_price, direction)
        niuniu_stop = self.niuniu_strategy.get_stop_loss(data, current_price, direction)
        gold_triangle_stop = self.gold_triangle_strategy.get_stop_loss(data, current_price, direction)
        
        if direction > 0:
            return max(tdi_stop, niuniu_stop, gold_triangle_stop)
        else:
            return min(tdi_stop, niuniu_stop, gold_triangle_stop)
        
    def get_take_profit(self, data: pd.DataFrame, current_price: float, direction: int) -> float:
        """获取止盈价格"""
        # 使用最激进的止盈价格
        tdi_tp = self.tdi_strategy.get_take_profit(data, current_price, direction)
        niuniu_tp = self.niuniu_strategy.get_take_profit(data, current_price, direction)
        gold_triangle_tp = self.gold_triangle_strategy.get_take_profit(data, current_price, direction)
        
        if direction > 0:
            return min(tdi_tp, niuniu_tp, gold_triangle_tp)
        else:
            return max(tdi_tp, niuniu_tp, gold_triangle_tp)
    
    def _adjust_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        根据市场环境动态调整策略权重
        
        Args:
            df: 市场数据
            
        Returns:
            Dict[str, float]: 调整后的策略权重
        """
        try:
            # 获取当前市场环境
            market_regime = self.market_analyzer.analyze_market_regime(df)
            volatility_regime = self.market_analyzer.analyze_volatility_regime(df)
            sentiment = self.sentiment_strategy.analyze_sentiment(df)
            
            # 初始化调整后的权重
            adjusted_weights = self.base_weights.copy()
            
            # 根据市场环境调整权重
            if market_regime == 'bullish':
                # 牛市环境下增加趋势和动量策略权重
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            elif market_regime == 'bearish':
                # 熊市环境下增加反转和均值回归策略权重
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            else:  # neutral
                # 震荡市环境下增加日内和突破策略权重
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            
            # 根据波动率调整权重
            if volatility_regime == 'high':
                # 高波动环境下降低风险策略权重
                adjusted_weights['cpgw'] *= 0.8
                adjusted_weights['niuniu'] *= 0.8
            else:  # low
                # 低波动环境下增加趋势策略权重
                adjusted_weights['cpgw'] *= 1.2
                adjusted_weights['niuniu'] *= 1.2
            
            # 根据市场情绪调整权重
            if sentiment == 'bullish':
                # 看多情绪下增加趋势策略权重
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            elif sentiment == 'bearish':
                # 看空情绪下增加反转策略权重
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            
            # 确保权重在限制范围内
            for strategy, (min_weight, max_weight) in self.weight_limits.items():
                adjusted_weights[strategy] = max(min_weight, min(adjusted_weights[strategy], max_weight))
            
            # 归一化权重
            total_weight = sum(adjusted_weights.values())
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            
            # 记录权重调整
            self.logger.info(f"Adjusted weights: {adjusted_weights}")
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Error adjusting weights: {str(e)}")
            return self.base_weights
    
    def _apply_risk_management(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用风险管理规则
        """
        try:
            # 计算当前回撤
            if 'returns' in data.columns:
                data['equity'] = (1 + data['returns']).cumprod()
                data['drawdown'] = (data['equity'] / data['equity'].cummax() - 1)
                
                # 如果回撤超过限制，逐步减少仓位
                drawdown_mask = data['drawdown'] < -self.risk_params['max_drawdown']
                data.loc[drawdown_mask, 'signal'] = data.loc[drawdown_mask, 'signal'] * 0.5
            
            # 应用仓位限制
            data['signal'] = data['signal'].clip(
                lower=-self.risk_params['position_limit'],
                upper=self.risk_params['position_limit']
            )
            
            return data
            
        except Exception as e:
            logger.error(f"应用风险管理时出错: {str(e)}")
            return data
    
    def _check_cooldown(self) -> bool:
        """检查是否过了冷却期"""
        if self.last_adjust_time is None:
            return True
        days_since_last_adjust = (datetime.now() - self.last_adjust_time).days
        return days_since_last_adjust >= self.weight_adjust_params['min_adjust_interval'] 