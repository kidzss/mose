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

logger = logging.getLogger(__name__)

class CombinedStrategy(Strategy):
    def __init__(self, config_path: str = None):
        """
        初始化组合策略
        
        Args:
            config_path: 配置文件路径
        """
        # 初始化基础策略
        super().__init__('CombinedStrategy')
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化各个策略
        self.strategies = {
            # 短期策略
            'intraday': IntradayMomentumStrategy(),
            'breakout': BreakoutStrategy(),
            
            # 中期策略
            'trend': TrendFollowingStrategy(),
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'bollinger': BollingerBandsStrategy(),
            
            # 长期策略
            'cpgw': CPGWStrategy(),
            'niuniu': NiuniuStrategyV3()
        }
        
        # 初始化仓位管理器
        self.position_manager = PositionManager()
        
        # 初始化市场分析器
        self.market_analyzer = MarketAnalysis()
        
        # 初始化市场情绪策略
        self.sentiment_strategy = MarketSentimentStrategy()
        
        # 设置冷却期
        self.cooldown_period = 10  # 增加冷却期到10天
        
        # 设置风险参数
        self.risk_params = {
            'max_drawdown': 0.25,  # 最大回撤25%
            'volatility_limit': 0.35,  # 波动率限制35%
            'position_limit': 0.9  # 仓位限制90%
        }
        
        # 设置权重调整参数
        self.weight_adjust_params = {
            'market_regime_factor': 1.15,  # 市场环境调整因子
            'volatility_factor': 1.1,  # 波动率调整因子
            'sentiment_factor': 1.05,  # 情绪调整因子
            'min_adjust_interval': 5  # 最小调整间隔（天）
        }
        
        # 记录上次调整时间
        self.last_adjust_time = None
        
        # 设置基础权重
        self.base_weights = {
            'intraday': 0.05,
            'breakout': 0.05,
            'trend': 0.10,
            'momentum': 0.10,
            'mean_reversion': 0.10,
            'bollinger': 0.10,
            'cpgw': 0.25,
            'niuniu': 0.25
        }
        
        # 设置权重限制
        self.weight_limits = {
            'intraday': (0.02, 0.08),
            'breakout': (0.02, 0.08),
            'trend': (0.05, 0.15),
            'momentum': (0.05, 0.15),
            'mean_reversion': (0.05, 0.15),
            'bollinger': (0.05, 0.15),
            'cpgw': (0.20, 0.30),
            'niuniu': (0.20, 0.30)
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
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 计算各个策略的指标
            for strategy in self.strategies.values():
                df = strategy.calculate_indicators(df)
            
            # 计算市场情绪指标
            df = self.sentiment_strategy.calculate_indicators(df)
            
            # 计算市场分析指标
            df = self.market_analyzer.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"计算指标时出错: {str(e)}")
            return df
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        try:
            # 生成各个策略的信号
            signals = {}
            for name, strategy in self.strategies.items():
                signals[name] = strategy.generate_signals(df)
            
            # 分析市场环境
            market_regime = self.market_analyzer.analyze_market_regime(df)
            volatility_regime = self.market_analyzer.analyze_volatility_regime(df)
            
            # 获取市场情绪
            sentiment = self.sentiment_strategy.analyze_sentiment(df)
            
            # 根据市场波动率调整仓位限制
            self.position_manager.adjust_position_limits(df['volatility'].iloc[-1])
            self.logger.info(f"Position limits adjusted based on volatility: {self.position_manager.position_limits}")
            
            # 调整策略权重
            if self._check_cooldown():
                adjusted_weights = self._adjust_weights(df)
                self.last_adjust_time = datetime.now()
            else:
                adjusted_weights = self.base_weights
            
            # 处理每个策略的信号
            for name, signal_df in signals.items():
                # 获取当前价格
                current_price = df['close'].iloc[-1]
                
                # 检查止损和止盈
                stop_loss_positions = self.position_manager.check_stop_loss(df['symbol'].iloc[-1], current_price)
                take_profit_positions = self.position_manager.check_take_profit(df['symbol'].iloc[-1], current_price)
                
                # 记录止损止盈
                if stop_loss_positions:
                    self.logger.info(f"Stop loss triggered for {df['symbol'].iloc[-1]}: {stop_loss_positions}")
                if take_profit_positions:
                    self.logger.info(f"Take profit triggered for {df['symbol'].iloc[-1]}: {take_profit_positions}")
                
                # 处理买入信号
                if signal_df['signal'].iloc[-1] > 0:
                    # 计算仓位大小
                    position_size = self._calculate_position_size(
                        name, 
                        signal_df['signal'].iloc[-1],
                        df['volatility'].iloc[-1]
                    )
                    
                    # 尝试开仓
                    if self.position_manager.can_open_position(name, df['symbol'].iloc[-1], position_size):
                        if self.position_manager.open_position(name, df['symbol'].iloc[-1], position_size, current_price):
                            self.logger.info(f"Opened position for {name} on {df['symbol'].iloc[-1]}: size={position_size:.4f}, price={current_price:.2f}")
                
                # 处理卖出信号
                elif signal_df['signal'].iloc[-1] < 0:
                    # 获取当前仓位
                    current_position = self.position_manager.get_position_size(name, df['symbol'].iloc[-1])
                    
                    # 如果有仓位，平仓
                    if current_position > 0:
                        pnl = self.position_manager.close_position(name, df['symbol'].iloc[-1], current_price)
                        self.logger.info(f"Closed position for {name} on {df['symbol'].iloc[-1]}: size={current_position:.4f}, price={current_price:.2f}, pnl={pnl:.2f}")
            
            # 合并信号
            df['signal'] = 0
            for name, signal_df in signals.items():
                df['signal'] += signal_df['signal'] * adjusted_weights[name] * self.position_manager.get_position_size(name, df['symbol'].iloc[-1])
            
            # 记录当前仓位
            self.logger.info(f"Current positions: {self.position_manager.get_current_positions()}")
            
            # 应用风险管理规则
            df = self._apply_risk_management(df)
            
            return df
            
        except Exception as e:
            logger.error(f"生成信号时出错: {str(e)}")
            return df
            
    def _calculate_position_size(self, strategy_name: str, signal_strength: float, market_volatility: float) -> float:
        """计算仓位大小"""
        stype = self.position_manager.get_strategy_type(strategy_name)
        if not stype:
            return 0.0
        
        # 基础仓位大小
        base_size = self.position_manager.position_limits[stype]
        
        # 根据市场波动率调整基础仓位
        if market_volatility > 0.3:  # 高波动率
            base_size *= 0.8
        elif market_volatility < 0.1:  # 低波动率
            base_size *= 1.2
        
        # 根据信号强度调整仓位
        position_size = base_size * abs(signal_strength)
        
        return position_size
    
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
                adjusted_weights['trend'] *= 1.2
                adjusted_weights['momentum'] *= 1.2
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            elif market_regime == 'bearish':
                # 熊市环境下增加反转和均值回归策略权重
                adjusted_weights['mean_reversion'] *= 1.2
                adjusted_weights['bollinger'] *= 1.2
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            else:  # neutral
                # 震荡市环境下增加日内和突破策略权重
                adjusted_weights['intraday'] *= 1.2
                adjusted_weights['breakout'] *= 1.2
                adjusted_weights['cpgw'] *= 1.1
                adjusted_weights['niuniu'] *= 1.1
            
            # 根据波动率调整权重
            if volatility_regime == 'high':
                # 高波动环境下降低风险策略权重
                for strategy in ['trend', 'momentum']:
                    adjusted_weights[strategy] *= 0.8
                # 增加稳健策略权重
                adjusted_weights['cpgw'] *= 1.2
                adjusted_weights['niuniu'] *= 1.2
            else:  # low
                # 低波动环境下增加趋势策略权重
                for strategy in ['trend', 'momentum']:
                    adjusted_weights[strategy] *= 1.2
            
            # 根据市场情绪调整权重
            if sentiment == 'bullish':
                # 看多情绪下增加趋势策略权重
                adjusted_weights['trend'] *= 1.1
                adjusted_weights['momentum'] *= 1.1
            elif sentiment == 'bearish':
                # 看空情绪下增加反转策略权重
                adjusted_weights['mean_reversion'] *= 1.1
                adjusted_weights['bollinger'] *= 1.1
            
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