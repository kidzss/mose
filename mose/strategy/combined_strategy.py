import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

from .strategy_base import Strategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .tdi_strategy import TDIStrategy
from .uss_market_forecast import USSMarketForecast
from .uss_gold_triangle_risk import USSGoldTriangleRisk
from .custom_strategy import CustomStrategy
from .bollinger_bands_strategy import BollingerBandsStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .momentum_strategy import MomentumStrategy

logger = logging.getLogger(__name__)

class CombinedStrategy(Strategy):
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化组合策略
        
        参数:
            config_path: 配置文件路径
        """
        super().__init__(name="CombinedStrategy")
        
        # 初始化子策略
        self.niuniu = NiuniuStrategyV3()
        self.tdi = TDIStrategy()
        self.market_forecast = USSMarketForecast()
        self.gold_triangle = USSGoldTriangleRisk()
        self.custom = CustomStrategy()
        self.bollinger = BollingerBandsStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.momentum = MomentumStrategy()
        
        # 初始化权重（使用优化后的权重）
        self.base_weights = {
            'niuniu': 0.15,          # 短期策略
            'tdi': 0.15,             # 短期策略
            'market_forecast': 0.1,   # 短期策略
            'gold_triangle': 0.1,     # 中期策略
            'custom': 0.1,           # 长期策略
            'bollinger': 0.15,       # 波动性策略
            'mean_reversion': 0.15,  # 反转策略
            'momentum': 0.1          # 短期动量策略
        }
        
        # 设置权重范围
        self.weight_limits = {
            'niuniu': {'min': 0.1, 'max': 0.3},
            'tdi': {'min': 0.1, 'max': 0.3},
            'market_forecast': {'min': 0.05, 'max': 0.2},
            'gold_triangle': {'min': 0.05, 'max': 0.2},
            'custom': {'min': 0.05, 'max': 0.2},
            'bollinger': {'min': 0.1, 'max': 0.3},
            'mean_reversion': {'min': 0.1, 'max': 0.3},
            'momentum': {'min': 0.05, 'max': 0.2}
        }
        
        # 当前权重
        self.weights = self.base_weights.copy()
        
        # 设置最大回撤限制
        self.max_drawdown_limit = -0.15
        
        # 设置每个策略的仓位限制
        self.position_limits = {
            'niuniu': 1.0,
            'tdi': 1.0,
            'market_forecast': 1.0,
            'gold_triangle': 0.8,
            'custom': 0.8,
            'bollinger': 1.0,
            'mean_reversion': 1.0,
            'momentum': 1.0
        }
        
        # 市场环境分析参数
        self.market_regime = None
        self.volatility_regime = None
        self.trend_strength = None
        
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        """
        try:
            # 计算每个策略的指标
            data = self.niuniu.calculate_indicators(data)
            data = self.tdi.calculate_indicators(data)
            data = self.market_forecast.calculate_indicators(data)
            data = self.gold_triangle.calculate_indicators(data)
            data = self.custom.calculate_indicators(data)
            data = self.bollinger.calculate_indicators(data)
            data = self.mean_reversion.calculate_indicators(data)
            data = self.momentum.calculate_indicators(data)
            
            # 分析市场环境
            self._analyze_market_environment(data)
            
            # 根据市场环境调整权重
            self._adjust_weights()
            
            return data
            
        except Exception as e:
            logger.error(f"计算指标时出错: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        try:
            # 计算指标
            df = self.calculate_indicators(data)
            
            # 获取每个策略的信号
            niuniu_signals = self.niuniu.generate_signals(data)
            tdi_signals = self.tdi.generate_signals(data)
            market_forecast_signals = self.market_forecast.generate_signals(data)
            gold_triangle_signals = self.gold_triangle.generate_signals(data)
            custom_signals = self.custom.generate_signals(data)
            bollinger_signals = self.bollinger.generate_signals(data)
            mean_reversion_signals = self.mean_reversion.generate_signals(data)
            momentum_signals = self.momentum.generate_signals(data)
            
            # 组合信号
            df['signal'] = (
                niuniu_signals['signal'] * self.weights['niuniu'] +
                tdi_signals['signal'] * self.weights['tdi'] +
                market_forecast_signals['signal'] * self.weights['market_forecast'] +
                gold_triangle_signals['signal'] * self.weights['gold_triangle'] +
                custom_signals['signal'] * self.weights['custom'] +
                bollinger_signals['signal'] * self.weights['bollinger'] +
                mean_reversion_signals['signal'] * self.weights['mean_reversion'] +
                momentum_signals['signal'] * self.weights['momentum']
            )
            
            # 应用风险管理
            df = self._apply_risk_management(df)
            
            return df
            
        except Exception as e:
            logger.error(f"生成信号时出错: {str(e)}")
            raise
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        """
        try:
            components = {
                'niuniu': self.niuniu.extract_signal_components(data),
                'tdi': self.tdi.extract_signal_components(data),
                'market_forecast': self.market_forecast.extract_signal_components(data),
                'gold_triangle': self.gold_triangle.extract_signal_components(data),
                'custom': self.custom.extract_signal_components(data),
                'bollinger': self.bollinger.extract_signal_components(data),
                'mean_reversion': self.mean_reversion.extract_signal_components(data),
                'momentum': self.momentum.extract_signal_components(data)
            }
            
            return components
            
        except Exception as e:
            logger.error(f"提取信号组件时出错: {str(e)}")
            raise
    
    def get_signal_metadata(self) -> Dict[str, Any]:
        """
        获取信号元数据
        """
        try:
            metadata = {
                'strategy_name': 'CombinedStrategy',
                'version': '1.0.0',
                'description': '组合策略，整合了多个策略',
                'components': {
                    'niuniu': self.niuniu.get_signal_metadata(),
                    'tdi': self.tdi.get_signal_metadata(),
                    'market_forecast': self.market_forecast.get_signal_metadata(),
                    'gold_triangle': self.gold_triangle.get_signal_metadata(),
                    'custom': self.custom.get_signal_metadata(),
                    'bollinger': self.bollinger.get_signal_metadata(),
                    'mean_reversion': self.mean_reversion.get_signal_metadata(),
                    'momentum': self.momentum.get_signal_metadata()
                },
                'weights': self.weights,
                'position_limits': self.position_limits,
                'max_drawdown_limit': self.max_drawdown_limit,
                'market_regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'trend_strength': self.trend_strength
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"获取信号元数据时出错: {str(e)}")
            raise
    
    def _analyze_market_environment(self, data: pd.DataFrame) -> None:
        """
        分析市场环境
        """
        try:
            # 分析市场趋势
            trend_signals = self.gold_triangle.extract_signal_components(data)
            momentum_signals = self.momentum.extract_signal_components(data)
            
            # 使用趋势和动量指标而不是信号
            trend_strength = trend_signals.get('trend_strength')
            adx = momentum_signals.get('adx')
            
            if trend_strength is not None and adx is not None:
                self.trend_strength = float(
                    trend_strength.iloc[-1] * 0.7 +
                    adx.iloc[-1] * 0.3
                )
            else:
                self.trend_strength = 0.0
            
            # 分析市场波动性
            volatility = data['close'].pct_change().rolling(window=20).std()
            self.volatility_regime = 'high' if volatility.iloc[-1] > volatility.mean() else 'low'
            
            # 分析市场状态
            if self.trend_strength > 0.7:
                self.market_regime = 'bullish'
            elif self.trend_strength < -0.7:
                self.market_regime = 'bearish'
            else:
                self.market_regime = 'neutral'
                
        except Exception as e:
            logger.error(f"分析市场环境时出错: {str(e)}")
            raise
    
    def _adjust_weights(self) -> None:
        """
        根据市场环境调整策略权重
        """
        try:
            # 重置为基准权重
            self.weights = self.base_weights.copy()
            
            # 根据市场环境调整权重
            if self.market_regime == 'bullish':
                # 在牛市中增加趋势跟踪和动量策略的权重
                self.weights['gold_triangle'] *= 1.5
                self.weights['momentum'] *= 1.5
                # 减少反转策略的权重
                self.weights['mean_reversion'] *= 0.7
                self.weights['bollinger'] *= 0.8
                
            elif self.market_regime == 'bearish':
                # 在熊市中增加反转和波动性策略的权重
                self.weights['mean_reversion'] *= 1.5
                self.weights['bollinger'] *= 1.5
                # 减少趋势跟踪策略的权重
                self.weights['gold_triangle'] *= 0.7
                self.weights['momentum'] *= 0.7
                
            # 根据波动性调整权重
            if self.volatility_regime == 'high':
                # 在高波动性环境中增加波动性策略的权重
                self.weights['bollinger'] *= 1.3
                self.weights['mean_reversion'] *= 1.2
                # 减少其他策略的权重
                for strategy in ['niuniu', 'tdi', 'market_forecast', 'gold_triangle', 'custom', 'momentum']:
                    self.weights[strategy] *= 0.9
            
            # 确保权重在限制范围内
            for strategy in self.weights:
                min_weight = self.weight_limits[strategy]['min']
                max_weight = self.weight_limits[strategy]['max']
                self.weights[strategy] = np.clip(self.weights[strategy], min_weight, max_weight)
            
            # 确保权重总和为1
            total_weight = sum(self.weights.values())
            for strategy in self.weights:
                self.weights[strategy] /= total_weight
                
        except Exception as e:
            logger.error(f"调整权重时出错: {str(e)}")
            raise
    
    def _apply_risk_management(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用风险管理规则
        """
        try:
            # 计算当前回撤
            data['equity'] = (1 + data['returns']).cumprod()
            data['drawdown'] = (data['equity'] / data['equity'].cummax() - 1)
            
            # 如果回撤超过限制，清空仓位
            data.loc[data['drawdown'] < self.max_drawdown_limit, 'signal'] = 0
            
            # 应用仓位限制
            data['signal'] = data['signal'].clip(
                lower=-max(self.position_limits.values()),
                upper=max(self.position_limits.values())
            )
            
            return data
            
        except Exception as e:
            logger.error(f"应用风险管理时出错: {str(e)}")
            raise 