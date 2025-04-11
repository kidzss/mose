import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

from .strategy_base import Strategy
from .niuniu_strategy_v3 import NiuniuStrategyV3

logger = logging.getLogger(__name__)

class CombinedStrategy(Strategy):
    def __init__(self):
        super().__init__(name="CombinedStrategy")
        
        # 初始化子策略
        self.niuniu = NiuniuStrategyV3()
        
        # 初始化权重
        self.weights = {
            'niuniu': 1.0
        }
        
        # 设置最大回撤限制
        self.max_drawdown_limit = -0.15
        
        # 设置每个策略的仓位限制
        self.position_limits = {
            'niuniu': 1.0
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        """
        try:
            # 计算每个策略的指标
            data = self.niuniu.calculate_indicators(data)
            
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
            data = self.calculate_indicators(data)
            
            # 获取每个策略的信号
            niuniu_signals = self.niuniu.generate_signals(data)
            
            # 组合信号
            data['signal'] = niuniu_signals['signal'] * self.weights['niuniu']
            
            # 应用风险管理
            data = self._apply_risk_management(data)
            
            return data
            
        except Exception as e:
            logger.error(f"生成信号时出错: {str(e)}")
            raise
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取信号组件
        """
        try:
            components = {
                'niuniu': self.niuniu.extract_signal_components(data)
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
                'description': '组合策略，整合了牛牛策略V3',
                'components': {
                    'niuniu': self.niuniu.get_signal_metadata()
                },
                'weights': self.weights,
                'position_limits': self.position_limits,
                'max_drawdown_limit': self.max_drawdown_limit
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"获取信号元数据时出错: {str(e)}")
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