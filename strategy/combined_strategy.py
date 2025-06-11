import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

from .strategy_base import Strategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .cpgw_strategy import CPGWStrategy
from .tdi_strategy import TDIStrategy
from .market_analysis import MarketAnalysis

logger = logging.getLogger(__name__)

class CombinedStrategy(Strategy):
    """
    简化的组合策略类
    整合三个核心策略：NiuniuV3、TDI、CPGW
    """
    
    def __init__(self, name: str = "Optimized Combined Strategy", parameters: dict = None):
        """初始化策略"""
        default_params = {
            # 核心策略权重
            'weight_niuniu': 0.50,   # 主要策略
            'weight_tdi': 0.30,      # 短期策略
            'weight_cpgw': 0.20,     # 补充策略
            
            # 信号阈值
            'signal_threshold': 0.6,  # 信号确认阈值
            'consensus_required': 2,  # 至少需要2个策略同意
            
            # 风险管理
            'max_position_size': 0.3,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0,
            
            # 市场环境适应
            'use_market_adaptation': True,
            'volatility_threshold': 0.02,
        }
        
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)
        
        # 初始化核心策略
        try:
            self.niuniu_strategy = NiuniuStrategyV3()
            self.tdi_strategy = TDIStrategy()
            self.cpgw_strategy = CPGWStrategy()
            
            # 市场分析器
            self.market_analyzer = MarketAnalysis()
            
            logger.info("✅ 组合策略初始化成功 - 集成了3个核心策略")
        except Exception as e:
            logger.error(f"❌ 组合策略初始化失败: {e}")
            raise
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有策略需要的技术指标"""
        df = data.copy()
        
        try:
            # 让每个策略计算自己的指标
            df = self.niuniu_strategy.calculate_indicators(df)
            df = self.tdi_strategy.calculate_indicators(df)
            df = self.cpgw_strategy.calculate_indicators(df)
            
            # 添加一些组合策略特有的指标
            df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            return df
        except Exception as e:
            logger.error(f"计算指标时出错: {e}")
            return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成组合交易信号"""
        df = data.copy()
        
        try:
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 获取各策略信号
            niuniu_signals = self.niuniu_strategy.generate_signals(df)
            tdi_signals = self.tdi_strategy.generate_signals(df)
            cpgw_signals = self.cpgw_strategy.generate_signals(df)
            
            # 提取信号值
            niuniu_signal = niuniu_signals['signal'] if 'signal' in niuniu_signals.columns else 0
            tdi_signal = tdi_signals['signal'] if 'signal' in tdi_signals.columns else 0
            cpgw_signal = cpgw_signals['signal'] if 'signal' in cpgw_signals.columns else 0
            
            # 计算加权组合信号
            combined_signal = (
                self.parameters['weight_niuniu'] * niuniu_signal +
                self.parameters['weight_tdi'] * tdi_signal +
                self.parameters['weight_cpgw'] * cpgw_signal
            )
            
            # 应用市场环境调整
            if self.parameters['use_market_adaptation']:
                combined_signal = self._apply_market_adaptation(df, combined_signal)
            
            # 信号过滤和标准化
            final_signal = self._filter_signals(combined_signal, niuniu_signal, tdi_signal, cpgw_signal)
            
            df['signal'] = final_signal
            df['signal_strength'] = abs(combined_signal)
            
            # 保存各策略的原始信号用于分析
            df['niuniu_signal'] = niuniu_signal
            df['tdi_signal'] = tdi_signal
            df['cpgw_signal'] = cpgw_signal
            
            return df
            
        except Exception as e:
            logger.error(f"生成信号时出错: {e}")
            df['signal'] = 0
            return df
    
    def _apply_market_adaptation(self, data: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """根据市场环境调整信号"""
        try:
            # 检查volatility列是否存在
            if 'volatility' not in data.columns or data['volatility'].empty:
                return signal
            
            current_volatility = data['volatility'].iloc[-1] if len(data) > 0 else 0
            
            # 高波动市场中降低信号强度
            if pd.notna(current_volatility) and current_volatility > self.parameters['volatility_threshold']:
                adjustment_factor = 0.7
                logger.debug(f"高波动市场，调整信号强度：{adjustment_factor}")
                signal = signal * adjustment_factor
            
            return signal
        except Exception as e:
            logger.error(f"市场适应调整出错: {e}")
            return signal
    
    def _filter_signals(self, combined_signal: pd.Series, niuniu_signal, 
                       tdi_signal, cpgw_signal) -> pd.Series:
        """过滤和标准化信号"""
        try:
            # 处理不同类型的信号输入
            def _extract_signal_value(signal, index):
                if isinstance(signal, pd.Series):
                    return signal.iloc[index] if len(signal) > index else 0
                elif isinstance(signal, (int, float)):
                    return signal
                else:
                    return 0
            
            if isinstance(combined_signal, pd.Series):
                final_signal = pd.Series(0, index=combined_signal.index)
            else:
                # 如果combined_signal不是Series，创建默认的
                final_signal = pd.Series(0, index=range(len(combined_signal)) if hasattr(combined_signal, '__len__') else range(1))
            
            for i in range(len(final_signal)):
                # 提取各策略信号值
                niuniu_val = _extract_signal_value(niuniu_signal, i)
                tdi_val = _extract_signal_value(tdi_signal, i)
                cpgw_val = _extract_signal_value(cpgw_signal, i)
                
                signals = [niuniu_val, tdi_val, cpgw_val]
                combined_val = _extract_signal_value(combined_signal, i)
                
                # 计算同向信号数量
                positive_signals = sum(1 for s in signals if s > 0)
                negative_signals = sum(1 for s in signals if s < 0)
                
                # 需要足够的共识才产生信号
                if positive_signals >= self.parameters['consensus_required'] and combined_val > self.parameters['signal_threshold']:
                    final_signal.iloc[i] = 1
                elif negative_signals >= self.parameters['consensus_required'] and combined_val < -self.parameters['signal_threshold']:
                    final_signal.iloc[i] = -1
                else:
                    final_signal.iloc[i] = 0
                    
            return final_signal
            
        except Exception as e:
            logger.error(f"信号过滤出错: {e}")
            # 返回安全的默认信号
            if isinstance(combined_signal, pd.Series):
                return pd.Series(0, index=combined_signal.index)
            else:
                return pd.Series(0, index=range(1))
    
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """计算建议仓位大小"""
        try:
            if signal == 0:
                return 0.0
            
            # 基础仓位大小
            base_size = 0.1
            
            # 根据信号强度调整
            if 'signal_strength' in data.columns and not data['signal_strength'].empty:
                signal_strength = data['signal_strength'].iloc[-1]
                size_multiplier = min(signal_strength * 2, 1.0)  # 最大不超过100%
                base_size *= size_multiplier
            
            # 应用最大仓位限制
            return min(base_size, self.parameters['max_position_size'])
            
        except Exception as e:
            logger.error(f"计算仓位大小出错: {e}")
            return 0.05  # 默认5%仓位
    
    def get_stop_loss(self, data: pd.DataFrame, current_price: float, direction: int) -> float:
        """获取止损价格"""
        try:
            if 'ATR' in data.columns and not data['ATR'].empty:
                atr = data['ATR'].iloc[-1]
                stop_distance = atr * self.parameters['stop_loss_atr']
            else:
                # 如果没有ATR，使用价格的2%作为止损
                stop_distance = current_price * 0.02
            
            if direction > 0:  # 多头
                return current_price - stop_distance
            else:  # 空头
                return current_price + stop_distance
                
        except Exception as e:
            logger.error(f"计算止损价格出错: {e}")
            # 默认2%止损
            return current_price * (0.98 if direction > 0 else 1.02)
    
    def get_take_profit(self, data: pd.DataFrame, current_price: float, direction: int) -> float:
        """获取止盈价格"""
        try:
            if 'ATR' in data.columns and not data['ATR'].empty:
                atr = data['ATR'].iloc[-1]
                profit_distance = atr * self.parameters['take_profit_atr']
            else:
                # 如果没有ATR，使用价格的3%作为止盈
                profit_distance = current_price * 0.03
            
            if direction > 0:  # 多头
                return current_price + profit_distance
            else:  # 空头
                return current_price - profit_distance
                
        except Exception as e:
            logger.error(f"计算止盈价格出错: {e}")
            # 默认3%止盈
            return current_price * (1.03 if direction > 0 else 0.97)
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略配置摘要"""
        return {
            'name': self.name,
            'strategies': ['NiuniuV3', 'TDI', 'CPGW'],
            'weights': {
                'niuniu': self.parameters['weight_niuniu'],
                'tdi': self.parameters['weight_tdi'],
                'cpgw': self.parameters['weight_cpgw']
            },
            'signal_threshold': self.parameters['signal_threshold'],
            'consensus_required': self.parameters['consensus_required'],
            'max_position_size': self.parameters['max_position_size']
        } 