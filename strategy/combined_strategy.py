import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from sklearn.model_selection import train_test_split
from .strategy_base import Strategy
from .niuniu_strategy_v3 import NiuniuStrategyV3
from .cpgw_strategy import CPGWStrategy
from .custom_cpgw_strategy import CustomCPGWStrategy

logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率
    """
    if len(returns) < 2:
        return 0.0
    
    # 计算年化收益率
    annual_return = (1 + returns.mean()) ** 252 - 1
    
    # 计算年化波动率
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 计算夏普比率
    if annual_volatility == 0:
        return 0.0
    
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    return sharpe

class CombinedStrategy(Strategy):
    def __init__(self, name: str = "CombinedStrategy"):
        super().__init__(name=name)
        self.niuniu = NiuniuStrategyV3()
        self.cpgw = CPGWStrategy()
        self.custom_cpgw = CustomCPGWStrategy()
        
        # 初始化权重
        self.weights = {
            'niuniu': 0.4,
            'cpgw': 0.3,
            'custom_cpgw': 0.3
        }
        
        # 参数优化范围
        self.param_ranges = {
            'niuniu_weight': (0.2, 0.6),
            'cpgw_weight': (0.2, 0.6),
            'custom_cpgw_weight': (0.2, 0.6),
            'niuniu_rsi_period': (5, 30),
            'niuniu_adx_period': (5, 30),
            'cpgw_ema_period': (5, 30),
            'custom_cpgw_ema_period': (5, 30)
        }
        
        # 存储每个股票的最优参数
        self.optimal_params = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        """
        try:
            # 计算各个策略的指标
            data = self.niuniu.calculate_indicators(data)
            data = self.cpgw.calculate_indicators(data)
            data = self.custom_cpgw.calculate_indicators(data)
            return data
        except Exception as e:
            logger.error(f"计算指标时出错: {str(e)}")
            raise

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        try:
            # 获取各个策略的信号
            niuniu_signals = self.niuniu.generate_signals(data)
            cpgw_signals = self.cpgw.generate_signals(data)
            custom_cpgw_signals = self.custom_cpgw.generate_signals(data)
            
            # 组合信号
            combined_signals = (
                niuniu_signals['signal'] * self.weights['niuniu'] +
                cpgw_signals['signal'] * self.weights['cpgw'] +
                custom_cpgw_signals['signal'] * self.weights['custom_cpgw']
            )
            
            data['signal'] = combined_signals
            return data
            
        except Exception as e:
            logger.error(f"生成信号时出错: {str(e)}")
            raise

    def optimize_parameters(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        优化策略参数
        """
        try:
            # 如果已经有该股票的最优参数，直接返回
            if symbol in self.optimal_params:
                return self.optimal_params[symbol]
            
            # 准备训练数据
            train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
            
            best_params = None
            best_sharpe = float('-inf')
            
            # 网格搜索最优参数
            for niuniu_weight in np.linspace(self.param_ranges['niuniu_weight'][0], 
                                           self.param_ranges['niuniu_weight'][1], 5):
                for cpgw_weight in np.linspace(self.param_ranges['cpgw_weight'][0], 
                                             self.param_ranges['cpgw_weight'][1], 5):
                    for custom_cpgw_weight in np.linspace(self.param_ranges['custom_cpgw_weight'][0], 
                                                        self.param_ranges['custom_cpgw_weight'][1], 5):
                        # 确保权重和为1
                        if abs(niuniu_weight + cpgw_weight + custom_cpgw_weight - 1.0) > 0.01:
                            continue
                        
                        # 更新权重
                        self.weights = {
                            'niuniu': niuniu_weight,
                            'cpgw': cpgw_weight,
                            'custom_cpgw': custom_cpgw_weight
                        }
                        
                        # 在训练集上生成信号
                        train_signals = self.generate_signals(train_data.copy())
                        
                        # 计算夏普比率
                        returns = train_signals['signal'] * train_data['returns']
                        sharpe = calculate_sharpe_ratio(returns)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                'niuniu_weight': niuniu_weight,
                                'cpgw_weight': cpgw_weight,
                                'custom_cpgw_weight': custom_cpgw_weight
                            }
            
            # 保存最优参数
            self.optimal_params[symbol] = best_params
            
            # 使用最优参数更新权重
            self.weights = {
                'niuniu': best_params['niuniu_weight'],
                'cpgw': best_params['cpgw_weight'],
                'custom_cpgw': best_params['custom_cpgw_weight']
            }
            
            logger.info(f"股票 {symbol} 的最优参数: {best_params}")
            return best_params
            
        except Exception as e:
            logger.error(f"优化参数时出错: {str(e)}")
            raise

    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取信号组件
        """
        try:
            return {
                'niuniu_signal': self.niuniu.generate_signals(data)['signal'],
                'cpgw_signal': self.cpgw.generate_signals(data)['signal'],
                'custom_cpgw_signal': self.custom_cpgw.generate_signals(data)['signal']
            }
        except Exception as e:
            logger.error(f"提取信号组件时出错: {str(e)}")
            raise

    def get_signal_metadata(self) -> Dict[str, Any]:
        """
        获取信号元数据
        """
        try:
            return {
                'weights': self.weights,
                'optimal_params': self.optimal_params
            }
        except Exception as e:
            logger.error(f"获取信号元数据时出错: {str(e)}")
            raise 