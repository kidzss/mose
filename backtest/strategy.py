import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Union, Any

class CombinedStrategy:
    """组合策略类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化组合策略
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str = None) -> dict:
        """加载配置"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return {}
            
    def analyze_market_environment(self, data: pd.DataFrame) -> dict:
        """
        分析市场环境
        
        参数:
            data: 市场数据
            
        返回:
            市场环境字典
        """
        try:
            if data.empty:
                return {
                    'trend': 'unknown',
                    'volatility': 'unknown',
                    'risk_level': 'unknown'
                }
                
            # 分析趋势
            trend = self._analyze_trend(data)
            
            # 分析波动性
            volatility = self._analyze_volatility(data)
            
            # 分析风险水平
            risk_level = self._analyze_risk_level(data)
            
            return {
                'trend': trend,
                'volatility': volatility,
                'risk_level': risk_level
            }
            
        except Exception as e:
            self.logger.error(f"分析市场环境时发生错误: {str(e)}")
            return {
                'trend': 'unknown',
                'volatility': 'unknown',
                'risk_level': 'unknown'
            }
            
    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """分析趋势"""
        try:
            if 'close' not in data.columns:
                return 'unknown'
                
            # 计算移动平均线
            ma20 = data['close'].rolling(window=20).mean()
            ma50 = data['close'].rolling(window=50).mean()
            
            if len(ma20) < 20 or len(ma50) < 50:
                return 'unknown'
                
            current_price = data['close'].iloc[-1]
            
            if current_price > ma20.iloc[-1] > ma50.iloc[-1]:
                return 'uptrend'
            elif current_price < ma20.iloc[-1] < ma50.iloc[-1]:
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"分析趋势时发生错误: {str(e)}")
            return 'unknown'
            
    def _analyze_volatility(self, data: pd.DataFrame) -> str:
        """分析波动性"""
        try:
            if 'close' not in data.columns:
                return 'unknown'
                
            # 计算波动率
            returns = data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            if volatility > 0.3:
                return 'high'
            elif volatility < 0.15:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            self.logger.error(f"分析波动性时发生错误: {str(e)}")
            return 'unknown'
            
    def _analyze_risk_level(self, data: pd.DataFrame) -> str:
        """分析风险水平"""
        try:
            if 'close' not in data.columns:
                return 'unknown'
                
            # 计算风险指标
            returns = data['close'].pct_change().dropna()
            
            # 计算夏普比率
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # 计算最大回撤
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            # 综合评估
            if max_drawdown > 0.2 or sharpe < 0:
                return 'high'
            elif max_drawdown < 0.1 and sharpe > 1:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            self.logger.error(f"分析风险水平时发生错误: {str(e)}")
            return 'unknown'
            
    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        生成交易信号
        
        参数:
            data: 市场数据
            
        返回:
            交易信号字典
        """
        try:
            if data.empty:
                return {}
                
            signals = {}
            for column in data.columns:
                if isinstance(data[column], pd.Series):
                    signal = self._generate_signal_for_symbol(data[column])
                    if signal:
                        signals[column] = signal
                        
            return signals
            
        except Exception as e:
            self.logger.error(f"生成交易信号时发生错误: {str(e)}")
            return {}
            
    def _generate_signal_for_symbol(self, data: pd.Series) -> Optional[float]:
        """为单个股票生成信号"""
        try:
            if data.empty:
                return None
                
            # 简单的移动平均策略
            ma_short = data.rolling(window=20).mean()
            ma_long = data.rolling(window=50).mean()
            
            if len(ma_short) < 20 or len(ma_long) < 50:
                return None
                
            # 生成信号
            if ma_short.iloc[-1] > ma_long.iloc[-1]:
                return 1.0  # 买入信号
            elif ma_short.iloc[-1] < ma_long.iloc[-1]:
                return -1.0  # 卖出信号
            else:
                return 0.0  # 持仓不变
                
        except Exception as e:
            self.logger.error(f"为单个股票生成信号时发生错误: {str(e)}")
            return None 