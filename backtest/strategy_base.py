from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    交易策略基类
    
    定义了策略需要实现的基本接口，包括：
    1. 计算技术指标
    2. 生成交易信号
    3. 优化策略参数
    4. 评估策略性能
    """
    
    def __init__(self, **kwargs):
        """
        初始化策略
        
        参数:
            **kwargs: 策略参数
        """
        self.params = kwargs
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            data: 历史数据，包含OHLCV等基本数据
            
        返回:
            添加了技术指标的DataFrame
        """
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 带有技术指标的历史数据
            
        返回:
            包含交易信号的DataFrame
        """
        pass
        
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, Any],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            data: 历史数据
            param_grid: 参数网格，格式为 {参数名: 参数值列表}
            metric: 优化指标，默认为夏普比率
            
        返回:
            最优参数组合
        """
        try:
            best_score = float('-inf')
            best_params = None
            
            # 生成参数组合
            param_combinations = self._generate_param_combinations(param_grid)
            
            # 遍历参数组合
            for params in param_combinations:
                # 更新策略参数
                self.params.update(params)
                
                # 计算指标和信号
                indicators = self.calculate_indicators(data)
                signals = self.generate_signals(indicators)
                
                # 计算绩效
                performance = self.evaluate_performance(data, signals)
                score = performance.get(metric, float('-inf'))
                
                # 更新最优参数
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            return best_params
            
        except Exception as e:
            logger.error(f"优化参数时出错: {e}")
            return {}
            
    def evaluate_performance(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        评估策略性能
        
        参数:
            data: 历史数据
            signals: 交易信号
            
        返回:
            包含各项性能指标的字典
        """
        try:
            # 计算每日收益率
            daily_returns = self._calculate_daily_returns(data, signals)
            
            # 计算累积收益率
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            
            # 计算年化收益率
            annual_return = self._calculate_annual_return(daily_returns)
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            return {
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': cumulative_returns.iloc[-1],
                'volatility': daily_returns.std() * np.sqrt(252),
                'win_rate': len(daily_returns[daily_returns > 0]) / len(daily_returns)
            }
            
        except Exception as e:
            logger.error(f"评估性能时出错: {e}")
            return {}
            
    def _calculate_daily_returns(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> pd.Series:
        """计算每日收益率"""
        try:
            # 获取价格变动
            price_changes = data['Close'].pct_change()
            
            # 根据信号计算收益率
            daily_returns = price_changes * signals['signal'].shift(1)
            
            return daily_returns.dropna()
            
        except Exception as e:
            logger.error(f"计算每日收益率时出错: {e}")
            return pd.Series()
            
    def _calculate_annual_return(self, daily_returns: pd.Series) -> float:
        """计算年化收益率"""
        try:
            total_return = (1 + daily_returns).prod()
            years = len(daily_returns) / 252
            return (total_return ** (1/years)) - 1
            
        except Exception as e:
            logger.error(f"计算年化收益率时出错: {e}")
            return 0.0
            
    def _calculate_sharpe_ratio(
        self,
        daily_returns: pd.Series,
        risk_free_rate: float = 0.03
    ) -> float:
        """计算夏普比率"""
        try:
            excess_returns = daily_returns - risk_free_rate/252
            if len(excess_returns) > 0:
                return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            return 0.0
            
        except Exception as e:
            logger.error(f"计算夏普比率时出错: {e}")
            return 0.0
            
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns - rolling_max
            return abs(drawdowns.min())
            
        except Exception as e:
            logger.error(f"计算最大回撤时出错: {e}")
            return 0.0
            
    def _generate_param_combinations(self, param_grid: Dict[str, Any]) -> list:
        """生成参数组合"""
        try:
            import itertools
            
            # 获取参数名和值列表
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # 生成笛卡尔积
            combinations = list(itertools.product(*param_values))
            
            # 转换为参数字典列表
            return [dict(zip(param_names, combo)) for combo in combinations]
            
        except Exception as e:
            logger.error(f"生成参数组合时出错: {e}")
            return []
            
    def get_required_data(self) -> Dict[str, Any]:
        """
        获取策略所需的数据配置
        
        返回:
            数据配置字典，包含所需的数据字段、时间范围等
        """
        return {
            'fields': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'min_history': 100,  # 最少需要的历史数据天数
            'frequency': '1d'    # 数据频率
        }
        
    def get_parameters(self) -> Dict[str, Any]:
        """获取当前策略参数"""
        return self.params.copy()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """设置策略参数"""
        self.params.update(params)
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据是否满足策略要求
        
        参数:
            data: 待验证的数据
            
        返回:
            数据是否有效
        """
        try:
            required_data = self.get_required_data()
            
            # 检查必需字段
            for field in required_data['fields']:
                if field not in data.columns:
                    logger.warning(f"缺少必需字段: {field}")
                    return False
                    
            # 检查数据量
            if len(data) < required_data['min_history']:
                logger.warning(f"数据量不足，需要至少 {required_data['min_history']} 条记录")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证数据时出错: {e}")
            return False 