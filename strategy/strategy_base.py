import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import logging
import enum
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MarketRegime(enum.Enum):
    """市场环境类型枚举"""
    UNKNOWN = "unknown"        # 未知/无法判断
    BULLISH = "bullish"        # 牛市趋势
    BEARISH = "bearish"        # 熊市趋势
    RANGING = "ranging"        # 震荡市
    VOLATILE = "volatile"      # 高波动
    LOW_VOLATILITY = "low_volatility"  # 低波动


class Strategy(ABC):
    """
    策略基类，所有交易策略都应继承自该类
    
    该基类定义了策略接口和通用功能：
    1. 计算技术指标
    2. 生成交易信号
    3. 提取信号组件
    4. 获取信号元数据
    5. 市场环境分析与调整
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            name: 策略名称
            parameters: 策略参数
        """
        self.name = name
        self.parameters = parameters
        self.logger = logging.getLogger(f"strategy.{name}")
        
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        pass
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        pass
        
    def get_stop_loss(self, price: float, position: int, market_regime: str) -> float:
        """
        计算止损价格
        
        Args:
            price: 当前价格
            position: 持仓方向 (1: 多头, -1: 空头)
            market_regime: 市场环境
            
        Returns:
            float: 止损价格
        """
        if market_regime == 'bullish':
            stop_loss_pct = 0.02  # 牛市止损更宽松
        elif market_regime == 'bearish':
            stop_loss_pct = 0.01  # 熊市止损更严格
        else:
            stop_loss_pct = 0.015  # 中性市场
            
        if position == 1:  # 多头
            return price * (1 - stop_loss_pct)
        else:  # 空头
            return price * (1 + stop_loss_pct)
            
    def get_take_profit(self, price: float, position: int, market_regime: str) -> float:
        """
        计算止盈价格
        
        Args:
            price: 当前价格
            position: 持仓方向 (1: 多头, -1: 空头)
            market_regime: 市场环境
            
        Returns:
            float: 止盈价格
        """
        if market_regime == 'bullish':
            take_profit_pct = 0.04  # 牛市止盈更高
        elif market_regime == 'bearish':
            take_profit_pct = 0.02  # 熊市止盈更低
        else:
            take_profit_pct = 0.03  # 中性市场
            
        if position == 1:  # 多头
            return price * (1 + take_profit_pct)
        else:  # 空头
            return price * (1 - take_profit_pct)
            
    def analyze_market_regime(self, df: pd.DataFrame) -> str:
        """
        分析市场环境
        
        Args:
            df: 包含技术指标的数据框
            
        Returns:
            str: 市场环境 ('bullish', 'bearish', 'neutral')
        """
        # 计算趋势指标
        ma_20 = df['close'].rolling(window=20).mean()
        ma_50 = df['close'].rolling(window=50).mean()
        
        # 计算波动率
        volatility = df['close'].pct_change().rolling(window=20).std()
        
        # 判断市场环境
        if ma_20.iloc[-1] > ma_50.iloc[-1] and volatility.iloc[-1] < volatility.quantile(0.7):
            return 'bullish'
        elif ma_20.iloc[-1] < ma_50.iloc[-1] and volatility.iloc[-1] > volatility.quantile(0.3):
            return 'bearish'
        else:
            return 'neutral'
            
    def optimize_parameters(self, 
                          df: pd.DataFrame,
                          param_grid: Dict[str, List[Any]],
                          metric: str = 'sharpe_ratio',
                          n_iter: int = 10) -> Dict[str, Any]:
        """
        优化策略参数
        
        Args:
            df: 历史数据
            param_grid: 参数网格
            metric: 优化指标
            n_iter: 迭代次数
            
        Returns:
            Dict[str, Any]: 优化后的参数
        """
        best_params = {}
        best_score = float('-inf')
        
        for _ in range(n_iter):
            # 随机选择参数组合
            params = {k: np.random.choice(v) for k, v in param_grid.items()}
            
            # 更新策略参数
            self.update_parameters(params)
            
            # 计算策略表现
            df = self.calculate_indicators(df)
            df = self.generate_signals(df)
            
            # 计算收益率
            df['position'] = df['signal'].shift(1).fillna(0)  # 使用前一天的信号作为今天的持仓
            df['returns'] = df['position'] * df['close'].pct_change()
            
            # 计算优化指标
            if metric == 'sharpe_ratio':
                returns = df['returns'].mean()
                volatility = df['returns'].std()
                if volatility > 0:
                    score = returns / volatility * np.sqrt(252)  # 年化夏普比率
                else:
                    score = float('-inf')
            elif metric == 'max_drawdown':
                cum_returns = (1 + df['returns']).cumprod()
                rolling_max = cum_returns.expanding().max()
                drawdown = (cum_returns - rolling_max) / rolling_max
                score = -drawdown.min()  # 负号因为我们要最小化最大回撤
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            # 更新最佳参数
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        self.logger.info(f"Best parameters found: {best_params} with score: {best_score}")
        return best_params
        
    def update_parameters(self, params: Dict[str, Any]):
        """
        更新策略参数
        
        Args:
            params: 新的参数值
        """
        self.parameters.update(params)
        self.logger.info(f"Updated parameters: {params}")
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略信息
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.__doc__
        }

    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        判断当前市场环境
        
        参数:
            data: 市场数据
            
        返回:
            市场环境枚举值
        """
        try:
            # 确保数据量足够进行分析
            if len(data) < 50:
                return MarketRegime.UNKNOWN
            
            # 复制一份数据以进行分析
            df = data.copy()
            
            # 检查数据列
            if 'close' not in df.columns:
                return MarketRegime.UNKNOWN
            
            # 计算基本指标
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            
            # 计算短期波动率（20日）
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # 年化
            
            # 获取最新数据
            latest = df.iloc[-1]
            
            # 1. 判断趋势方向
            trend_direction = 0
            if latest['ma20'] > latest['ma50'] * 1.01:  # 短期均线在长期均线上方1%以上
                trend_direction = 1  # 上升趋势
            elif latest['ma20'] < latest['ma50'] * 0.99:  # 短期均线在长期均线下方1%以上
                trend_direction = -1  # 下降趋势
            
            # 2. 判断波动性
            avg_volatility = df['volatility'].mean()
            high_volatility = latest['volatility'] > avg_volatility * 1.5
            low_volatility = latest['volatility'] < avg_volatility * 0.5
            
            # 3. 综合判断市场环境
            if trend_direction == 1:
                if high_volatility:
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.BULLISH
            elif trend_direction == -1:
                if high_volatility:
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.BEARISH
            else:  # 无明显趋势
                if high_volatility:
                    return MarketRegime.VOLATILE
                elif low_volatility:
                    return MarketRegime.LOW_VOLATILITY
                else:
                    return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"市场环境判断失败: {e}")
            return MarketRegime.UNKNOWN

    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据市场环境调整信号
        
        参数:
            data: 原始数据
            signals: 信号数据
            
        返回:
            调整后的信号
        """
        # 获取当前市场环境
        regime = self.get_market_regime(data)
        
        # 根据不同环境调整信号
        adjusted_signals = signals.copy()
        
        # 1. 在高波动环境下，减少信号频率
        if regime == MarketRegime.VOLATILE:
            # 使用更严格的过滤条件
            self.logger.info(f"高波动环境，减少信号频率")
            # 检查信号强度，只保留强信号
            for i in range(1, len(adjusted_signals)):
                if abs(adjusted_signals['signal'].iloc[i]) < 0.8:  # 信号强度阈值
                    adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        # 2. 在低波动环境下，增加信号频率
        elif regime == MarketRegime.LOW_VOLATILITY:
            self.logger.info(f"低波动环境，增加信号敏感度")
            # 此处可以增强弱信号
            
        # 3. 在熊市环境下，更谨慎处理做多信号
        elif regime == MarketRegime.BEARISH:
            self.logger.info(f"熊市环境，减少多头信号")
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 降低多头信号强度或忽略弱信号
                    if adjusted_signals['signal'].iloc[i] < 0.7:  # 较弱的多头信号
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        # 4. 在牛市环境下，更谨慎处理做空信号
        elif regime == MarketRegime.BULLISH:
            self.logger.info(f"牛市环境，减少空头信号")
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] < 0:
                    # 降低空头信号强度或忽略弱信号
                    if abs(adjusted_signals['signal'].iloc[i]) < 0.7:  # 较弱的空头信号
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        self.logger.info(f"市场环境: {regime.value}，信号调整完成")
        return adjusted_signals

    def get_position_size(self, data: pd.DataFrame, signal: float) -> float:
        """
        计算仓位大小
        
        参数:
            data: 市场数据
            signal: 信号值(-1.0至1.0)
            
        返回:
            仓位大小(0.0-1.0)
        """
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 基础仓位 = 信号强度
        base_position = abs(signal)
        
        # 根据市场环境调整仓位
        if regime == MarketRegime.VOLATILE:
            # 高波动环境，降低仓位
            return base_position * 0.5
        elif regime == MarketRegime.BEARISH and signal > 0:
            # 熊市环境下的多头信号，降低仓位
            return base_position * 0.7
        elif regime == MarketRegime.BULLISH and signal < 0:
            # 牛市环境下的空头信号，降低仓位
            return base_position * 0.7
            
        # 默认返回基础仓位
        return base_position

    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float,
                                stop_loss: float, position: int) -> float:
        """
        是否应该调整止损价格(追踪止损)
        
        参数:
            data: 市场数据
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            新的止损价格，如果不需要调整则返回原止损价格
        """
        # 默认实现 - 不调整止损
        return stop_loss

    def analyze(self, data: Union[pd.DataFrame, pd.Series], market_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析数据并生成交易信号
        
        参数:
            data: 历史数据，可以是DataFrame或Series
            market_state: 市场状态信息
            
        返回:
            分析结果字典
        """
        try:
            # 如果输入是Series，转换为DataFrame
            if isinstance(data, pd.Series):
                df = pd.DataFrame(data).T
            else:
                df = data.copy()
                
            # 计算技术指标
            df_with_indicators = self.calculate_indicators(df)
            
            # 生成交易信号
            df_with_signals = self.generate_signals(df_with_indicators)
            
            # 提取信号组件
            signal_components = self.extract_signal_components(df_with_signals)
            
            # 获取市场环境
            market_regime = self.get_market_regime(df_with_signals)
            
            # 计算综合得分
            score = self._calculate_score(signal_components)
            
            # 生成建议
            recommendations = self._generate_recommendations(
                df_with_signals,
                signal_components,
                market_regime,
                score
            )
            
            return {
                'score': score,
                'signals': df_with_signals['signal'].iloc[-1],
                'market_regime': market_regime.value,
                'signal_components': signal_components,
                'recommendations': recommendations,
                'risk_level': self._determine_risk_level(score, market_regime)
            }
            
        except Exception as e:
            self.logger.error(f"分析数据时出错: {str(e)}")
            return {
                'score': 0,
                'signals': 0,
                'market_regime': MarketRegime.UNKNOWN.value,
                'signal_components': {},
                'recommendations': [f"分析失败: {str(e)}"],
                'risk_level': 'high'
            }

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """设置策略参数"""
        self.parameters.update(parameters)
        
    def get_name(self) -> str:
        """获取策略名称"""
        return self.name

    def get_required_columns(self) -> List[str]:
        """获取策略所需的数据列"""
        return ['open', 'high', 'low', 'close', 'volume']

    def _calculate_score(self, signal_components: Dict[str, pd.Series]) -> float:
        """
        计算综合得分
        
        参数:
            signal_components: 信号组件字典
            
        返回:
            0-100之间的得分
        """
        try:
            # 简单的得分计算方法，可以根据需要调整
            weights = {
                'trend': 0.4,
                'momentum': 0.3,
                'volume': 0.2,
                'volatility': 0.1
            }
            
            score = 0
            for component, weight in weights.items():
                if component in signal_components:
                    value = signal_components[component].iloc[-1]
                    score += value * weight
                    
            # 将得分标准化到0-100之间
            score = max(0, min(100, (score + 1) * 50))
            return score
            
        except Exception as e:
            self.logger.error(f"计算得分时出错: {str(e)}")
            return 0
            
    def _generate_recommendations(self, data: pd.DataFrame, signal_components: Dict[str, pd.Series],
                                market_regime: MarketRegime, score: float) -> List[str]:
        """
        生成交易建议
        
        参数:
            data: 带有技术指标的DataFrame
            signal_components: 信号组件字典
            market_regime: 市场环境
            score: 综合得分
            
        返回:
            建议列表
        """
        try:
            recommendations = []
            
            # 根据得分生成建议
            if score > 70:
                recommendations.append("强烈建议买入")
            elif score > 50:
                recommendations.append("建议买入")
            elif score > 30:
                recommendations.append("建议观望")
            else:
                recommendations.append("建议卖出")
                
            # 根据市场环境添加建议
            if market_regime == MarketRegime.BULLISH:
                recommendations.append("市场处于上升趋势，可以考虑增加仓位")
            elif market_regime == MarketRegime.BEARISH:
                recommendations.append("市场处于下降趋势，建议减少仓位")
            elif market_regime == MarketRegime.RANGING:
                recommendations.append("市场处于震荡状态，建议谨慎操作")
            elif market_regime == MarketRegime.VOLATILE:
                recommendations.append("市场波动较大，建议控制风险")
                
            # 根据信号组件添加具体建议
            for component, series in signal_components.items():
                value = series.iloc[-1]
                if component == 'trend' and abs(value) > 0.8:
                    recommendations.append(f"趋势信号{'强烈' if abs(value) > 0.9 else ''}{'看多' if value > 0 else '看空'}")
                elif component == 'momentum' and abs(value) > 0.8:
                    recommendations.append(f"动量指标显示{'超买' if value > 0 else '超卖'}")
                elif component == 'volume' and abs(value) > 1.5:
                    recommendations.append(f"成交量{'显著放大' if value > 0 else '显著萎缩'}")
                    
            return recommendations
            
        except Exception as e:
            self.logger.error(f"生成建议时出错: {str(e)}")
            return ["生成建议失败"]
            
    def _determine_risk_level(self, score: float, market_regime: MarketRegime) -> str:
        """
        确定风险等级
        
        参数:
            score: 综合得分
            market_regime: 市场环境
            
        返回:
            风险等级: 'low', 'normal', 'warning', 'high'
        """
        try:
            # 基础风险等级
            if score < 30:
                risk_level = 'high'
            elif score < 50:
                risk_level = 'warning'
            elif score < 70:
                risk_level = 'normal'
            else:
                risk_level = 'low'
                
            # 根据市场环境调整风险等级
            if market_regime in [MarketRegime.VOLATILE, MarketRegime.BEARISH]:
                # 提高一级风险
                risk_levels = ['low', 'normal', 'warning', 'high']
                current_index = risk_levels.index(risk_level)
                risk_level = risk_levels[min(current_index + 1, len(risk_levels) - 1)]
                
            return risk_level
            
        except Exception as e:
            self.logger.error(f"确定风险等级时出错: {str(e)}")
            return 'high'

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        参数:
            data: 待验证的DataFrame
            
        返回:
            bool: 数据是否有效
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # 检查必需列是否存在
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"数据缺少必需列: {required_columns}")
                return False
                
            # 检查数据是否为空
            if data.empty:
                self.logger.error("数据为空")
                return False
                
            # 检查是否有缺失值
            if data[required_columns].isnull().any().any():
                self.logger.warning("数据存在缺失值")
                
            return True
            
        except Exception as e:
            self.logger.error(f"验证数据时出错: {str(e)}")
            return False
            
    def _calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算收益率
        
        参数:
            prices: 价格序列
            
        返回:
            Series: 收益率序列
        """
        try:
            return prices.pct_change()
        except Exception as e:
            self.logger.error(f"计算收益率时出错: {str(e)}")
            return pd.Series()
            
    def _calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        计算波动率
        
        参数:
            returns: 收益率序列
            window: 计算窗口
            
        返回:
            Series: 波动率序列
        """
        try:
            return returns.rolling(window=window).std() * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"计算波动率时出错: {str(e)}")
            return pd.Series()
            
    def _calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """
        计算回撤
        
        参数:
            prices: 价格序列
            
        返回:
            Series: 回撤序列
        """
        try:
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            return drawdown
        except Exception as e:
            self.logger.error(f"计算回撤时出错: {str(e)}")
            return pd.Series()
            
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        参数:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        返回:
            float: 夏普比率
        """
        try:
            excess_returns = returns - risk_free_rate/252
            if len(excess_returns) < 2:
                return 0.0
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        except Exception as e:
            self.logger.error(f"计算夏普比率时出错: {str(e)}")
            return 0.0

class MonitorStrategy(Strategy):
    """监控策略基类"""
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(name, parameters)
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.indicator_columns = []  # 子类定义需要的技术指标

    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否包含所需的列"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"数据中缺少以下列: {missing_columns}")
            return False
        return True

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备策略所需的数据"""
        if not self.validate_data(df):
            raise ValueError("数据缺少必需的列")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        try:
            if df is None or df.empty:
                self.logger.warning("输入数据为空")
                return pd.DataFrame()

            # 标准化列名
            df.columns = df.columns.str.lower()
            
            # 准备数据
            df = self.prepare_data(df)
            
            # 计算指标
            df = self.calculate_indicators(df)
            
            # 生成信号
            return self._generate_signals_impl(df)
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return pd.DataFrame()

    def _generate_signals_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """具体的信号生成逻辑，由子类实现"""
        raise NotImplementedError
