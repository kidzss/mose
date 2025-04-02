import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import logging
import enum

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

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化策略
        
        参数:
            name: 策略名称
            parameters: 策略参数字典
        """
        self.name = name
        self.parameters = parameters or {}
        self.version = '1.0.0'
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"strategy.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            data: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame，其中:
            1 = 买入信号
            0 = 持有/无信号
            -1 = 卖出信号
        """
        pass
    
    @abstractmethod
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        pass
    
    @abstractmethod
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        pass
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        更新策略参数
        
        参数:
            new_parameters: 新的参数字典
        """
        self.parameters.update(new_parameters)
        self.logger.info(f"更新策略参数: {new_parameters}")
    
    def optimize_parameters(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]], 
                           metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            data: 历史价格数据
            param_grid: 参数网格，包含要测试的参数值列表
            metric: 优化目标指标，默认为夏普比率
            
        返回:
            优化后的参数字典
        """
        self.logger.info(f"开始参数优化，优化指标: {metric}")
        
        # 导入必要的库
        from itertools import product
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        best_score = -float('inf')
        best_params = None
        
        # 遍历所有参数组合
        for combo in combinations:
            # 创建参数字典
            params = {name: value for name, value in zip(param_names, combo)}
            
            # 更新策略参数
            self.update_parameters(params)
            
            # 生成信号
            result = self.generate_signals(data)
            
            # 计算绩效（简化版，实际应实现完整的回测）
            if metric == 'sharpe_ratio':
                # 计算每日收益率（假设简单的买入持有策略）
                result['returns'] = result['close'].pct_change() * result['signal'].shift(1)
                
                # 计算夏普比率
                annual_factor = 252  # 假设交易日为252天
                mean_return = result['returns'].mean()
                std_return = result['returns'].std()
                if std_return > 0:
                    score = np.sqrt(annual_factor) * mean_return / std_return
                else:
                    score = 0
            else:
                # 如果实现了其他指标，可以在此处添加
                score = 0
                
            # 更新最佳参数
            if score > best_score:
                best_score = score
                best_params = params
                
            self.logger.debug(f"参数: {params}, 得分: {score}")
        
        # 设置为最佳参数
        if best_params:
            self.update_parameters(best_params)
            self.logger.info(f"优化完成，最佳参数: {best_params}, 得分: {best_score}")
        else:
            self.logger.warning("优化未能找到更好的参数")
            
        return best_params
    
    def __str__(self) -> str:
        """返回策略的字符串表示"""
        return f"{self.name} (v{self.version}) - 参数: {self.parameters}"

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

    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格
        
        参数:
            data: 市场数据
            entry_price: 入场价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            止损价格
        """
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 基础止损比例
        base_stop_pct = 0.05  # 5%止损
        
        # 根据市场环境调整止损比例
        if regime == MarketRegime.VOLATILE:
            # 高波动环境，扩大止损范围
            stop_pct = base_stop_pct * 1.5
        elif regime == MarketRegime.LOW_VOLATILITY:
            # 低波动环境，缩小止损范围
            stop_pct = base_stop_pct * 0.8
        else:
            stop_pct = base_stop_pct
        
        # 计算止损价格
        if position > 0:  # 多头
            return entry_price * (1 - stop_pct)
        elif position < 0:  # 空头
            return entry_price * (1 + stop_pct)
        return 0.0

    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止盈价格
        
        参数:
            data: 市场数据
            entry_price: 入场价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            止盈价格
        """
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 基础止盈比例
        base_tp_pct = 0.10  # 10%止盈
        
        # 根据市场环境调整止盈比例
        if regime == MarketRegime.BULLISH and position > 0:
            # 牛市多头，扩大止盈目标
            tp_pct = base_tp_pct * 1.5
        elif regime == MarketRegime.BEARISH and position < 0:
            # 熊市空头，扩大止盈目标
            tp_pct = base_tp_pct * 1.5
        else:
            tp_pct = base_tp_pct
        
        # 计算止盈价格
        if position > 0:  # 多头
            return entry_price * (1 + tp_pct)
        elif position < 0:  # 空头
            return entry_price * (1 - tp_pct)
        return 0.0

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

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        返回:
            包含策略信息的字典
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'description': self.__doc__ or "无描述",
            'version': getattr(self, 'version', '1.0.0')
        }

    def analyze(self, data: Dict[str, pd.DataFrame], market_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析市场数据并生成交易信号
        
        参数:
            data: 股票数据字典，键为股票代码，值为对应的DataFrame
            market_state: 市场状态字典，包含 trend, volatility, risk_level 等信息
            
        返回:
            包含分析结果的字典
        """
        try:
            market_state = market_state or {}
            results = {}
            for symbol, df in data.items():
                # 计算技术指标
                df_with_indicators = self.calculate_indicators(df)
                if df_with_indicators.empty:
                    continue
                    
                # 生成信号
                df_with_signals = self.generate_signals(df_with_indicators)
                if df_with_signals.empty:
                    continue
                
                # 获取市场环境并调整信号
                market_regime = self.get_market_regime(df_with_signals)
                df_with_signals = self.adjust_for_market_regime(df_with_signals, df_with_signals)
                    
                # 获取最新信号
                latest_signal = df_with_signals['signal'].iloc[-1]
                
                # 如果有信号，添加到结果中
                if latest_signal != 0:
                    # 计算推荐仓位大小
                    position_size = self.get_position_size(df_with_signals, latest_signal)
                    
                    # 计算止损止盈
                    entry_price = df_with_signals['close'].iloc[-1]
                    stop_loss = self.get_stop_loss(df_with_signals, entry_price, 
                                                  1 if latest_signal > 0 else -1)
                    take_profit = self.get_take_profit(df_with_signals, entry_price, 
                                                      1 if latest_signal > 0 else -1)
                    
                    # 提取信号组件
                    signal_components = self.extract_signal_components(df_with_signals)
                    
                    results[symbol] = {
                        'signal': latest_signal,
                        'position_size': position_size,
                        'price': entry_price,
                        'timestamp': df_with_signals.index[-1],
                        'market_regime': market_regime.value,
                        'market_state': market_state,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal_components': {k: v.normalized.iloc[-1] if hasattr(v, 'normalized') else v.iloc[-1] for k, v in signal_components.items()}
                    }
                    
            return results
        except Exception as e:
            self.logger.error(f"分析数据时出错: {e}", exc_info=True)
            return {}

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
