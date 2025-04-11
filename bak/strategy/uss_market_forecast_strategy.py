import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from strategy.strategy_base import Strategy, MarketRegime

class MarketForecastStrategy(Strategy):
    """
    Market Forecast 策略
    
    该策略基于三条曲线的反转信号生成买卖信号：
    1. Momentum (短期)
    2. NearTerm (中期)
    3. Intermediate (长期)
    
    买入条件: 三条曲线几乎同时在底部区域反转上升
    卖出条件: 三条曲线几乎同时在顶部区域反转下降
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化Market Forecast策略
        
        参数:
            parameters: 策略参数字典，可包含以下键:
                - intermediate_len: 长期曲线窗口大小，默认20
                - near_term_len: 中期曲线窗口大小，默认10
                - momentum_len: 短期曲线窗口大小，默认5
                - bottom_zone: 底部区域阈值，默认30
                - top_zone: 顶部区域阈值，默认70
                - stop_loss_pct: 止损百分比，默认5.0
                - take_profit_pct: 止盈百分比，默认15.0
                - use_market_regime: 是否使用市场环境，默认True
        """
        default_params = {
            'intermediate_len': 20,
            'near_term_len': 10,
            'momentum_len': 5,
            'bottom_zone': 30,
            'top_zone': 70,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0,
            'max_position_size': 0.2,
            'use_market_regime': True
        }
        
        # 合并默认参数和传入的参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__("MarketForecastStrategy", default_params)
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算Market Forecast策略所需的技术指标
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了技术指标的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法计算指标")
            return data
            
        # 检查必要的列是否存在
        required_columns = ['low', 'high', 'close']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data
        
        try:
            # 获取参数
            intermediate_len = self.parameters['intermediate_len']
            near_term_len = self.parameters['near_term_len']
            momentum_len = self.parameters['momentum_len']
            bottom_zone = self.parameters['bottom_zone']
            top_zone = self.parameters['top_zone']
            
            # 计算Intermediate曲线 (长期)
            data['lowest_low_intermediate'] = data['low'].rolling(window=intermediate_len).min()
            data['highest_high_intermediate'] = data['high'].rolling(window=intermediate_len).max()
            data['Intermediate'] = (data['close'] - data['lowest_low_intermediate']) / (data['highest_high_intermediate'] - data['lowest_low_intermediate']) * 100
            
            # 计算NearTerm曲线 (中期)
            data['lowest_low_near_term'] = data['low'].rolling(window=near_term_len).min()
            data['highest_high_near_term'] = data['high'].rolling(window=near_term_len).max()
            data['NearTerm'] = (data['close'] - data['lowest_low_near_term']) / (data['highest_high_near_term'] - data['lowest_low_near_term']) * 100
            
            # 计算Momentum曲线 (短期)
            data['lowest_low_momentum'] = data['low'].rolling(window=momentum_len).min()
            data['highest_high_momentum'] = data['high'].rolling(window=momentum_len).max()
            data['Momentum'] = (data['close'] - data['lowest_low_momentum']) / (data['highest_high_momentum'] - data['lowest_low_momentum']) * 100
            
            # 计算反转信号
            # 检测底部反转（上升）
            data['Momentum_bottom_reversal'] = (data['Momentum'].shift(1) < data['Momentum']) & (data['Momentum'].shift(1) < bottom_zone)
            data['NearTerm_bottom_reversal'] = (data['NearTerm'].shift(1) < data['NearTerm']) & (data['NearTerm'].shift(1) < bottom_zone)
            data['Intermediate_bottom_reversal'] = (data['Intermediate'].shift(1) < data['Intermediate']) & (data['Intermediate'].shift(1) < bottom_zone)
            
            # 检测顶部反转（下降）
            data['Momentum_top_reversal'] = (data['Momentum'].shift(1) > data['Momentum']) & (data['Momentum'].shift(1) > top_zone)
            data['NearTerm_top_reversal'] = (data['NearTerm'].shift(1) > data['NearTerm']) & (data['NearTerm'].shift(1) > top_zone)
            data['Intermediate_top_reversal'] = (data['Intermediate'].shift(1) > data['Intermediate']) & (data['Intermediate'].shift(1) > top_zone)
            
            # 计算ATR
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['atr14'] = tr.rolling(14).mean()
            
            # 填充NaN值
            data = data.bfill().ffill()
            
            return data
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据Market Forecast策略生成交易信号
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            添加了信号列的DataFrame，其中:
            1: 买入信号
            0: 持有/无信号
            -1: 卖出信号
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法生成信号")
            return data
            
        # 检查是否已计算技术指标
        if 'Momentum' not in data.columns:
            data = self.calculate_indicators(data)
            
        # 检查必要的列是否存在
        required_columns = ['Momentum_bottom_reversal', 'NearTerm_bottom_reversal', 'Intermediate_bottom_reversal',
                           'Momentum_top_reversal', 'NearTerm_top_reversal', 'Intermediate_top_reversal']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data
        
        try:
            # 初始化信号列
            data['signal'] = 0
            
            # 买入信号：三条曲线几乎同时在底部区域反转上升
            buy_signal = (
                data['Momentum_bottom_reversal'] & 
                data['NearTerm_bottom_reversal'] & 
                data['Intermediate_bottom_reversal']
            )
            
            # 卖出信号：三条曲线几乎同时在顶部区域反转下跌
            sell_signal = (
                data['Momentum_top_reversal'] & 
                data['NearTerm_top_reversal'] & 
                data['Intermediate_top_reversal']
            )
            
            # 设置信号
            data.loc[buy_signal, 'signal'] = 1
            data.loc[sell_signal, 'signal'] = -1
            
            # 如果启用市场环境适配，调整信号
            if self.parameters['use_market_regime']:
                data = self.adjust_for_market_regime(data, data)
                
            # 修复潜在的类型推断警告
            data = data.infer_objects(copy=False)
                
            return data
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data
    
    def adjust_for_market_regime(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        根据市场环境调整信号
        
        参数:
            data: 原始数据
            signals: 信号数据
            
        返回:
            调整后的信号
        """
        # 首先调用基类的市场环境调整方法
        adjusted_signals = super().adjust_for_market_regime(data, signals)
        
        # 获取当前市场环境
        regime = self.get_market_regime(data)
        
        # 根据Market Forecast策略特性进行额外的调整
        if regime == MarketRegime.BEARISH:
            self.logger.info("Market Forecast策略: 熊市环境，增强卖出信号权重")
            # 在熊市中，更严格地筛选买入信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 在熊市中，要求至少两个指标超过40才考虑买入
                    momentum = adjusted_signals['Momentum'].iloc[i]
                    near_term = adjusted_signals['NearTerm'].iloc[i]
                    intermediate = adjusted_signals['Intermediate'].iloc[i]
                    
                    indicators_above_threshold = sum([
                        momentum > 40,
                        near_term > 40,
                        intermediate > 40
                    ])
                    
                    if indicators_above_threshold < 2:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                
        elif regime == MarketRegime.BULLISH:
            self.logger.info("Market Forecast策略: 牛市环境，增强买入信号权重")
            # 在牛市中，更严格地筛选卖出信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] < 0:
                    # 在牛市中，要求至少两个指标低于60才考虑卖出
                    momentum = adjusted_signals['Momentum'].iloc[i]
                    near_term = adjusted_signals['NearTerm'].iloc[i]
                    intermediate = adjusted_signals['Intermediate'].iloc[i]
                    
                    indicators_below_threshold = sum([
                        momentum < 60,
                        near_term < 60,
                        intermediate < 60
                    ])
                    
                    if indicators_below_threshold < 2:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        elif regime == MarketRegime.VOLATILE:
            self.logger.info("Market Forecast策略: 高波动环境，减少信号频率")
            # 高波动环境下，更严格地筛选信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] != 0:
                    # 在高波动环境中，要求三个指标更一致
                    momentum = adjusted_signals['Momentum'].iloc[i]
                    near_term = adjusted_signals['NearTerm'].iloc[i]
                    intermediate = adjusted_signals['Intermediate'].iloc[i]
                    
                    # 计算三个指标之间的标准差，判断一致性
                    values = [momentum, near_term, intermediate]
                    std_dev = np.std(values)
                    
                    # 在高波动环境中，要求三个指标更一致（标准差更小）
                    if std_dev > 15:  # 如果标准差大于15，表示指标不一致
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        return adjusted_signals
    
    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        根据Market Forecast指标判断当前市场环境
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            市场环境类型
        """
        if data is None or data.empty:
            return MarketRegime.UNKNOWN
            
        try:
            # 获取最近的数据点
            recent_data = data.iloc[-20:]
            
            # 检查必要的列是否存在
            if 'Momentum' not in recent_data.columns or 'NearTerm' not in recent_data.columns or 'Intermediate' not in recent_data.columns:
                return MarketRegime.UNKNOWN
            
            # 计算三条曲线的平均值
            avg_momentum = recent_data['Momentum'].mean()
            avg_near_term = recent_data['NearTerm'].mean()
            avg_intermediate = recent_data['Intermediate'].mean()
            
            # 计算收盘价的波动率
            volatility = recent_data['close'].pct_change().std() * 100 if 'close' in recent_data.columns else 1.5
            
            # 判断市场环境
            if avg_momentum > 70 and avg_near_term > 70 and avg_intermediate > 70:
                return MarketRegime.BULLISH
            elif avg_momentum < 30 and avg_near_term < 30 and avg_intermediate < 30:
                return MarketRegime.BEARISH
            elif volatility > 2.5:
                return MarketRegime.VOLATILE
            elif abs(avg_near_term - 50) < 10 and abs(avg_intermediate - 50) < 10:
                return MarketRegime.RANGING
            elif avg_momentum > 50 and avg_near_term > 50 and avg_intermediate > 50:
                return MarketRegime.BULLISH
            elif avg_momentum < 50 and avg_near_term < 50 and avg_intermediate < 50:
                return MarketRegime.BEARISH
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"判断市场环境时出错: {e}")
            return MarketRegime.UNKNOWN
    
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """
        根据Market Forecast指标计算仓位大小
        
        参数:
            data: 市场数据
            signal: 信号值（1=多头，-1=空头）
            
        返回:
            仓位大小(0.0-1.0)
        """
        # 首先调用基类的仓位计算
        base_position = super().get_position_size(data, signal)
        
        try:
            # 获取最近的三个指标值
            if data is None or data.empty or len(data) < 1:
                return base_position
                
            if 'Momentum' not in data.columns or 'NearTerm' not in data.columns or 'Intermediate' not in data.columns:
                return base_position
                
            momentum = data['Momentum'].iloc[-1]
            near_term = data['NearTerm'].iloc[-1]
            intermediate = data['Intermediate'].iloc[-1]
            
            # 根据指标强度调整仓位
            if signal > 0:  # 多头
                # 三个指标越低（接近底部），反转上升时仓位越大
                avg_value = (momentum + near_term + intermediate) / 3
                strength_factor = (50 - avg_value) / 50  # 离底部越近，系数越大
                strength_factor = max(0.5, min(1.5, strength_factor))  # 限制在0.5-1.5之间
                
            else:  # 空头
                # 三个指标越高（接近顶部），反转下降时仓位越大
                avg_value = (momentum + near_term + intermediate) / 3
                strength_factor = (avg_value - 50) / 50  # 离顶部越近，系数越大
                strength_factor = max(0.5, min(1.5, strength_factor))  # 限制在0.5-1.5之间
                
            # 根据强度因子调整仓位大小
            position_size = base_position * strength_factor
            
            # 限制最大仓位
            return min(position_size, self.parameters['max_position_size'])
            
        except Exception as e:
            self.logger.error(f"计算仓位大小时出错: {e}")
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
        try:
            # 使用ATR计算止损
            if 'atr14' in data.columns and len(data) > 0:
                atr = data['atr14'].iloc[-1]
                
                # 获取市场环境
                regime = self.get_market_regime(data)
                
                # 根据市场环境调整ATR倍数
                atr_multiplier = 2.0  # 默认值
                
                if regime == MarketRegime.VOLATILE:
                    # 高波动环境，增加止损宽度
                    atr_multiplier = 3.0
                elif regime == MarketRegime.LOW_VOLATILITY:
                    # 低波动环境，减少止损宽度
                    atr_multiplier = 1.5
                
                # 计算止损
                if position > 0:  # 多头
                    return entry_price - (atr * atr_multiplier)
                elif position < 0:  # 空头
                    return entry_price + (atr * atr_multiplier)
                
            # 如果没有ATR数据，使用百分比止损
            stop_loss_pct = self.parameters['stop_loss_pct'] / 100
            
            if position > 0:  # 多头
                return entry_price * (1 - stop_loss_pct)
            elif position < 0:  # 空头
                return entry_price * (1 + stop_loss_pct)
                
            return 0
            
        except Exception as e:
            self.logger.error(f"计算止损价格时出错: {e}")
            return super().get_stop_loss(data, entry_price, position)
    
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
        try:
            # 计算ATR
            if 'atr14' in data.columns and len(data) > 0:
                atr = data['atr14'].iloc[-1]
                
                # 获取市场环境
                regime = self.get_market_regime(data)
                
                # 根据市场环境调整ATR倍数
                tp_multiplier = 3.0  # 默认值
                
                if regime == MarketRegime.BULLISH and position > 0:
                    # 牛市中的多头，使用更大的止盈目标
                    tp_multiplier = 5.0
                elif regime == MarketRegime.BEARISH and position < 0:
                    # 熊市中的空头，使用更大的止盈目标
                    tp_multiplier = 5.0
                
                # 计算止盈
                if position > 0:  # 多头
                    return entry_price + (atr * tp_multiplier)
                elif position < 0:  # 空头
                    return entry_price - (atr * tp_multiplier)
            
            # 如果没有ATR数据，使用百分比止盈
            take_profit_pct = self.parameters['take_profit_pct'] / 100
            
            if position > 0:  # 多头
                return entry_price * (1 + take_profit_pct)
            elif position < 0:  # 空头
                return entry_price * (1 - take_profit_pct)
                
            return 0
            
        except Exception as e:
            self.logger.error(f"计算止盈价格时出错: {e}")
            return super().get_take_profit(data, entry_price, position)
    
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
            调整后的止损价格
        """
        try:
            # 获取市场环境
            regime = self.get_market_regime(data)
            
            # 计算ATR
            if 'atr14' in data.columns and len(data) > 0:
                atr = data['atr14'].iloc[-1]
            else:
                # 如果没有ATR数据，使用价格的一定比例
                atr = current_price * 0.02
            
            # 计算利润
            if position > 0:
                profit_pct = (current_price / entry_price - 1) * 100
            else:
                profit_pct = (1 - current_price / entry_price) * 100
                
            # 利润达到一定水平时开始追踪止损
            if profit_pct > 5:
                # 根据市场环境和持仓方向调整追踪参数
                if (regime == MarketRegime.BULLISH and position > 0) or \
                   (regime == MarketRegime.BEARISH and position < 0):
                    # 趋势明确时使用较宽松的追踪止损
                    trail_pct = 2.0  # 允许2%回撤
                else:
                    # 其他情况使用较紧的追踪止损
                    trail_pct = 1.5  # 允许1.5%回撤
                    
                # 计算新的止损价格
                if position > 0:
                    new_stop = current_price * (1 - trail_pct/100)
                    # 止损只上移不下移
                    if new_stop > stop_loss:
                        return new_stop
                else:
                    new_stop = current_price * (1 + trail_pct/100)
                    # 止损只下移不上移
                    if new_stop < stop_loss:
                        return new_stop
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"调整止损价格时出错: {e}")
            return stop_loss
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        result = self.calculate_indicators(data)
        
        # 提取关键组件
        components = {
            'momentum': result.get('Momentum', pd.Series()),
            'near_term': result.get('NearTerm', pd.Series()),
            'intermediate': result.get('Intermediate', pd.Series()),
            'price': result.get('close', pd.Series()),
            'atr': result.get('atr14', pd.Series())
        }
        
        return components
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号组件的元数据
        
        返回:
            字典，包含每个信号组件的元数据
        """
        return {
            'momentum': {
                'name': 'Momentum',
                'description': f"{self.parameters['momentum_len']}周期动量曲线",
                'color': 'blue',
                'line_style': 'solid',
                'importance': 'high'
            },
            'near_term': {
                'name': 'NearTerm',
                'description': f"{self.parameters['near_term_len']}周期中期曲线",
                'color': 'red',
                'line_style': 'solid',
                'importance': 'high'
            },
            'intermediate': {
                'name': 'Intermediate',
                'description': f"{self.parameters['intermediate_len']}周期长期曲线",
                'color': 'green',
                'line_style': 'solid',
                'importance': 'high'
            },
            'price': {
                'name': '价格',
                'description': '资产收盘价',
                'color': 'black',
                'line_style': 'solid',
                'importance': 'high'
            },
            'atr': {
                'name': 'ATR',
                'description': '14周期真实波动范围均值',
                'color': 'orange',
                'line_style': 'dashed',
                'importance': 'low'
            }
        }

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        返回:
            包含策略信息的字典
        """
        return {
            "name": self.name,
            "version": "1.0.0",
            "description": "Market Forecast策略，基于三条曲线（Momentum、NearTerm、Intermediate）的反转信号生成买卖信号",
            "parameters": self.parameters,
            "author": "System",
            "creation_date": "2023-01-01",
            "last_modified_date": "2023-12-31",
            "risk_level": "medium",
            "performance_metrics": {
                "sharpe_ratio": None,
                "max_drawdown": None,
                "win_rate": None
            },
            "suitable_market_regimes": ["bullish", "bearish"],
            "tags": ["technical", "reversal", "momentum"]
        } 