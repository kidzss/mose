import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .strategy_base import Strategy, MarketRegime

class GoldTriangleStrategy(Strategy):
    """
    黄金三角交易策略（优化版）
    
    策略说明:
    1. 买入条件: 
       - SMA_5 > SMA_10 (短期均线上穿中期均线)
       - SMA_10 > SMA_20 (中期均线上穿长期均线)
       - 前一天 SMA_10 < SMA_20 (确认是交叉点)
       - 如果risk_averse=True，则还需要价格 > SMA_100
    
    2. 卖出条件:
       - SMA_10 < SMA_20 (中期均线下穿长期均线)
       
    优化点:
    1. 增加信号过滤机制减少虚假信号
    2. 添加支撑位和阻力位判断
    3. 增加交叉质量评估
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化黄金三角策略
        
        参数:
            parameters: 策略参数字典，可包含:
                - short_period: 短期均线周期，默认5
                - mid_period: 中期均线周期，默认10
                - long_period: 长期均线周期，默认20
                - trend_period: 趋势均线周期，默认100
                - risk_averse: 是否规避风险，默认True
                - use_market_regime: 是否使用市场环境，默认True
                - volume_filter: 是否使用成交量过滤，默认True
                - min_cross_quality: 最小交叉质量阈值，默认0.5
                - support_resistance_period: 支撑阻力回溯期，默认20
                - false_signal_filter: 是否过滤虚假信号，默认True
                - min_holding_days: 最小持仓周期，默认3
        """
        default_params = {
            'short_period': 5,
            'mid_period': 10,
            'long_period': 20,
            'trend_period': 100,
            'risk_averse': True,
            'use_market_regime': True,
            'volume_filter': True,
            'min_cross_quality': 0.5,
            'support_resistance_period': 20,
            'false_signal_filter': True,
            'min_holding_days': 3
        }
        
        # 合并默认参数和用户参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__('GoldTriangleStrategy', default_params)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标
        
        参数:
            df: 原始OHLCV数据
            
        返回:
            添加了技术指标的DataFrame
        """
        try:
            if df is None or df.empty:
                self.logger.warning("数据为空，无法计算指标")
                return pd.DataFrame()
            
            # 复制数据以避免修改原始数据
            df = df.copy()
            
            # 计算移动平均线
            df['ma5'] = df['close'].rolling(window=self.parameters['short_period']).mean()
            df['ma10'] = df['close'].rolling(window=self.parameters['mid_period']).mean()
            df['ma20'] = df['close'].rolling(window=self.parameters['long_period']).mean()
            df['ma100'] = df['close'].rolling(window=self.parameters['trend_period']).mean()
            
            # 添加ATR指标用于止损计算
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr14'] = tr.rolling(14).mean()
            
            # 计算支撑位和阻力位
            sr_period = self.parameters['support_resistance_period']
            df['resistance'] = df['high'].rolling(sr_period).max()
            df['support'] = df['low'].rolling(sr_period).min()
            
            # 计算相对位置 (当前价格在支撑阻力间的位置，0表示支撑位，1表示阻力位)
            df['sr_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # 计算RSI指标
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算成交量相对变化
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
            
            # 计算交叉质量 (均线角度和距离)
            df['ma5_slope'] = df['ma5'].pct_change(5) * 100  # 短期均线斜率
            df['ma10_slope'] = df['ma10'].pct_change(5) * 100  # 中期均线斜率
            df['ma20_slope'] = df['ma20'].pct_change(5) * 100  # 长期均线斜率
            
            # 计算均线间距
            df['ma5_10_gap'] = (df['ma5'] / df['ma10'] - 1) * 100  # 短中期均线间距百分比
            df['ma10_20_gap'] = (df['ma10'] / df['ma20'] - 1) * 100  # 中长期均线间距百分比
            
            # 填充NaN值
            df = df.bfill().ffill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return df
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            df: 包含OHLCV数据的DataFrame
            
        返回:
            添加了'signal'列的DataFrame
        """
        try:
            if df is None or df.empty:
                self.logger.warning("数据为空，无法生成信号")
                return pd.DataFrame()
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 初始化信号列
            df['signal'] = 0
            
            # 用于记录当前持仓天数
            holding_days = 0
            last_signal = 0
            
            # 生成信号
            for i in range(1, len(df)):
                # 获取当前和前一天的数据
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # 更新持仓天数
                if last_signal != 0:
                    holding_days += 1
                else:
                    holding_days = 0
                
                # 基础买入条件
                buy_cond1 = current['ma5'] > current['ma10']  # 短期均线上穿中期均线
                buy_cond2 = current['ma10'] > current['ma20']  # 中期均线上穿长期均线
                buy_cond3 = previous['ma10'] < previous['ma20']  # 确认是交叉点
                buy_cond4 = current['close'] > current['ma100']  # 价格在长期趋势线上方
                
                # 基础卖出条件
                sell_cond = current['ma10'] < current['ma20']  # 中期均线下穿长期均线
                
                # 计算信号质量评分 (0-1之间)
                cross_quality = self._calculate_cross_quality(df, i)
                
                # 生成基础信号
                signal = 0
                
                if last_signal <= 0 and buy_cond1 and buy_cond2 and buy_cond3:
                    if not self.parameters['risk_averse'] or buy_cond4:
                        # 应用过滤器
                        if self._apply_buy_filters(df, i, cross_quality):
                            signal = 1
                elif last_signal >= 0 and sell_cond:
                    # 确保持仓时间已达到最小要求
                    if not self.parameters['false_signal_filter'] or holding_days >= self.parameters['min_holding_days']:
                        # 应用过滤器
                        if self._apply_sell_filters(df, i, cross_quality):
                            signal = -1
                
                # 如果没有新信号，保持前一个仓位
                if signal == 0:
                    signal = last_signal
                
                df.loc[df.index[i], 'signal'] = signal
                last_signal = signal
            
            # 如果启用市场环境适配，调整信号
            if self.parameters['use_market_regime']:
                df = self.adjust_for_market_regime(df, df)
                
            # 在返回前进行类型推断
            df = df.infer_objects(copy=False)
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return df
        
    def _calculate_cross_quality(self, df: pd.DataFrame, i: int) -> float:
        """
        计算均线交叉质量评分
        
        参数:
            df: 数据集
            i: 当前索引
            
        返回:
            质量评分 (0-1)
        """
        try:
            # 获取当前数据
            current = df.iloc[i]
            
            # 考虑三个因素：
            # 1. 均线角度 (斜率越陡，得分越高)
            # 2. 均线间距 (间距越大，得分越高)
            # 3. 交易量确认 (量比越大，得分越高)
            
            # 角度评分 (斜率)
            slope_score = 0
            if current['ma5_slope'] > 0 and current['ma10_slope'] > 0:
                # 正斜率代表上升趋势
                slope_score = min(1, (current['ma5_slope'] + current['ma10_slope']) / 4)
            
            # 间距评分
            gap_score = 0
            if current['ma5_10_gap'] > 0 and current['ma10_20_gap'] > 0:
                # 正间距代表短期均线在上方
                gap_score = min(1, (current['ma5_10_gap'] + current['ma10_20_gap']) / 3)
            
            # 成交量评分
            volume_score = min(1, current['volume_ratio'] / 2) if current['volume_ratio'] > 1 else 0
            
            # 综合评分 (加权平均)
            quality = 0.4 * slope_score + 0.4 * gap_score + 0.2 * volume_score
            
            return quality
        
        except Exception as e:
            self.logger.error(f"计算交叉质量时出错: {e}")
            return 0
    
    def _apply_buy_filters(self, df: pd.DataFrame, i: int, cross_quality: float) -> bool:
        """
        应用买入信号过滤器
        
        参数:
            df: 数据集
            i: 当前索引
            cross_quality: 交叉质量评分
            
        返回:
            是否通过过滤器
        """
        try:
            # 获取当前数据
            current = df.iloc[i]
            
            # 1. 交叉质量检查
            if cross_quality < self.parameters['min_cross_quality']:
                return False
            
            # 2. 成交量确认 (如果启用)
            if self.parameters['volume_filter'] and current['volume_ratio'] < 1.2:
                return False
            
            # 3. 价格位置检查 (不在阻力位附近)
            if current['sr_position'] > 0.9:  # 接近阻力位
                return False
            
            # 4. RSI超买检查
            if current['rsi'] > 70:  # 超买区域
                return False
            
            # 通过所有过滤器
            return True
        
        except Exception as e:
            self.logger.error(f"应用买入过滤器时出错: {e}")
            return True  # 出错时默认通过
    
    def _apply_sell_filters(self, df: pd.DataFrame, i: int, cross_quality: float) -> bool:
        """
        应用卖出信号过滤器
        
        参数:
            df: 数据集
            i: 当前索引
            cross_quality: 交叉质量评分
            
        返回:
            是否通过过滤器
        """
        try:
            # 获取当前数据
            current = df.iloc[i]
            
            # 1. 交叉质量检查 (对卖出要求不太严格)
            if cross_quality < self.parameters['min_cross_quality'] * 0.6:
                return False
            
            # 2. 价格位置检查 (不在支撑位附近)
            if current['sr_position'] < 0.1:  # 接近支撑位
                return False
            
            # 3. RSI超卖检查
            if current['rsi'] < 30:  # 超卖区域
                return False
            
            # 通过所有过滤器
            return True
        
        except Exception as e:
            self.logger.error(f"应用卖出过滤器时出错: {e}")
            return True  # 出错时默认通过
    
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
        
        # 根据黄金三角策略特性进行额外的调整
        if regime == MarketRegime.BEARISH:
            self.logger.info("黄金三角策略: 熊市环境，仅保留强势买入信号")
            # 在熊市中，更严格地筛选买入信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] > 0:
                    # 检查三线差距，只有差距明显时才买入
                    ma5 = adjusted_signals['ma5'].iloc[i]
                    ma10 = adjusted_signals['ma10'].iloc[i]
                    ma20 = adjusted_signals['ma20'].iloc[i]
                    
                    # 计算均线间距比例
                    ma5_10_gap = adjusted_signals['ma5_10_gap'].iloc[i]
                    ma10_20_gap = adjusted_signals['ma10_20_gap'].iloc[i]
                    
                    # 在熊市中，提高买入门槛
                    if ma5_10_gap < 1.0 or ma10_20_gap < 1.0:  # 提高至1.0%的间距
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
                    # 熊市中需要额外检查RSI不超过60
                    if adjusted_signals['rsi'].iloc[i] > 60:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                        
        elif regime == MarketRegime.BULLISH:
            self.logger.info("黄金三角策略: 牛市环境，减少卖出信号")
            # 在牛市中，减少卖出信号
            for i in range(len(adjusted_signals)):
                if adjusted_signals['signal'].iloc[i] < 0:
                    # 只有当中期均线明显低于长期均线时才产生卖出信号
                    ma10_20_gap = adjusted_signals['ma10_20_gap'].iloc[i]
                    
                    # 在牛市中，卖出需要更明显的下跌信号
                    if ma10_20_gap > -1.0:  # 间距需要至少-1.0%
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
                    
                    # 牛市中，RSI低于40时不要卖出
                    if adjusted_signals['rsi'].iloc[i] < 40:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 0
        
        elif regime == MarketRegime.RANGING:
            self.logger.info("黄金三角策略: 震荡市场，优化区间交易")
            for i in range(len(adjusted_signals)):
                # 在震荡市场中，增强在支撑位附近买入和阻力位附近卖出
                sr_position = adjusted_signals['sr_position'].iloc[i]
                
                if sr_position < 0.2 and adjusted_signals['signal'].iloc[i] == 0:
                    # 接近支撑位考虑买入
                    if adjusted_signals['rsi'].iloc[i] < 40:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = 1
                
                elif sr_position > 0.8 and adjusted_signals['signal'].iloc[i] == 0:
                    # 接近阻力位考虑卖出
                    if adjusted_signals['rsi'].iloc[i] > 60:
                        adjusted_signals.loc[adjusted_signals.index[i], 'signal'] = -1
        
        return adjusted_signals
        
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
            'ma5': result.get('ma5', pd.Series()),
            'ma10': result.get('ma10', pd.Series()),
            'ma20': result.get('ma20', pd.Series()),
            'ma100': result.get('ma100', pd.Series()),
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
            'ma5': {
                'name': '短期均线',
                'description': f"{self.parameters['short_period']}日简单移动平均线",
                'type': 'trend',
                'time_scale': 'short',
                'min_value': None,
                'max_value': None
            },
            'ma10': {
                'name': '中期均线',
                'description': f"{self.parameters['mid_period']}日简单移动平均线",
                'type': 'trend',
                'time_scale': 'medium',
                'min_value': None,
                'max_value': None
            },
            'ma20': {
                'name': '长期均线',
                'description': f"{self.parameters['long_period']}日简单移动平均线",
                'type': 'trend', 
                'time_scale': 'medium',
                'min_value': None,
                'max_value': None
            },
            'ma100': {
                'name': '趋势均线',
                'description': f"{self.parameters['trend_period']}日简单移动平均线",
                'type': 'trend',
                'time_scale': 'long',
                'min_value': None,
                'max_value': None
            },
            'sr_position': {
                'name': '支撑阻力位置',
                'description': '价格在支撑位和阻力位之间的相对位置',
                'type': 'position',
                'time_scale': 'medium',
                'min_value': 0,
                'max_value': 1
            },
            'rsi': {
                'name': 'RSI',
                'description': '14日相对强弱指标',
                'type': 'oscillator',
                'time_scale': 'medium',
                'min_value': 0,
                'max_value': 100
            },
            'volume_ratio': {
                'name': '成交量比率',
                'description': '当前成交量与10日平均成交量的比率',
                'type': 'volume',
                'time_scale': 'short',
                'min_value': 0,
                'max_value': None
            },
            'atr14': {
                'name': 'ATR',
                'description': '14日平均真实波幅',
                'type': 'volatility',
                'time_scale': 'medium',
                'min_value': 0,
                'max_value': None
            }
        }
        
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格 - 使用ATR方法
        
        参数:
            data: 市场数据
            entry_price: 入场价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            止损价格
        """
        if data is None or data.empty or len(data) < 14 or 'atr14' not in data.columns:
            return super().get_stop_loss(data, entry_price, position)
            
        # 获取最近的ATR值
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
            
        # 使用ATR倍数作为止损距离
        if position > 0:
            return entry_price - atr_multiplier * atr
        elif position < 0:
            return entry_price + atr_multiplier * atr
            
        return 0.0
        
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float, 
                              stop_loss: float, position: int) -> float:
        """
        实现追踪止损
        
        参数:
            data: 市场数据
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向(1=多, -1=空)
            
        返回:
            调整后的止损价格
        """
        # 获取市场环境
        regime = self.get_market_regime(data)
        
        # 计算利润
        if position > 0:
            profit_pct = (current_price / entry_price - 1) * 100
        else:
            profit_pct = (1 - current_price / entry_price) * 100
            
        # 根据市场环境和利润调整止损
        if profit_pct > 5:  # 利润超过5%
            atr = data['atr14'].iloc[-1] if 'atr14' in data.columns else (current_price * 0.02)
            
            # 在牛市中多头或熊市中空头，使用更激进的追踪止损
            if (regime == MarketRegime.BULLISH and position > 0) or \
               (regime == MarketRegime.BEARISH and position < 0):
                # 更紧密的追踪止损
                new_stop = current_price - (position * atr * 1.5)
                
                # 只有当新止损更有利时才更新
                if (position > 0 and new_stop > stop_loss) or \
                   (position < 0 and new_stop < stop_loss):
                    return new_stop
                    
            # 其他市场环境使用标准追踪止损
            elif profit_pct > 10:  # 利润超过10%，开始使用追踪止损
                new_stop = current_price - (position * atr * 2)
                
                # 只有当新止损更有利时才更新
                if (position > 0 and new_stop > stop_loss) or \
                   (position < 0 and new_stop < stop_loss):
                    return new_stop
                
        return stop_loss
        
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
        
        # 获取ATR
        atr = data['atr14'].iloc[-1] if 'atr14' in data.columns else (entry_price * 0.02)
        
        # 基础止盈倍数
        tp_multiplier = 3.0
        
        # 根据市场环境调整止盈倍数
        if regime == MarketRegime.BULLISH and position > 0:
            # 牛市中的多头，使用更大的止盈目标
            tp_multiplier = 4.0
        elif regime == MarketRegime.BEARISH and position < 0:
            # 熊市中的空头，使用更大的止盈目标
            tp_multiplier = 4.0
        
        # 计算止盈价格
        if position > 0:
            return entry_price + atr * tp_multiplier
        elif position < 0:
            return entry_price - atr * tp_multiplier
            
        return 0.0

    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        判断当前市场环境
        
        参数:
            data: 市场数据
            
        返回:
            市场环境类型: MarketRegime枚举类型
        """
        if data is None or data.empty or len(data) < 20:
            return MarketRegime.UNKNOWN
            
        # 计算20日收益率标准差作为波动率指标
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # 计算20日方向性指标
            direction = abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / (
                data['close'].iloc[-20:].max() - data['close'].iloc[-20:].min())
                
            # 根据波动率和方向性判断市场环境
            if volatility > 0.02:  # 高波动
                return MarketRegime.VOLATILE
            elif direction > 0.6:  # 强方向性
                if data['close'].iloc[-1] > data['close'].iloc[-20]:
                    return MarketRegime.BULLISH
                else:
                    return MarketRegime.BEARISH
            else:
                return MarketRegime.RANGING
        
        return MarketRegime.UNKNOWN 