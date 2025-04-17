import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import talib

from .strategy_base import Strategy, MarketRegime
from .signal_interface import SignalComponent, SignalMetadata, SignalType, SignalTimeframe

class NiuniuStrategy(Strategy):
    """
    牛牛策略 - 基于趋势跟踪和动量突破的交易策略
    
    策略特点：
    1. 多周期趋势跟踪
    2. 动量突破确认
    3. 波动率自适应
    4. 动态仓位管理
    """
    
    def __init__(self):
        # 初始化基类
        parameters = {
            'fast_period': 30,      # 快速周期
            'slow_period': 43,      # 慢速周期
            'rsi_period': 3,        # RSI周期
            'macd_fast': 3,         # MACD快线
            'macd_slow': 19,         # MACD慢线
            'macd_signal': 9,        # MACD信号线
            'adx_period': 3,         # ADX周期
            'adx_threshold': 10,     # ADX阈值
            'atr_period': 8,         # ATR周期
            'atr_multiplier': 0.5,   # ATR乘数
            'volume_threshold': 0.5, # 成交量阈值
            'signal_threshold': 0.5, # 信号阈值
            'profit_target': 0.15,   # 止盈目标
            'stop_loss': -0.15,      # 止损目标
            'trailing_stop': 0.03,   # 追踪止损
            'volatility_threshold': 0.02, # 波动率阈值
            'trend_strength_threshold': 0.6, # 趋势强度阈值
            'market_regime_window': 20, # 市场环境判断窗口
            'max_position_size': 1.0, # 最大仓位
            'min_position_size': 0.2, # 最小仓位
            'position_sizing_factor': 0.5, # 仓位调整因子
            'max_positions': 3,      # 最大持仓数
            'min_trade_interval': 5   # 最小交易间隔
        }
        super().__init__(name="Niuniu", parameters=parameters)
        
        # 设置参数
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, data):
        """计算技术指标"""
        try:
            df = data.copy()
            
            # 计算移动平均线
            df['fast_ma'] = talib.SMA(df['close'], timeperiod=self.fast_period)
            df['slow_ma'] = talib.SMA(df['close'], timeperiod=self.slow_period)
            
            # 计算RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # 计算MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # 计算ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
            
            # 计算ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
            
            # 计算成交量指标
            df['volume_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise
    
    def determine_market_regime(self, data):
        """判断市场环境"""
        try:
            df = data.copy()
            
            # 计算趋势强度
            trend_strength = df['adx'] / 100.0
            
            # 计算波动率
            volatility = df['atr'] / df['close']
            
            # 计算成交量趋势
            volume_trend = df['volume_ratio'] - 1
            
            # 判断市场环境
            if trend_strength.iloc[-1] > self.trend_strength_threshold:
                if df['close'].iloc[-1] > df['fast_ma'].iloc[-1]:
                    return MarketRegime.BULLISH
                else:
                    return MarketRegime.BEARISH
            elif volatility.iloc[-1] > self.volatility_threshold:
                return MarketRegime.VOLATILE
            elif volume_trend.iloc[-1] > self.volume_threshold:
                return MarketRegime.BULLISH
            elif volume_trend.iloc[-1] < -self.volume_threshold:
                return MarketRegime.BEARISH
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"判断市场环境时出错: {str(e)}")
            return MarketRegime.UNKNOWN
    
    def calculate_position_size(self, data: pd.DataFrame, market_regime: str, current_price: float, atr: float) -> float:
        """计算仓位大小"""
        try:
            # 基础仓位
            base_size = 1.0
            
            # 根据市场环境调整
            if market_regime == MarketRegime.BULLISH:
                base_size *= 1.2
            elif market_regime == MarketRegime.BEARISH:
                base_size *= 0.8
            elif market_regime == MarketRegime.VOLATILE:
                base_size *= 0.6
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                base_size *= 0.9
            
            # 根据ATR调整
            volatility_factor = 1.0 / (1.0 + atr)
            base_size *= volatility_factor
            
            # 限制仓位范围
            position_size = max(self.min_position_size, min(self.max_position_size, base_size))
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算仓位大小时出错: {str(e)}")
            return self.min_position_size
    
    def generate_signals(self, data):
        """生成交易信号"""
        try:
            df = data.copy()
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 判断市场环境
            market_regime = self.determine_market_regime(df)
            
            # 生成信号
            signals = pd.Series(index=df.index, data=0)
            
            # 趋势信号
            trend_signal = np.where(
                (df['fast_ma'] > df['slow_ma']) & 
                (df['rsi'] > 50) & 
                (df['macd'] > df['macd_signal']),
                1,
                np.where(
                    (df['fast_ma'] < df['slow_ma']) & 
                    (df['rsi'] < 50) & 
                    (df['macd'] < df['macd_signal']),
                    -1,
                    0
                )
            )
            
            # 动量信号
            momentum_signal = np.where(
                (df['close'] > df['fast_ma']) & 
                (df['volume_ratio'] > 1.2),
                1,
                np.where(
                    (df['close'] < df['fast_ma']) & 
                    (df['volume_ratio'] < 0.8),
                    -1,
                    0
                )
            )
            
            # 综合信号
            signals = np.where(
                (trend_signal == 1) & (momentum_signal == 1),
                1,
                np.where(
                    (trend_signal == -1) & (momentum_signal == -1),
                    -1,
                    0
                )
            )
            
            return pd.Series(signals, index=df.index)
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            raise
    
    def backtest(self, data):
        """回测策略"""
        try:
            df = data.copy()
            
            # 生成信号
            signals = self.generate_signals(df)
            
            # 初始化回测结果
            positions = pd.Series(index=df.index, data=0)
            returns = pd.Series(index=df.index, data=0)
            equity = pd.Series(index=df.index, data=1.0)
            
            # 模拟交易
            for i in range(1, len(df)):
                # 获取当前信号
                current_signal = signals.iloc[i]
                
                # 更新持仓
                if current_signal == 1 and positions.iloc[i-1] <= 0:
                    positions.iloc[i] = 1
                elif current_signal == -1 and positions.iloc[i-1] >= 0:
                    positions.iloc[i] = -1
                else:
                    positions.iloc[i] = positions.iloc[i-1]
                
                # 计算收益
                returns.iloc[i] = positions.iloc[i] * (df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
                
                # 更新权益
                equity.iloc[i] = equity.iloc[i-1] * (1 + returns.iloc[i])
            
            # 计算回测指标
            total_return = equity.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            max_drawdown = (equity / equity.cummax() - 1).min()
            
            return {
                'positions': positions,
                'returns': returns,
                'equity': equity,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"回测策略时出错: {str(e)}")
            raise
    
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """获取市场环境"""
        try:
            return self.determine_market_regime(data)
        except Exception as e:
            self.logger.error(f"获取市场环境时出错: {str(e)}")
            return MarketRegime.UNKNOWN
    
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """获取仓位大小"""
        try:
            market_regime = self.get_market_regime(data)
            current_price = data['close'].iloc[-1]
            atr = data['atr'].iloc[-1]
            
            return self.calculate_position_size(data, market_regime, current_price, atr)
        except Exception as e:
            self.logger.error(f"获取仓位大小时出错: {str(e)}")
            return self.min_position_size
    
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """获取止损价格"""
        try:
            atr = data['atr'].iloc[-1]
            if position > 0:
                return entry_price * (1 - self.stop_loss)
            else:
                return entry_price * (1 + self.stop_loss)
        except Exception as e:
            self.logger.error(f"获取止损价格时出错: {str(e)}")
            return entry_price
    
    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """获取止盈价格"""
        try:
            if position > 0:
                return entry_price * (1 + self.profit_target)
            else:
                return entry_price * (1 - self.profit_target)
        except Exception as e:
            self.logger.error(f"获取止盈价格时出错: {str(e)}")
            return entry_price
    
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float, 
                          stop_loss: float, position: int) -> float:
        """判断是否需要调整止损"""
        try:
            if position > 0:
                # 多头持仓
                if current_price > stop_loss * (1 + self.trailing_stop):
                    return current_price * (1 - self.trailing_stop)
            else:
                # 空头持仓
                if current_price < stop_loss * (1 - self.trailing_stop):
                    return current_price * (1 + self.trailing_stop)
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"判断是否需要调整止损时出错: {str(e)}")
            return stop_loss
    
    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """优化策略参数"""
        try:
            # 定义参数网格
            param_grid = {
                'fast_period': [20, 30, 40],
                'slow_period': [40, 60, 80],
                'rsi_period': [3, 5, 7],
                'macd_fast': [3, 5, 7],
                'macd_slow': [15, 20, 25],
                'macd_signal': [7, 9, 11],
                'adx_period': [3, 5, 7],
                'adx_threshold': [10, 15, 20],
                'atr_period': [5, 8, 11],
                'atr_multiplier': [0.3, 0.5, 0.7],
                'volume_threshold': [0.3, 0.5, 0.7],
                'signal_threshold': [0.3, 0.5, 0.7],
                'profit_target': [0.1, 0.15, 0.2],
                'stop_loss': [-0.15, -0.2, -0.25],
                'trailing_stop': [0.02, 0.03, 0.04],
                'volatility_threshold': [0.01, 0.02, 0.03],
                'trend_strength_threshold': [0.5, 0.6, 0.7],
                'market_regime_window': [15, 20, 25],
                'max_position_size': [0.8, 1.0, 1.2],
                'min_position_size': [0.1, 0.2, 0.3],
                'position_sizing_factor': [0.3, 0.5, 0.7],
                'max_positions': [2, 3, 4],
                'min_trade_interval': [3, 5, 7]
            }
            
            # 初始化最佳参数
            best_params = {}
            best_sharpe = -np.inf
            
            # 随机搜索
            for _ in range(100):  # 尝试100次随机组合
                # 随机选择参数
                current_params = {}
                for param, values in param_grid.items():
                    current_params[param] = np.random.choice(values)
                
                # 检查参数有效性
                if current_params['slow_period'] <= current_params['fast_period']:
                    continue
                if current_params['macd_slow'] <= current_params['macd_fast']:
                    continue
                if current_params['min_position_size'] >= current_params['max_position_size']:
                    continue
                
                # 更新参数
                for key, value in current_params.items():
                    setattr(self, key, value)
                
                # 回测
                results = self.backtest(data)
                
                # 更新最佳参数
                if results['sharpe_ratio'] > best_sharpe:
                    best_sharpe = results['sharpe_ratio']
                    best_params = current_params
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"优化策略参数时出错: {str(e)}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        try:
            return {
                'name': self.name,
                'version': self.version,
                'parameters': self.parameters,
                'description': self.__doc__
            }
        except Exception as e:
            self.logger.error(f"获取策略信息时出错: {str(e)}")
            return {}
    
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """提取信号组件"""
        try:
            df = data.copy()
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 提取信号组件
            components = {
                'trend': pd.Series(
                    np.where(df['fast_ma'] > df['slow_ma'], 1, -1),
                    index=df.index
                ),
                'momentum': pd.Series(
                    np.where(df['rsi'] > 50, 1, -1),
                    index=df.index
                ),
                'macd': pd.Series(
                    np.where(df['macd'] > df['macd_signal'], 1, -1),
                    index=df.index
                ),
                'volume': pd.Series(
                    np.where(df['volume_ratio'] > 1, 1, -1),
                    index=df.index
                )
            }
            
            return components
            
        except Exception as e:
            self.logger.error(f"提取信号组件时出错: {str(e)}")
            return {}
    
    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取信号元数据"""
        try:
            return {
                'trend': {
                    'name': '趋势信号',
                    'description': '基于快速和慢速移动平均线的趋势信号',
                    'signal_type': SignalType.TREND,
                    'timeframe': SignalTimeframe.DAILY,
                    'weight': 1.0
                },
                'momentum': {
                    'name': '动量信号',
                    'description': '基于RSI的动量信号',
                    'signal_type': SignalType.MOMENTUM,
                    'timeframe': SignalTimeframe.DAILY,
                    'weight': 1.0
                },
                'macd': {
                    'name': 'MACD信号',
                    'description': '基于MACD的趋势信号',
                    'signal_type': SignalType.TREND,
                    'timeframe': SignalTimeframe.DAILY,
                    'weight': 1.0
                },
                'volume': {
                    'name': '成交量信号',
                    'description': '基于成交量的趋势信号',
                    'signal_type': SignalType.TREND,
                    'timeframe': SignalTimeframe.DAILY,
                    'weight': 1.0
                }
            }
        except Exception as e:
            self.logger.error(f"获取信号元数据时出错: {str(e)}")
            return {} 