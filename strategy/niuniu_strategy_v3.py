import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import talib

from .strategy_base import Strategy, MarketRegime
from .signal_interface import SignalComponent, SignalMetadata, SignalType, SignalTimeframe

class NiuniuStrategyV3(Strategy):
    """
    牛牛策略 V3 - 基于趋势跟踪和动量突破的交易策略
    
    策略特点：
    1. 多周期趋势跟踪
    2. 动量突破确认
    3. 波动率自适应
    4. 动态仓位管理
    5. 市场环境自适应
    6. 智能止损止盈
    7. 牛线交易线交叉
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化牛牛策略
        
        参数:
            parameters: 策略参数字典，可包含以下键:
                # 趋势参数
                - fast_period: 快速均线周期，默认10
                - slow_period: 慢速均线周期，默认30
                
                # 动量参数
                - rsi_period: RSI周期，默认14
                - rsi_oversold: RSI超卖阈值，默认25
                - rsi_overbought: RSI超买阈值，默认75
                
                # 趋势强度参数
                - adx_period: ADX周期，默认14
                - adx_threshold: ADX阈值，默认20
                
                # 风险控制参数
                - stop_loss: 止损比例，默认0.08
                - take_profit: 止盈比例，默认0.25
                - trailing_stop: 追踪止损比例，默认0.05
                
                # 仓位管理参数
                - max_position_size: 最大仓位，默认0.8
                - min_position_size: 最小仓位，默认0.4
                
                # 交易频率控制
                - min_hold_days: 最小持仓天数，默认3
                - max_trades_per_day: 每日最大交易次数，默认3
        """
        # 默认参数
        default_params = {
            # 趋势参数
            'fast_period': 10,      # 快速均线周期
            'slow_period': 30,      # 慢速均线周期
            
            # 动量参数
            'rsi_period': 14,       # RSI周期
            'rsi_oversold': 25,     # RSI超卖阈值
            'rsi_overbought': 75,   # RSI超买阈值
            
            # 趋势强度参数
            'adx_period': 14,       # ADX周期
            'adx_threshold': 20,    # ADX阈值
            
            # 风险控制参数
            'stop_loss': 0.08,      # 止损比例
            'take_profit': 0.25,    # 止盈比例
            'trailing_stop': 0.05,  # 追踪止损比例
            
            # 仓位管理参数
            'max_position_size': 0.8, # 最大仓位
            'min_position_size': 0.4, # 最小仓位
            
            # 交易频率控制
            'min_hold_days': 3,     # 最小持仓天数
            'max_trades_per_day': 3, # 每日最大交易次数
        }
        
        # 更新参数
        if parameters:
            default_params.update(parameters)
            
        # 初始化基类
        super().__init__("NiuniuV3", default_params)
        
        # 设置参数
        for key, value in default_params.items():
            setattr(self, key, value)
            
        self.logger = logging.getLogger(__name__)
        
        # 初始化交易记录
        self.trade_history = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        
    def extract_signal_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        提取并标准化策略的核心信号组件
        
        参数:
            data: 包含OHLCV和技术指标的DataFrame
            
        返回:
            字典，包含标准化后的信号组件
        """
        try:
            self.logger.info("开始提取信号组件")
            self.logger.info(f"输入数据列: {data.columns.tolist()}")
            
            # 计算技术指标
            df = self.calculate_indicators(data)
            self.logger.info(f"计算指标后的数据列: {df.columns.tolist()}")
            
            # 确保所有值都是数值类型
            components = {}
            for col in ['fast_ma', 'slow_ma', 'RSI', 'volume_ratio', 'volatility', 'ADX']:
                if col in df.columns:
                    self.logger.info(f"处理列 {col}, 类型: {df[col].dtype}")
                    components[col] = pd.to_numeric(df[col], errors='coerce')
                    self.logger.info(f"转换后的类型: {components[col].dtype}")
                else:
                    self.logger.warning(f"列 {col} 不在数据中")
            
            self.logger.info("信号组件提取完成")
            return components
            
        except Exception as e:
            self.logger.error(f"提取信号组件时出错: {str(e)}")
            self.logger.error(f"错误类型: {type(e)}")
            self.logger.error(f"错误详情: {e.__dict__ if hasattr(e, '__dict__') else 'No __dict__'}")
            raise

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            添加了信号列的DataFrame
        """
        try:
            # 确保数据包含必要的列
            required_columns = ['RSI', 'fast_ma', 'slow_ma', 'ADX']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"数据中缺少 {col} 列")
                    return data
                    
            # 初始化信号列
            data['signal'] = 0
            
            # 获取当前指标值
            current_rsi = float(data['RSI'].iloc[-1])
            current_fast_ma = float(data['fast_ma'].iloc[-1])
            current_slow_ma = float(data['slow_ma'].iloc[-1])
            current_adx = float(data['ADX'].iloc[-1])
            
            # 生成信号
            if current_rsi < self.rsi_oversold and current_fast_ma > current_slow_ma and current_adx > self.adx_threshold:
                data['signal'].iloc[-1] = 1
            elif current_rsi > self.rsi_overbought and current_fast_ma < current_slow_ma and current_adx > self.adx_threshold:
                data['signal'].iloc[-1] = -1
                
            return data
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            return data

    def get_signal_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取信号元数据
        
        返回:
            包含信号元数据的字典
        """
        return {
            'trend': {
                'name': '趋势强度',
                'description': '趋势强度指标',
                'signal_type': SignalType.TREND,
                'timeframe': SignalTimeframe.DAILY,
                'weight': 0.4
            },
            'momentum': {
                'name': '动量指标',
                'description': '动量指标',
                'signal_type': SignalType.MOMENTUM,
                'timeframe': SignalTimeframe.DAILY,
                'weight': 0.3
            },
            'volume': {
                'name': '成交量',
                'description': '成交量指标',
                'signal_type': SignalType.VOLATILITY,
                'timeframe': SignalTimeframe.DAILY,
                'weight': 0.2
            },
            'volatility': {
                'name': '波动率',
                'description': '波动率指标',
                'signal_type': SignalType.VOLATILITY,
                'timeframe': SignalTimeframe.DAILY,
                'weight': 0.1
            }
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        参数:
            df: 包含OHLCV数据的DataFrame
            
        返回:
            添加了技术指标的DataFrame
        """
        try:
            # 确保列名小写
            df = df.copy()
            df.columns = df.columns.str.lower()
            
            # 确保数据是数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算移动平均线
            df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
            
            # 计算RSI
            df['RSI'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            # 计算ADX
            df['ADX'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=self.adx_period)
            
            # 计算成交量比率
            volume_ma = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / volume_ma
            
            # 计算波动率
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # 计算ATR
            df['ATR'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # 填充NaN值
            df = df.fillna(method='ffill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise

    def _apply_trade_frequency_limits(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用交易频率限制
        
        参数:
            data: 包含信号的DataFrame
            
        返回:
            应用了交易频率限制的DataFrame
        """
        try:
            df = data.copy()
            
            # 初始化每日交易计数
            daily_trades = {}
            
            # 遍历每个交易日
            for date in df.index.date:
                if date not in daily_trades:
                    daily_trades[date] = 0
                
                # 获取当天的信号
                day_signals = df[df.index.date == date]
                
                # 如果当天已有最大交易次数，将后续信号设为0
                if daily_trades[date] >= self.max_trades_per_day:
                    df.loc[day_signals.index, 'signal'] = 0
                else:
                    # 更新交易计数
                    daily_trades[date] += len(day_signals[day_signals['signal'] != 0])
            
            return df
            
        except Exception as e:
            self.logger.error(f"应用交易频率限制时出错: {str(e)}")
            raise

    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """
        回测策略
        
        参数:
            data: 历史数据
            initial_capital: 初始资金
            
        返回:
            回测结果字典
        """
        try:
            # 生成信号
            df = self.generate_signals(data)
            
            # 初始化回测结果
            positions = pd.Series(index=df.index, data=0.0)  # 使用float类型
            returns = pd.Series(index=df.index, data=0.0)    # 使用float类型
            equity = pd.Series(index=df.index, data=float(initial_capital))  # 使用float类型
            drawdown = pd.Series(index=df.index, data=0.0)   # 使用float类型
            
            # 记录交易
            trades = []
            current_position = 0.0  # 使用float类型
            entry_price = 0.0       # 使用float类型
            entry_date = None
            stop_loss = 0.0         # 使用float类型
            take_profit = 0.0       # 使用float类型
            trailing_stop = 0.0     # 使用float类型
            
            # 计算动态仓位大小
            def calculate_position_size(price: float, volatility: float) -> float:
                """根据波动率动态调整仓位大小"""
                base_size = float(self.max_position_size)  # 转换为float
                volatility_factor = 1.0 - (volatility / 0.1)  # 使用float
                return max(float(self.min_position_size), 
                         min(float(self.max_position_size), 
                             base_size * volatility_factor))
            
            # 模拟交易
            for i in range(1, len(df)):
                current_price = float(df['close'].iloc[i])  # 转换为float
                current_date = df.index[i]
                current_volatility = float(df['volatility'].iloc[i])  # 转换为float
                
                # 更新持仓
                if df['signal'].iloc[i] == 1 and current_position <= 0:
                    # 开多仓
                    position_size = calculate_position_size(current_price, current_volatility)
                    current_position = position_size
                    entry_price = current_price
                    entry_date = current_date
                    stop_loss = entry_price * (1.0 - self.stop_loss)
                    take_profit = entry_price * (1.0 + self.take_profit)
                    trailing_stop = entry_price * (1.0 - self.trailing_stop)
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': current_position,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                elif df['signal'].iloc[i] == -1 and current_position >= 0:
                    # 开空仓
                    position_size = calculate_position_size(current_price, current_volatility)
                    current_position = -position_size
                    entry_price = current_price
                    entry_date = current_date
                    stop_loss = entry_price * (1.0 + self.stop_loss)
                    take_profit = entry_price * (1.0 - self.take_profit)
                    trailing_stop = entry_price * (1.0 + self.trailing_stop)
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': current_position,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                
                # 更新止损止盈
                if current_position != 0:
                    if current_position > 0:  # 多仓
                        # 动态调整追踪止损
                        trailing_stop = max(trailing_stop, current_price * (1.0 - self.trailing_stop))
                        # 动态调整止盈
                        take_profit = max(take_profit, current_price * (1.0 + self.take_profit))
                        if current_price <= stop_loss or current_price >= take_profit:
                            current_position = 0.0
                    else:  # 空仓
                        # 动态调整追踪止损
                        trailing_stop = min(trailing_stop, current_price * (1.0 + self.trailing_stop))
                        # 动态调整止盈
                        take_profit = min(take_profit, current_price * (1.0 - self.take_profit))
                        if current_price >= stop_loss or current_price <= take_profit:
                            current_position = 0.0
                
                # 更新持仓
                positions.iloc[i] = current_position
                
                # 计算收益
                returns.iloc[i] = current_position * (current_price / float(df['close'].iloc[i-1]) - 1.0)
                
                # 更新权益
                equity.iloc[i] = equity.iloc[i-1] * (1.0 + returns.iloc[i])
                
                # 计算回撤
                drawdown.iloc[i] = (equity.iloc[i] / equity.cummax().iloc[i] - 1.0)
            
            # 计算回测指标
            total_return = (equity.iloc[-1] / initial_capital - 1.0)
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0.0
            max_drawdown = drawdown.min()
            
            # 计算交易统计
            winning_trades = len([t for t in trades if t['entry_price'] * (1.0 + t['position'] * self.take_profit) > t['entry_price']])
            losing_trades = len([t for t in trades if t['entry_price'] * (1.0 + t['position'] * self.stop_loss) < t['entry_price']])
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # 计算平均交易收益
            avg_trade_return = total_return / total_trades if total_trades > 0 else 0.0
            
            # 计算盈亏比
            profit_factor = abs(winning_trades * self.take_profit / (losing_trades * self.stop_loss)) if losing_trades > 0 else float('inf')
            
            # 计算平均交易持续时间
            avg_trade_duration = len(df) / total_trades if total_trades > 0 else 0.0
            
            return {
                'equity_curve': equity,
                'returns': returns,
                'drawdown_curve': drawdown,
                'signal': df['signal'],
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration,
                'final_capital': equity.iloc[-1],
                'initial_capital': initial_capital,
                'trades': trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_trades': total_trades,
                'avg_profit': self.take_profit * initial_capital,
                'avg_loss': self.stop_loss * initial_capital,
                'max_profit': self.take_profit * initial_capital,
                'max_loss': self.stop_loss * initial_capital
            }
            
        except Exception as e:
            self.logger.error(f"回测过程中出错: {str(e)}")
            raise

    def get_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        获取市场环境
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            市场环境 (MarketRegime枚举值)
        """
        try:
            # 确保数据包含必要的列
            required_columns = ['ADX', 'volatility']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"数据中缺少 {col} 列")
                    return MarketRegime.RANGING  # 默认返回震荡市场
                    
            # 获取当前指标值
            current_adx = float(data['ADX'].iloc[-1])
            current_volatility = float(data['volatility'].iloc[-1])
            
            # 判断市场环境
            if current_adx > self.adx_threshold:
                if current_volatility > 0.02:  # 高波动
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.BULLISH if data['close'].iloc[-1] > data['close'].iloc[-2] else MarketRegime.BEARISH
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"获取市场环境时出错: {str(e)}")
            return MarketRegime.RANGING  # 出错时返回震荡市场
            
    def get_position_size(self, data: pd.DataFrame, signal: int) -> float:
        """
        根据信号和市场环境确定仓位大小
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            signal: 交易信号 (1: 买入, -1: 卖出, 0: 无信号)
            
        返回:
            仓位大小 (0.0 - 1.0)
        """
        if signal == 0:
            return 0.0
            
        try:
            # 获取市场环境
            market_regime = self.get_market_regime(data)
            
            # 根据市场环境调整仓位
            if market_regime == MarketRegime.BULLISH:
                return float(self.max_position_size)
            elif market_regime == MarketRegime.BEARISH:
                return float(self.max_position_size) * 0.5
            elif market_regime == MarketRegime.VOLATILE:
                return float(self.min_position_size)
            else:  # RANGING
                return float(self.min_position_size) * 0.5
                
        except Exception as e:
            self.logger.error(f"计算仓位大小时出错: {str(e)}")
            return float(self.min_position_size)
            
    def should_trade(self, data: pd.DataFrame) -> bool:
        """
        判断是否应该交易
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            是否应该交易
        """
        try:
            # 检查每日交易次数限制
            current_date = data.index[-1].date()
            if self.last_trade_date == current_date:
                if self.daily_trade_count >= self.max_trades_per_day:
                    return False
            else:
                self.daily_trade_count = 0
                self.last_trade_date = current_date
                
            # 检查最小持仓时间
            if self.trade_history:
                last_trade = self.trade_history[-1]
                days_since_last_trade = (data.index[-1] - last_trade['date']).days
                if days_since_last_trade < self.min_hold_days:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"判断是否应该交易时出错: {str(e)}")
            return False
            
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        生成交易信号
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            交易信号 (1: 买入, -1: 卖出, 0: 无信号)
        """
        try:
            # 检查是否应该交易
            if not self.should_trade(data):
                return 0
                
            # 获取市场环境
            market_regime = self.get_market_regime(data)
            self.logger.info(f"当前市场环境: {market_regime}, 类型: {type(market_regime)}")
            
            # 确保数据包含必要的列
            required_columns = ['RSI', 'fast_ma', 'slow_ma', 'ADX']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"数据中缺少 {col} 列")
                    return 0
            
            # 获取当前指标值并转换为浮点数
            current_rsi = float(data['RSI'].iloc[-1])
            current_fast_ma = float(data['fast_ma'].iloc[-1])
            current_slow_ma = float(data['slow_ma'].iloc[-1])
            current_adx = float(data['ADX'].iloc[-1])
            
            self.logger.info(f"当前指标值 - RSI: {current_rsi}, Fast MA: {current_fast_ma}, Slow MA: {current_slow_ma}, ADX: {current_adx}")
            
            # 生成信号
            if market_regime == MarketRegime.BULLISH:
                # 上升趋势中的买入条件
                if (current_rsi < self.rsi_oversold and 
                    current_fast_ma > current_slow_ma and
                    current_adx > self.adx_threshold):
                    self.logger.info("生成买入信号: 上升趋势中的买入条件满足")
                    return 1
                # 上升趋势中的卖出条件
                elif (current_rsi > self.rsi_overbought and 
                      current_fast_ma < current_slow_ma and
                      current_adx > self.adx_threshold):
                    self.logger.info("生成卖出信号: 上升趋势中的卖出条件满足")
                    return -1
                    
            elif market_regime == MarketRegime.BEARISH:
                # 下降趋势中的买入条件
                if (current_rsi < self.rsi_oversold and 
                    current_fast_ma > current_slow_ma and
                    current_adx > self.adx_threshold):
                    self.logger.info("生成买入信号: 下降趋势中的买入条件满足")
                    return 1
                # 下降趋势中的卖出条件
                elif (current_rsi > self.rsi_overbought and 
                      current_fast_ma < current_slow_ma and
                      current_adx > self.adx_threshold):
                    self.logger.info("生成卖出信号: 下降趋势中的卖出条件满足")
                    return -1
                    
            elif market_regime == MarketRegime.RANGING:
                # 震荡市场中的交易条件
                if current_rsi < self.rsi_oversold:
                    self.logger.info("生成买入信号: 震荡市场中的超卖条件满足")
                    return 1
                elif current_rsi > self.rsi_overbought:
                    self.logger.info("生成卖出信号: 震荡市场中的超买条件满足")
                    return -1
                    
            self.logger.info("无交易信号生成")
            return 0
            
        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}")
            self.logger.error(f"错误类型: {type(e)}")
            self.logger.error(f"错误详情: {e.__dict__ if hasattr(e, '__dict__') else 'No __dict__'}")
            return 0
            
    def get_stop_loss(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止损价格
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            entry_price: 入场价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            止损价格
        """
        try:
            # 使用ATR动态调整止损
            atr = data['atr'].iloc[-1]
            stop_distance = max(self.stop_loss * entry_price, atr * 2)
            
            if position == 1:  # 多头
                return entry_price - stop_distance
            elif position == -1:  # 空头
                return entry_price + stop_distance
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算止损价格时出错: {str(e)}")
            return 0.0
            
    def get_take_profit(self, data: pd.DataFrame, entry_price: float, position: int) -> float:
        """
        计算止盈价格
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            entry_price: 入场价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            止盈价格
        """
        try:
            # 使用ATR动态调整止盈
            atr = data['atr'].iloc[-1]
            profit_distance = max(self.take_profit * entry_price, atr * 4)
            
            if position == 1:  # 多头
                return entry_price + profit_distance
            elif position == -1:  # 空头
                return entry_price - profit_distance
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算止盈价格时出错: {str(e)}")
            return 0.0
            
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float, 
                              stop_loss: float, position: int) -> float:
        """
        判断是否应该调整止损价格
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            新的止损价格，如果不需要调整则返回原止损价格
        """
        try:
            # 计算当前收益
            if position == 1:  # 多头
                profit = (current_price - stop_loss) / stop_loss
                if profit > self.trailing_stop:
                    # 使用ATR动态调整追踪止损
                    atr = data['atr'].iloc[-1]
                    new_stop = current_price - max(self.trailing_stop * current_price, atr * 2)
                    return max(new_stop, stop_loss)
            elif position == -1:  # 空头
                profit = (stop_loss - current_price) / stop_loss
                if profit > self.trailing_stop:
                    # 使用ATR动态调整追踪止损
                    atr = data['atr'].iloc[-1]
                    new_stop = current_price + max(self.trailing_stop * current_price, atr * 2)
                    return min(new_stop, stop_loss)
                    
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"判断是否应该调整止损价格时出错: {str(e)}")
            return stop_loss
            
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        返回:
            包含策略信息的字典
        """
        return {
            "name": self.name,
            "version": "3.0.0",
            "description": "牛牛策略V3，基于趋势跟踪和动量突破的交易策略",
            "parameters": self.parameters,
            "author": "System",
            "creation_date": "2024-03-14",
            "last_modified_date": "2024-03-14",
            "risk_level": "medium",
            "performance_metrics": {
                "sharpe_ratio": None,
                "max_drawdown": None,
                "win_rate": None
            },
            "suitable_market_regimes": ["trending", "ranging"],
            "tags": ["technical", "trend-following", "momentum"]
        } 