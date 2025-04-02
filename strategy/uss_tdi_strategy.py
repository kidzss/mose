import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from strategy.strategy_base import Strategy

class TDIStrategy(Strategy):
    """
    TDI (Traders Dynamic Index) 策略
    
    该策略基于RSI和其移动平均线的交叉关系生成买卖信号。
    
    买入条件: 短期MA下穿长期MA，且前一天短期MA上穿长期MA
    卖出条件: 短期MA上穿长期MA，且前一天短期MA下穿长期MA
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化TDI策略
        
        参数:
            parameters: 策略参数字典，可包含以下键:
                - rsi_period: RSI计算周期，默认14
                - ma1_period: 长期MA周期，默认34
                - ma2_period: 短期MA周期，默认7
                - stop_loss_pct: 止损百分比，默认5.0
                - take_profit_pct: 止盈百分比，默认15.0
                - max_position_size: 最大仓位大小，默认0.2
        """
        default_params = {
            'rsi_period': 14,
            'ma1_period': 34,
            'ma2_period': 7,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0,
            'max_position_size': 0.2
        }
        
        # 合并默认参数和传入的参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__("TDIStrategy", default_params)
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算TDI策略所需的技术指标
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了技术指标的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法计算指标")
            return data
            
        # 检查必要的列是否存在
        if 'close' not in data.columns:
            self.logger.warning("数据中缺少 close 列")
            return data
        
        try:
            # 获取参数
            rsi_period = self.parameters['rsi_period']
            ma1_period = self.parameters['ma1_period']
            ma2_period = self.parameters['ma2_period']
            
            # 计算RSI
            close_diff = data['close'].diff()
            gain = close_diff.clip(lower=0)
            loss = -close_diff.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # 避免除以零
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算RSI的移动平均线
            data['MA1'] = data['RSI'].rolling(window=ma1_period).mean()  # 长期MA
            data['MA2'] = data['RSI'].rolling(window=ma2_period).mean()  # 短期MA
            
            # 计算趋势线
            data['TREND'] = data['RSI'].rolling(window=ma1_period).mean()
            
            return data
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据TDI策略生成交易信号
        
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
            
        # 检查必要的列是否存在
        required_columns = ['MA1', 'MA2']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data
        
        try:
            # 初始化信号列
            data['signal'] = 0
            
            # 计算交叉条件
            data['MA2_below_MA1'] = data['MA2'] < data['MA1']
            
            # 生成买卖信号
            for i in range(1, len(data)):
                # 买入信号: 短期MA下穿长期MA，且前一天短期MA上穿长期MA
                if data['MA2_below_MA1'].iloc[i] and not data['MA2_below_MA1'].iloc[i-1]:
                    data.loc[data.index[i], 'signal'] = 1
                
                # 卖出信号: 短期MA上穿长期MA，且前一天短期MA下穿长期MA
                elif not data['MA2_below_MA1'].iloc[i] and data['MA2_below_MA1'].iloc[i-1]:
                    data.loc[data.index[i], 'signal'] = -1
            
            return data
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data
    
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        根据RSI和其移动平均线判断当前市场环境
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            
        返回:
            市场环境类型: "bullish", "bearish", "sideways", "volatile"
        """
        if data is None or data.empty:
            return "unknown"
            
        try:
            # 获取最近的数据点
            recent_data = data.iloc[-20:]
            
            # 检查必要的列是否存在
            if 'RSI' not in recent_data.columns or 'MA1' not in recent_data.columns or 'MA2' not in recent_data.columns:
                return "unknown"
            
            # 计算RSI的平均值和标准差
            avg_rsi = recent_data['RSI'].mean()
            std_rsi = recent_data['RSI'].std()
            
            # 计算收盘价的波动率
            volatility = recent_data['close'].pct_change().std() * 100 if 'close' in recent_data.columns else 1.5
            
            # 判断市场环境
            if avg_rsi > 60 and recent_data['MA2'].iloc[-1] > recent_data['MA1'].iloc[-1]:
                return "bullish"
            elif avg_rsi < 40 and recent_data['MA2'].iloc[-1] < recent_data['MA1'].iloc[-1]:
                return "bearish"
            elif volatility > 2.5 or std_rsi > 10:
                return "volatile"
            else:
                return "sideways"
        except Exception as e:
            self.logger.error(f"判断市场环境时出错: {e}")
            return "normal"
    
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
            
        max_position = self.parameters.get('max_position_size', 0.2)
        
        try:
            # 获取市场环境
            market_regime = self.get_market_regime(data)
            
            # 根据市场环境调整仓位
            if market_regime == "bullish":
                return max_position
            elif market_regime == "bearish":
                return max_position * 0.5 if signal == 1 else max_position
            elif market_regime == "volatile":
                return max_position * 0.7
            else:  # sideways or unknown
                return max_position * 0.8
        except Exception as e:
            self.logger.error(f"计算仓位大小时出错: {e}")
            return max_position * 0.5
    
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
        stop_loss_pct = self.parameters.get('stop_loss_pct', 5.0)
        
        if position == 1:  # 多头
            return entry_price * (1 - stop_loss_pct / 100)
        elif position == -1:  # 空头
            return entry_price * (1 + stop_loss_pct / 100)
        else:
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
        take_profit_pct = self.parameters.get('take_profit_pct', 15.0)
        
        if position == 1:  # 多头
            return entry_price * (1 + take_profit_pct / 100)
        elif position == -1:  # 空头
            return entry_price * (1 - take_profit_pct / 100)
        else:
            return 0.0
    
    def should_adjust_stop_loss(self, data: pd.DataFrame, current_price: float, 
                              stop_loss: float, position: int) -> float:
        """
        判断是否应该调整止损价格（追踪止损）
        
        参数:
            data: 包含价格数据和技术指标的DataFrame
            current_price: 当前价格
            stop_loss: 当前止损价格
            position: 仓位方向 (1: 多头, -1: 空头)
            
        返回:
            新的止损价格
        """
        if data is None or data.empty:
            return stop_loss
            
        try:
            # 获取最近的数据
            recent_data = data.iloc[-5:]
            
            # 计算ATR (Average True Range)
            high = recent_data['High'] if 'High' in recent_data.columns else recent_data['close'] * 1.01
            low = recent_data['Low'] if 'Low' in recent_data.columns else recent_data['close'] * 0.99
            close = recent_data['close'] if 'close' in recent_data.columns else recent_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.mean()
            
            # 根据ATR调整止损
            if position == 1:  # 多头
                new_stop = current_price - 2 * atr
                return max(new_stop, stop_loss)  # 只上调止损，不下调
            elif position == -1:  # 空头
                new_stop = current_price + 2 * atr
                return min(new_stop, stop_loss)  # 只下调止损，不上调
            else:
                return stop_loss
        except Exception as e:
            self.logger.error(f"调整止损时出错: {e}")
            return stop_loss
    
    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            data: 包含价格数据的DataFrame
            
        返回:
            优化后的参数字典
        """
        # 这里可以实现参数优化逻辑，如网格搜索或遗传算法
        # 简单起见，这里返回默认参数
        return self.parameters
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        返回:
            包含策略信息的字典
        """
        return {
            "name": self.name,
            "version": "1.0.0",
            "description": "TDI (Traders Dynamic Index) 策略，基于RSI和其移动平均线的交叉关系生成买卖信号",
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
            "suitable_market_regimes": ["bullish", "sideways", "volatile"],
            "tags": ["technical", "oscillator", "rsi", "trend-following"]
        } 