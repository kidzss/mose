import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from strategy.strategy_base import Strategy

class NiuniuStrategy(Strategy):
    """
    牛牛策略 (Niuniu Strategy)
    
    该策略基于牛线（主力成本线）和交易线的交叉关系生成买卖信号。
    
    买入条件: 牛线上穿交易线
    卖出条件: 牛线下穿交易线
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        初始化牛牛策略
        
        参数:
            parameters: 策略参数字典，可包含以下键:
                - weights_window: 权重窗口大小，默认20
                - trade_line_window: 交易线窗口大小，默认2
                - stop_loss_pct: 止损百分比，默认5.0
                - take_profit_pct: 止盈百分比，默认15.0
                - max_position_size: 最大仓位大小，默认0.2
        """
        default_params = {
            'weights_window': 20,
            'trade_line_window': 2,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0,
            'max_position_size': 0.2
        }
        
        # 合并默认参数和传入的参数
        if parameters:
            default_params.update(parameters)
            
        super().__init__("NiuniuStrategy", default_params)
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算牛牛策略所需的技术指标
        
        参数:
            data: 包含OHLCV数据的DataFrame
            
        返回:
            添加了技术指标的DataFrame
        """
        if data is None or data.empty:
            self.logger.warning("数据为空，无法计算指标")
            return data
            
        # 检查必要的列是否存在
        required_columns = ['close', 'low', 'open', 'high']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data
        
        try:
            # 获取参数
            weights_window = self.parameters['weights_window']
            trade_line_window = self.parameters['trade_line_window']
            
            # 计算 MID 值
            data['MID'] = (3 * data['close'] + data['low'] + data['open'] + data['high']) / 6

            # 计算牛线（主力成本线）
            weights = np.arange(weights_window, 0, -1)  # 权重为weights_window到1
            
            # 确保数据足够计算牛线
            if len(data) < weights_window:
                self.logger.warning(f"数据行数 ({len(data)}) 不足以计算牛线 (需要至少{weights_window}行)")
                return data
                
            # 使用向量化操作计算加权MID
            weighted_mid = pd.DataFrame({
                f'weighted_mid_{i}': data['MID'].shift(i) * weights[i] for i in range(weights_window)
            })
            data['Bull_Line'] = weighted_mid.sum(axis=1) / weights.sum()

            # 计算买卖线
            data['Trade_Line'] = data['Bull_Line'].rolling(window=trade_line_window).mean()
            
            return data
        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        根据牛牛策略生成交易信号
        
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
        required_columns = ['Bull_Line', 'Trade_Line']
        for col in required_columns:
            if col not in data.columns:
                self.logger.warning(f"数据中缺少 {col} 列")
                return data
        
        try:
            # 初始化信号列
            data['signal'] = 0
            
            # 生成买卖信号
            buy_signal = (data['Bull_Line'] > data['Trade_Line']) & (data['Bull_Line'].shift(1) <= data['Trade_Line'].shift(1))
            sell_signal = (data['Trade_Line'] > data['Bull_Line']) & (data['Trade_Line'].shift(1) <= data['Bull_Line'].shift(1))
            
            # 设置信号
            data.loc[buy_signal, 'signal'] = 1
            data.loc[sell_signal, 'signal'] = -1
            
            return data
        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}")
            return data
    
    def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        根据牛线和交易线判断当前市场环境
        
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
            if 'Bull_Line' not in recent_data.columns or 'Trade_Line' not in recent_data.columns:
                return "unknown"
            
            # 计算牛线和交易线的差值
            bull_trade_diff = recent_data['Bull_Line'] - recent_data['Trade_Line']
            
            # 计算收盘价的波动率
            volatility = recent_data['close'].pct_change().std() * 100 if 'close' in recent_data.columns else 1.5
            
            # 计算牛线的斜率
            bull_slope = (recent_data['Bull_Line'].iloc[-1] - recent_data['Bull_Line'].iloc[0]) / len(recent_data)
            
            # 判断市场环境
            if bull_slope > 0.5 and bull_trade_diff.mean() > 0:
                return "bullish"
            elif bull_slope < -0.5 and bull_trade_diff.mean() < 0:
                return "bearish"
            elif volatility > 2.5:
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
            high = recent_data['high'] if 'high' in recent_data.columns else recent_data['close'] * 1.01
            low = recent_data['low'] if 'low' in recent_data.columns else recent_data['close'] * 0.99
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
            "description": "牛牛策略，基于牛线（主力成本线）和交易线的交叉关系生成买卖信号",
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
            "suitable_market_regimes": ["bullish", "sideways"],
            "tags": ["technical", "trend-following", "cost-line"]
        } 